#!/usr/bin/env python3
"""Surface-aware Clio DSG evaluation — designed for 2D Place nodes.

Key insight: DSG Place nodes (layer 20, Place2dNodeAttributes) are 2D surface
patches on room surfaces (walls, floors, ceilings). They are NOT 3D object nodes.
Evaluation metrics are chosen accordingly:

  1. Surface Coverage (Precision/Recall/F1) — primary object metric
  2. Chamfer Distance — place mesh <-> GT bbox surface
  3. Object Proximity — distance from GT center to nearest place
  4. Room Assignment (with unannotated region filtering)
  5. Room 3D IoU — valid for rooms (large volumes vs surface OBB)
  6. Place Surface Type Classification — wall/floor/ceiling/other
  7. 3D IoU — kept with caveat (expected ~0 for 2D surfaces)

Usage:
    python3 eval_m/eval_clio_v2.py --dsg /tmp/clio_output/<scene>/dsg.json \\
        --rooms /data/YueChang/Clio/<scene>/rooms_<scene>.yaml \\
        --tasks /data/YueChang/Clio/<scene>/tasks_<scene>.yaml \\
        --output eval_m/<scene>_results.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dsg(path):
    """Load DSG JSON and return all data."""
    with open(path) as f:
        dsg = json.load(f)

    places = []
    objects = []
    rooms = []

    for node in dsg["nodes"]:
        layer = node["layer"]
        attrs = node.get("attributes", {})
        pos = attrs.get("position", [0, 0, 0])
        entry = {
            "id": node["id"],
            "layer": layer,
            "position": np.array(pos, dtype=np.float64),
            "name": attrs.get("name", ""),
            "semantic_label": attrs.get("semantic_label", -1),
            "boundary": np.array(attrs.get("boundary", []), dtype=np.float64),
            "ellipse_centroid": np.array(attrs.get("ellipse_centroid", [0, 0, 0]), dtype=np.float64),
            "ellipse_matrix_compress": attrs.get("ellipse_matrix_compress", []),
            "ellipse_matrix_expand": attrs.get("ellipse_matrix_expand", []),
            "pcl_mesh_connections": attrs.get("pcl_mesh_connections", []),
            "pcl_boundary_connections": attrs.get("pcl_boundary_connections", []),
            "color": attrs.get("color", {}),
        }
        if layer == 20:
            places.append(entry)
        elif layer == 2:
            objects.append(entry)
        elif layer == 4:
            rooms.append(entry)

    mesh_pts = np.array(dsg["mesh"]["points"], dtype=np.float64)
    mesh_faces = np.array(dsg["mesh"]["faces"], dtype=np.int64)

    # Pre-compute per-place mesh points
    for place in places:
        indices = place["pcl_mesh_connections"]
        place["mesh_points"] = mesh_pts[indices] if len(indices) > 0 else np.empty((0, 3))

    print(f"DSG loaded: {len(places)} places, {len(objects)} objects, "
          f"{len(rooms)} rooms, {len(mesh_pts):,} mesh pts")
    return dsg, places, objects, rooms, mesh_pts, mesh_faces


# ---------------------------------------------------------------------------
# Bbox utilities
# ---------------------------------------------------------------------------

def _bbox_from_center_extents_quat(center, extents, quat):
    """Convert center + extents + quaternion to oriented bbox corners (8x3)."""
    center = np.asarray(center, dtype=np.float64)
    extents = np.asarray(extents, dtype=np.float64)
    R = Rotation.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
    signs = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                       [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    corners_local = signs * extents
    corners_world = corners_local @ R.T + center
    return corners_world


def _get_obb_params(center, extents, quat):
    """Return OBB as (center, extents, R_matrix)."""
    center = np.asarray(center, dtype=np.float64)
    extents = np.asarray(extents, dtype=np.float64)
    R = Rotation.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
    return center, extents, R


def is_point_in_bbox(point, corners):
    """Check if a 3D point is inside an oriented bounding box (SAT-based)."""
    point = np.asarray(point, dtype=np.float64)
    center = corners.mean(axis=0)
    edges = corners[[1, 2, 4]] - corners[[0, 0, 0]]
    axes = edges / np.linalg.norm(edges, axis=1, keepdims=True)
    d = point - center
    half_extents = 0.5 * np.linalg.norm(edges, axis=1)
    for i in range(3):
        if np.abs(np.dot(d, axes[i])) > half_extents[i] + 1e-6:
            return False
    return True


def _points_in_obb(points, center, extents, R):
    """Check which points are inside an OBB. Returns boolean array."""
    pts_local = (points - center) @ R  # (N, 3)
    return np.all(np.abs(pts_local) <= extents + 1e-6, axis=1)


def _obb_corners(center, extents, R):
    """Get 8 corners of OBB."""
    signs = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                       [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    return (signs * extents) @ R.T + center


# ---------------------------------------------------------------------------
# OBB fitting for places
# ---------------------------------------------------------------------------

def fit_obb_from_points(points):
    """Fit an oriented bounding box to 3D points using PCA.

    Returns (center, extents, R_matrix) or None if insufficient points.
    """
    if len(points) < 3:
        return None
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigenvectors columns are axes (sorted by eigenvalue ascending)
    R = eigenvectors  # rotation matrix: world -> local
    aligned = centered @ R
    min_a = aligned.min(axis=0)
    max_a = aligned.max(axis=0)
    extents = (max_a - min_a) / 2.0
    center = centroid + R @ ((max_a + min_a) / 2.0)
    # Ensure minimum extents for degenerate cases (planar points)
    min_extent = 0.02
    extents = np.maximum(extents, min_extent)
    return center, extents, R


# ---------------------------------------------------------------------------
# 3D IoU via Monte Carlo sampling
# ---------------------------------------------------------------------------

def compute_3d_iou(center1, extents1, R1, center2, extents2, R2,
                   num_samples=50000, seed=42):
    """Compute 3D IoU between two oriented bounding boxes via Monte Carlo.

    Samples points in the combined bounding region and estimates
    intersection / union ratio.
    """
    rng = np.random.RandomState(seed)
    corners1 = _obb_corners(center1, extents1, R1)
    corners2 = _obb_corners(center2, extents2, R2)
    all_c = np.vstack([corners1, corners2])
    min_b = all_c.min(axis=0) - 0.01
    max_b = all_c.max(axis=0) + 0.01

    samples = rng.uniform(min_b, max_b, (num_samples, 3))
    in1 = _points_in_obb(samples, center1, extents1, R1)
    in2 = _points_in_obb(samples, center2, extents2, R2)

    intersection = np.sum(in1 & in2)
    union = np.sum(in1 | in2)
    if union == 0:
        return 0.0, 0.0, 0.0

    iou = intersection / union
    # Also compute volume ratio: intersection / min(vol1, vol2) as an alternative
    vol1 = 8 * np.prod(extents1)
    vol2 = 8 * np.prod(extents2)
    vol_min = min(vol1, vol2)
    vol_intersection_est = (intersection / num_samples) * np.prod(max_b - min_b)

    return iou, float(intersection / num_samples), float(union / num_samples)


# ---------------------------------------------------------------------------
# Multi-plane 2D IoU
# ---------------------------------------------------------------------------

def _points_in_convex_hull_2d(points, hull_xy):
    """Test if 2D points are inside a convex polygon using cross products."""
    hull = np.asarray(hull_xy)
    if len(hull) < 3:
        return np.zeros(len(points), dtype=bool)
    n = len(hull)
    signs = []
    for i in range(n):
        p1 = hull[i]
        p2 = hull[(i + 1) % n]
        edge = p2 - p1
        to_pts = points - p1
        cross = edge[0] * to_pts[:, 1] - edge[1] * to_pts[:, 0]
        signs.append(cross)
    signs = np.array(signs)
    all_pos = np.all(signs >= 0, axis=0)
    all_neg = np.all(signs <= 0, axis=0)
    return all_pos | all_neg


def compute_2d_iou_grid(corners1_2d, corners2_2d, grid_size=200):
    """Compute 2D IoU between two convex polygons via grid sampling."""
    all_xy = np.vstack([corners1_2d, corners2_2d])
    xmin, ymin = all_xy.min(axis=0) - 0.01
    xmax, ymax = all_xy.max(axis=0) + 0.01

    xs = np.linspace(xmin, xmax, grid_size)
    ys = np.linspace(ymin, ymax, grid_size)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys)
    points = np.stack([X.ravel(), Y.ravel()], axis=1)

    in1 = _points_in_convex_hull_2d(points, corners1_2d)
    in2 = _points_in_convex_hull_2d(points, corners2_2d)

    intersection = np.sum(in1 & in2)
    union = np.sum(in1 | in2)
    return intersection / union if union > 0 else 0.0


def project_corners_to_plane(corners, plane):
    """Project 3D corners to 2D by dropping one axis."""
    if plane == "xy":
        return corners[:, :2]
    elif plane == "xz":
        return corners[:, [0, 2]]
    elif plane == "yz":
        return corners[:, [1, 2]]
    else:
        raise ValueError(f"Unknown plane: {plane}")


# ---------------------------------------------------------------------------
# GT data loading
# ---------------------------------------------------------------------------

def load_gt_rooms(path):
    """Load GT room bounding boxes."""
    with open(path) as f:
        data = yaml.safe_load(f)
    rooms = {}
    for room_id, bboxes in data.items():
        if not isinstance(bboxes, list):
            bboxes = [bboxes]
        rooms[room_id] = []
        for b in bboxes:
            center, extents, R = _get_obb_params(b["center"], b["extents"], b["rotation"])
            corners = _bbox_from_center_extents_quat(b["center"], b["extents"], b["rotation"])
            rooms[room_id].append({
                "center": center,
                "extents": extents,
                "R": R,
                "rotation_q": b["rotation"],
                "corners": corners,
            })
    print(f"GT rooms loaded: {len(rooms)} rooms")
    return rooms


def load_gt_objects(path):
    """Load GT task object bounding boxes."""
    with open(path) as f:
        data = yaml.safe_load(f)
    objects = []
    for task_name, bboxes in data.items():
        if not isinstance(bboxes, list):
            bboxes = [bboxes]
        for b in bboxes:
            if not isinstance(b, dict) or "center" not in b:
                continue
            try:
                center, extents, R = _get_obb_params(b["center"], b["extents"], b["rotation"])
                corners = _bbox_from_center_extents_quat(b["center"], b["extents"], b["rotation"])
                objects.append({
                    "task": task_name,
                    "center": center,
                    "extents": extents,
                    "R": R,
                    "rotation_q": b["rotation"],
                    "corners": corners,
                })
            except (KeyError, TypeError, ValueError):
                continue
    print(f"GT objects loaded: {len(objects)} objects from {len(data)} tasks")
    return objects


# ---------------------------------------------------------------------------
# Evaluation: Room Assignment
# ---------------------------------------------------------------------------

def evaluate_room_assignment(places, gt_rooms):
    """Assign each place to the GT room containing its position."""
    assignments = defaultdict(list)
    unassigned = []
    for i, place in enumerate(places):
        pos = place["position"]
        best_room = None
        for room_id, bboxes in gt_rooms.items():
            for bbox in bboxes:
                if is_point_in_bbox(pos, bbox["corners"]):
                    best_room = room_id
                    break
            if best_room is not None:
                break
        if best_room is not None:
            assignments[best_room].append(i)
        else:
            unassigned.append(i)

    total = len(places)
    assigned = total - len(unassigned)
    print(f"\n=== Room Assignment ===")
    print(f"Total places: {total}")
    print(f"Assigned to rooms: {assigned} ({100*assigned/total:.1f}%)" if total else "N/A")
    print(f"Unassigned: {len(unassigned)} ({100*len(unassigned)/total:.1f}%)" if total else "N/A")
    print(f"Rooms covered: {len(assignments)}/{len(gt_rooms)}")
    for room_id in sorted(gt_rooms.keys()):
        count = len(assignments.get(room_id, []))
        flag = "" if count > 0 else " (MISSED)"
        print(f"  Room {room_id}: {count} places{flag}")
    return dict(assignments), unassigned


# ---------------------------------------------------------------------------
# Evaluation: Object Proximity
# ---------------------------------------------------------------------------

def evaluate_object_proximity(places, gt_objects):
    """Distance from each GT object center to nearest place."""
    if not places:
        return [], {}
    place_positions = np.array([p["position"] for p in places])
    tree = cKDTree(place_positions)
    distances = []
    for obj in gt_objects:
        dist, idx = tree.query(obj["center"])
        distances.append({
            "task": obj["task"],
            "distance": float(dist),
            "nearest_place_id": places[idx]["id"],
            "obj_center": obj["center"].tolist(),
            "place_position": places[idx]["position"].tolist(),
            "obj_extents": obj["extents"].tolist(),
        })

    dists = [d["distance"] for d in distances]
    print(f"\n=== Object Proximity ===")
    print(f"Total GT objects: {len(gt_objects)}")
    if dists:
        print(f"Distance stats (m): min={min(dists):.3f}, max={max(dists):.3f}, "
              f"mean={np.mean(dists):.3f}, median={np.median(dists):.3f}")
        for t in [0.1, 0.25, 0.5, 1.0, 2.0]:
            count = sum(1 for d in dists if d < t)
            print(f"  Within {t:.2f}m: {count}/{len(dists)} ({100*count/len(dists):.1f}%)")

    stats = {
        "min": float(min(dists)) if dists else 0,
        "max": float(max(dists)) if dists else 0,
        "mean": float(np.mean(dists)) if dists else 0,
        "median": float(np.median(dists)) if dists else 0,
    }
    return distances, stats


# ---------------------------------------------------------------------------
# Evaluation: Room Centroid Distance
# ---------------------------------------------------------------------------

def evaluate_room_centroid_distance(places, gt_rooms):
    """Distance from each GT room center to nearest place."""
    if not places:
        return []
    place_positions = np.array([p["position"] for p in places])
    tree = cKDTree(place_positions)
    results = []
    for room_id, bboxes in gt_rooms.items():
        for j, bbox in enumerate(bboxes):
            dist, idx = tree.query(bbox["center"])
            results.append({
                "room_id": room_id,
                "bbox_idx": j,
                "room_center": bbox["center"].tolist(),
                "distance": float(dist),
                "nearest_place_id": places[idx]["id"],
                "nearest_place_pos": places[idx]["position"].tolist(),
            })
    dists = [r["distance"] for r in results]
    print(f"\n=== Room Centroid Coverage ===")
    if dists:
        print(f"Room bboxes: {len(results)}")
        print(f"Distance stats (m): min={min(dists):.3f}, max={max(dists):.3f}, "
              f"mean={np.mean(dists):.3f}, median={np.median(dists):.3f}")
    return results


# ---------------------------------------------------------------------------
# NEW: Place OBB fitting and 3D IoU with GT objects
# ---------------------------------------------------------------------------

def evaluate_3d_iou_objects(places, gt_objects, mesh_pts, num_samples=50000):
    """Fit OBB to each place's mesh points and compute 3D IoU with GT objects.

    For each GT object:
      1. Find nearest K places
      2. Merge their mesh points, fit OBB
      3. Compute 3D IoU between merged-place-OBB and GT OBB
    """
    if not places:
        return []

    # Pre-fit OBB for each place
    place_obbs = []
    for place in places:
        pts = place["mesh_points"]
        if len(pts) >= 3:
            obb = fit_obb_from_points(pts)
            place_obbs.append(obb)
        else:
            place_obbs.append(None)

    place_centroids = np.array([p["position"] for p in places])
    tree = cKDTree(place_centroids)

    results = []
    for obj in gt_objects:
        # Find places near this object
        K = min(5, len(places))
        dists, indices = tree.query(obj["center"], k=K)
        if K == 1:
            indices = np.array([indices])
            dists = np.array([dists])

        # Try individual places and merged places
        best_iou = 0.0
        best_place_idx = None
        best_type = "none"

        # Check individual places
        for idx in indices:
            obb = place_obbs[idx]
            if obb is None:
                continue
            pc, pe, pR = obb
            iou, _, _ = compute_3d_iou(pc, pe, pR,
                                       obj["center"], obj["extents"], obj["R"],
                                       num_samples=num_samples)
            if iou > best_iou:
                best_iou = iou
                best_place_idx = int(idx)
                best_type = "single"

        # Check merged places
        merged_pts = []
        for idx in indices:
            pts = places[idx]["mesh_points"]
            if len(pts) > 0:
                merged_pts.append(pts)
        if merged_pts:
            all_pts = np.vstack(merged_pts)
            merged_obb = fit_obb_from_points(all_pts)
            if merged_obb is not None:
                mc, me, mR = merged_obb
                iou, _, _ = compute_3d_iou(mc, me, mR,
                                           obj["center"], obj["extents"], obj["R"],
                                           num_samples=num_samples)
                if iou > best_iou:
                    best_iou = iou
                    best_place_idx = None
                    best_type = "merged"

        results.append({
            "task": obj["task"],
            "best_3d_iou": float(best_iou),
            "best_type": best_type,
            "best_place_idx": best_place_idx,
            "obj_center": obj["center"].tolist(),
            "obj_extents": obj["extents"].tolist(),
        })

    ious = [r["best_3d_iou"] for r in results]
    nonzero = [i for i in ious if i > 0]
    print(f"\n=== 3D IoU (Place OBB vs GT Object) ===")
    print(f"Objects with IoU > 0: {len(nonzero)}/{len(ious)}")
    if nonzero:
        print(f"IoU stats: min={min(nonzero):.4f}, max={max(nonzero):.4f}, "
              f"mean={np.mean(nonzero):.4f}, median={np.median(nonzero):.4f}")
    else:
        print("All 3D IoUs are 0")
    return results


# ---------------------------------------------------------------------------
# NEW: 3D IoU with GT rooms
# ---------------------------------------------------------------------------

def evaluate_3d_iou_rooms(places, gt_rooms, mesh_pts, num_samples=50000):
    """Fit OBB to places in each room and compute 3D IoU with GT room bboxes."""
    if not places:
        return []

    place_centroids = np.array([p["position"] for p in places])
    place_obbs = []
    for place in places:
        pts = place["mesh_points"]
        if len(pts) >= 3:
            place_obbs.append(fit_obb_from_points(pts))
        else:
            place_obbs.append(None)

    results = []
    for room_id, bboxes in gt_rooms.items():
        for j, gt_bbox in enumerate(bboxes):
            # Find places inside or near this room
            inside_indices = []
            for i, place in enumerate(places):
                if is_point_in_bbox(place["position"], gt_bbox["corners"]):
                    inside_indices.append(i)

            if not inside_indices:
                results.append({
                    "room_id": room_id,
                    "bbox_idx": j,
                    "best_3d_iou": 0.0,
                    "places_inside": 0,
                    "place_indices": [],
                })
                continue

            # Merge mesh points from all places in room
            merged_pts = []
            for idx in inside_indices:
                pts = places[idx]["mesh_points"]
                if len(pts) > 0:
                    merged_pts.append(pts)

            best_iou = 0.0
            if merged_pts:
                all_pts = np.vstack(merged_pts)
                merged_obb = fit_obb_from_points(all_pts)
                if merged_obb is not None:
                    mc, me, mR = merged_obb
                    best_iou, _, _ = compute_3d_iou(
                        mc, me, mR,
                        gt_bbox["center"], gt_bbox["extents"], gt_bbox["R"],
                        num_samples=num_samples)

            results.append({
                "room_id": room_id,
                "bbox_idx": j,
                "best_3d_iou": float(best_iou),
                "places_inside": len(inside_indices),
                "place_indices": inside_indices,
            })

    ious = [r["best_3d_iou"] for r in results]
    nonzero = [i for i in ious if i > 0]
    print(f"\n=== 3D IoU (Place OBB vs GT Room) ===")
    print(f"Room bboxes with IoU > 0: {len(nonzero)}/{len(ious)}")
    if nonzero:
        print(f"IoU stats: min={min(nonzero):.4f}, max={max(nonzero):.4f}, "
              f"mean={np.mean(nonzero):.4f}, median={np.median(nonzero):.4f}")
    else:
        print("All 3D IoUs are 0")
    return results


# ---------------------------------------------------------------------------
# NEW: Multi-plane 2D IoU
# ---------------------------------------------------------------------------

def _get_dominant_plane(extents):
    """Return the plane name ('xy', 'xz', 'yz') perpendicular to smallest extent."""
    idx = np.argmin(np.abs(extents))
    return {0: "yz", 1: "xz", 2: "xy"}[idx]


def _convex_hull_2d(points_2d):
    """Compute convex hull of 2D points, returning hull vertices in CCW order."""
    from scipy.spatial import ConvexHull
    if len(points_2d) < 3:
        return points_2d
    try:
        hull = ConvexHull(points_2d)
        return points_2d[hull.vertices]
    except Exception:
        return points_2d


def evaluate_multiple_2d_iou(places, gt_objects, planes=("xy", "xz", "yz")):
    """Compute 2D IoU between place mesh points and GT bbox on multiple planes.

    Uses place mesh points (not just boundary polygon), projects to each plane,
    computes convex hull, and then IoU with GT bbox projection.
    Also computes 'best' plane: the GT bbox's dominant plane.
    """
    if not places:
        return []

    place_centroids = np.array([p["position"] for p in places])
    tree = cKDTree(place_centroids)

    all_planes = list(planes) + ["best"]

    results = []
    for obj in gt_objects:
        K = min(5, len(places))
        dists, indices = tree.query(obj["center"], k=K)
        if K == 1:
            indices = np.array([indices])

        # Collect mesh points from nearby places
        nearby_pts = []
        for idx in indices:
            pts = places[idx]["mesh_points"]
            if len(pts) > 0:
                nearby_pts.append(pts)
        merged_pts = np.vstack(nearby_pts) if nearby_pts else np.empty((0, 3))

        obj_corners = obj["corners"]
        obj_dominant_plane = _get_dominant_plane(obj["extents"])
        best_per_plane = {}

        for plane in all_planes:
            actual_plane = obj_dominant_plane if plane == "best" else plane
            obj_2d = project_corners_to_plane(obj_corners, actual_plane)

            best_iou = 0.0
            if len(merged_pts) >= 3:
                pts_2d = project_corners_to_plane(merged_pts, actual_plane)
                hull = _convex_hull_2d(pts_2d)
                if len(hull) >= 3:
                    try:
                        best_iou = compute_2d_iou_grid(hull, obj_2d)
                    except Exception:
                        best_iou = 0.0

            best_per_plane[plane] = {"iou": float(best_iou)}

        results.append({
            "task": obj["task"],
            "obj_center": obj["center"].tolist(),
            "obj_extents": obj["extents"].tolist(),
            "dominant_plane": obj_dominant_plane,
            **{f"iou_{p}": best_per_plane[p]["iou"] for p in all_planes},
        })

    for plane in all_planes:
        ious = [r[f"iou_{plane}"] for r in results]
        nonzero = [i for i in ious if i > 0]
        label = f"{plane.upper()} ({'dominant' if plane == 'best' else 'world'})"
        print(f"\n=== 2D IoU {label} ===")
        print(f"Objects with IoU > 0: {len(nonzero)}/{len(ious)}")
        if nonzero:
            print(f"IoU stats: min={min(nonzero):.4f}, max={max(nonzero):.4f}, "
                  f"mean={np.mean(nonzero):.4f}, median={np.median(nonzero):.4f}")

    return results


# ---------------------------------------------------------------------------
# NEW: Chamfer distance (place mesh points <-> GT bbox surface)
# ---------------------------------------------------------------------------

def _sample_bbox_surface(center, extents, R, n_samples=1000, seed=42):
    """Sample points uniformly on the surface of an oriented bounding box."""
    rng = np.random.RandomState(seed)
    # 6 faces, each with area proportional to the other two extents
    face_areas = []
    for axis in range(3):
        other_axes = [i for i in range(3) if i != axis]
        face_areas.append(2 * extents[other_axes[0]] * extents[other_axes[1]])
    face_probs = np.array(face_areas) / sum(face_areas)

    samples = []
    for _ in range(n_samples):
        # Choose a face
        axis = rng.choice(3, p=face_probs)
        sign = rng.choice([-1, 1])
        other_axes = [i for i in range(3) if i != axis]

        # Sample point on the chosen face
        local_pt = np.zeros(3)
        local_pt[axis] = sign * extents[axis]
        for ax in other_axes:
            local_pt[ax] = rng.uniform(-extents[ax], extents[ax])

        # Transform to world
        world_pt = R @ local_pt + center
        samples.append(world_pt)

    return np.array(samples)


def evaluate_chamfer_distance(places, gt_objects):
    """Compute Chamfer distance between place mesh points and GT bbox surfaces.

    For each GT object:
      1. Sample points on the GT bbox surface
      2. Collect mesh points from nearby places
      3. Compute bidirectional Chamfer distance
    """
    if not places:
        return []

    place_centroids = np.array([p["position"] for p in places])
    tree = cKDTree(place_centroids)

    results = []
    for obj in gt_objects:
        # Sample GT bbox surface
        gt_surface_pts = _sample_bbox_surface(
            obj["center"], obj["extents"], obj["R"], n_samples=1000)

        # Find nearby places and collect mesh points
        diag = 2 * np.linalg.norm(obj["extents"])
        K = min(10, len(places))
        dists, indices = tree.query(obj["center"], k=K)
        if K == 1:
            indices = np.array([indices])

        nearby_pts = []
        for idx in indices:
            pts = places[idx]["mesh_points"]
            if len(pts) > 0:
                # Only include points within 2x bbox diagonal
                d = np.linalg.norm(pts - obj["center"], axis=1)
                nearby_pts.append(pts[d < max(diag, 1.5)])
        if nearby_pts:
            place_pts = np.vstack(nearby_pts)
        else:
            place_pts = np.empty((0, 3))

        if len(place_pts) == 0:
            results.append({
                "task": obj["task"],
                "chamfer": None,
                "place_to_gt_mean": None,
                "gt_to_place_mean": None,
            })
            continue

        # Build KD trees
        place_tree = cKDTree(place_pts)
        gt_tree = cKDTree(gt_surface_pts)

        # Place -> GT
        p2g_dists, _ = place_tree.query(gt_surface_pts, k=1)
        # GT -> Place
        g2p_dists, _ = gt_tree.query(place_pts, k=1)

        chamfer = float(np.mean(p2g_dists) + np.mean(g2p_dists))

        results.append({
            "task": obj["task"],
            "chamfer": chamfer,
            "place_to_gt_mean": float(np.mean(p2g_dists)),
            "gt_to_place_mean": float(np.mean(g2p_dists)),
            "place_to_gt_max": float(np.max(p2g_dists)),
            "gt_to_place_max": float(np.max(g2p_dists)),
            "n_place_points": len(place_pts),
            "n_gt_surface_samples": len(gt_surface_pts),
        })

    valid = [r for r in results if r["chamfer"] is not None]
    if valid:
        chamfers = [r["chamfer"] for r in valid]
        print(f"\n=== Chamfer Distance (place mesh <-> GT bbox surface) ===")
        print(f"  Valid objects: {len(valid)}/{len(results)}")
        print(f"  Chamfer (m): min={min(chamfers):.4f}, max={max(chamfers):.4f}, "
              f"mean={np.mean(chamfers):.4f}, median={np.median(chamfers):.4f}")
        p2g = [r["place_to_gt_mean"] for r in valid]
        g2p = [r["gt_to_place_mean"] for r in valid]
        print(f"  Place->GT mean: min={min(p2g):.4f}, max={max(p2g):.4f}, "
              f"avg={np.mean(p2g):.4f}")
        print(f"  GT->Place mean: min={min(g2p):.4f}, max={max(g2p):.4f}, "
              f"avg={np.mean(g2p):.4f}")

    return results


# ---------------------------------------------------------------------------
# NEW: Point-in-bbox precision/recall
# ---------------------------------------------------------------------------

def evaluate_point_coverage(places, gt_objects):
    """For each GT object, compute fraction of place mesh points inside the bbox.

    Precision: fraction of nearby places' mesh points inside GT bbox
    Recall: fraction of GT bbox volume "covered" by place points (via sampling)
    """
    if not places:
        return []

    results = []
    for obj in gt_objects:
        # Find places whose centroids are within 2x the bbox diagonal
        diag = 2 * np.linalg.norm(obj["extents"])
        nearby_places = []
        for i, place in enumerate(places):
            dist = np.linalg.norm(place["position"] - obj["center"])
            if dist < max(diag, 1.0):
                nearby_places.append(i)

        # Collect all mesh points from nearby places
        all_pts = []
        for idx in nearby_places:
            pts = places[idx]["mesh_points"]
            if len(pts) > 0:
                all_pts.append(pts)

        if not all_pts:
            results.append({
                "task": obj["task"],
                "precision": 0.0,
                "recall": 0.0,
                "total_place_points": 0,
                "points_in_bbox": 0,
            })
            continue

        all_pts = np.vstack(all_pts)
        inside = _points_in_obb(all_pts, obj["center"], obj["extents"], obj["R"])
        precision = float(np.mean(inside)) if len(all_pts) > 0 else 0.0

        # Recall: sample GT bbox volume, check if near any place point
        if len(all_pts) > 0:
            place_tree = cKDTree(all_pts)
            rng = np.random.RandomState(42)
            samples_local = rng.uniform(-1, 1, (2000, 3)) * obj["extents"]
            samples_world = samples_local @ obj["R"].T + obj["center"]
            nn_dists, _ = place_tree.query(samples_world)
            threshold = 0.1  # 10cm
            recall = float(np.mean(nn_dists < threshold))
        else:
            recall = 0.0

        results.append({
            "task": obj["task"],
            "precision": float(precision),
            "recall": float(recall),
            "total_place_points": len(all_pts),
            "points_in_bbox": int(np.sum(inside)),
        })

    precs = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    print(f"\n=== Point-in-Bbox Coverage ===")
    print(f"Precision: mean={np.mean(precs):.4f}, median={np.median(precs):.4f}")
    print(f"Recall:    mean={np.mean(recalls):.4f}, median={np.median(recalls):.4f}")
    nz_prec = sum(1 for p in precs if p > 0)
    nz_rec = sum(1 for r in recalls if r > 0)
    print(f"Precision > 0: {nz_prec}/{len(precs)}, Recall > 0: {nz_rec}/{len(recalls)}")
    return results


# ---------------------------------------------------------------------------
# NEW: Place statistics
# ---------------------------------------------------------------------------

def compute_place_statistics(places, mesh_pts):
    """Compute per-place statistics: area, mesh points count, bbox volume."""
    stats = []
    for place in places:
        boundary = place.get("boundary")
        n_boundary = len(boundary) if boundary is not None else 0
        n_mesh = len(place.get("pcl_mesh_connections", []))

        # Compute approximate area of boundary polygon (projected to best-fit plane)
        area = 0.0
        if boundary is not None and len(boundary) >= 3:
            # Use Newell's method / simple triangulation from centroid
            boundary_np = np.asarray(boundary)
            centroid = boundary_np.mean(axis=0)
            total = 0.0
            n = len(boundary_np)
            for i in range(n):
                j = (i + 1) % n
                # area of triangle (centroid, i, j)
                v1 = boundary_np[i] - centroid
                v2 = boundary_np[j] - centroid
                total += 0.5 * np.linalg.norm(np.cross(v1, v2))
            area = float(total)

        # OBB volume
        obb_vol = 0.0
        pts = place.get("mesh_points")
        if pts is not None and len(pts) >= 3:
            obb = fit_obb_from_points(pts)
            if obb is not None:
                obb_vol = float(8 * np.prod(obb[1]))

        stats.append({
            "place_id": place["id"],
            "name": place["name"],
            "position": place["position"].tolist(),
            "n_boundary_verts": n_boundary,
            "n_mesh_points": n_mesh,
            "boundary_area_m2": area,
            "obb_volume_m3": obb_vol,
        })
    return stats


# ---------------------------------------------------------------------------
# NEW: Place surface normal classification
# ---------------------------------------------------------------------------

def classify_places_by_normal(places):
    """Classify each place by its dominant surface normal direction.

    Uses PCA on mesh points to determine the surface normal (smallest
    principal component). Classifies as: wall, floor, ceiling, other.

    Returns:
        list of dicts with place_id, normal, type, normal_vec
    """
    results = []
    for place in places:
        pts = place.get("mesh_points")
        if pts is None or len(pts) < 3:
            results.append({
                "place_id": place["id"],
                "type": "unknown",
                "normal": [0, 0, 0],
                "confidence": 0.0,
            })
            continue

        pts = np.asarray(pts, dtype=np.float64)
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Smallest eigenvalue -> normal direction
        normal = eigenvectors[:, 0]
        # Ensure normal points "outward" (positive z for floor, etc.)
        if normal[2] < 0:
            normal = -normal

        abs_normal = np.abs(normal)
        dominant = np.argmax(abs_normal)
        ratio = abs_normal[dominant] / (abs_normal.sum() + 1e-10)

        if dominant == 2 and ratio > 0.7:
            if normal[2] > 0:
                ptype = "floor"
            else:
                ptype = "ceiling"
        elif dominant in (0, 1) and ratio > 0.5:
            ptype = "wall"
        else:
            ptype = "other"

        results.append({
            "place_id": place["id"],
            "type": ptype,
            "normal": normal.tolist(),
            "normal_dominant_ratio": float(ratio),
            "position": place["position"].tolist(),
        })

    # Summary
    from collections import Counter
    type_counts = Counter(r["type"] for r in results)
    print(f"\n=== Place Surface Type Classification ===")
    for t in ["wall", "floor", "ceiling", "other", "unknown"]:
        c = type_counts.get(t, 0)
        print(f"  {t}: {c}/{len(results)} ({100*c/len(results):.1f}%)" if results else f"  {t}: 0")

    return results


# ---------------------------------------------------------------------------
# NEW: Surface Coverage metrics (replaces 3D IoU for 2D surfaces)
# ---------------------------------------------------------------------------

def evaluate_surface_coverage(places, gt_objects, distance_thresholds=(0.1, 0.25, 0.5)):
    """Surface-based coverage: how well do place mesh points cover GT bbox surfaces?

    For each GT object:
      1. Sample points on GT bbox surface (6 faces)
      2. For each sample, find distance to nearest place mesh point
      3. Compute:
         - Surface Precision: fraction of place points within threshold of GT surface
         - Surface Recall: fraction of GT surface samples within threshold of any place
         - Surface F1: harmonic mean of precision and recall

    This replaces 3D IoU which is mathematically invalid for 2D surface places.
    """
    if not places:
        return []

    # Build global KD tree of all place mesh points
    all_pts_list = []
    place_pt_ranges = []  # (start_idx, end_idx) for each place
    for place in places:
        pts = place.get("mesh_points")
        if pts is not None and len(pts) > 0:
            start = len(all_pts_list)
            all_pts_list.append(np.asarray(pts, dtype=np.float64))
            end = start + len(pts)
            place_pt_ranges.append((start, end, place["id"]))
        else:
            place_pt_ranges.append((0, 0, place["id"]))

    results = []
    for obj in gt_objects:
        # Sample GT bbox surface
        gt_surface = _sample_bbox_surface(
            obj["center"], obj["extents"], obj["R"], n_samples=2000, seed=42)

        # Find nearby place mesh points (within 2x bbox diagonal)
        diag = 2.5 * np.linalg.norm(obj["extents"])
        diag = max(diag, 1.0)

        nearby_pt_list = []
        for place in places:
            pts = place.get("mesh_points")
            if pts is None or len(pts) == 0:
                continue
            pts = np.asarray(pts, dtype=np.float64)
            dists = np.linalg.norm(pts - obj["center"], axis=1)
            nearby = pts[dists < diag]
            if len(nearby) > 0:
                nearby_pt_list.append(nearby)

        if not nearby_pt_list:
            results.append({
                "task": obj["task"],
                "thresholds": {str(t): {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                              for t in distance_thresholds},
                "n_nearby_place_points": 0,
            })
            continue

        nearby_pts = np.vstack(nearby_pt_list)
        place_tree = cKDTree(nearby_pts)
        gt_tree = cKDTree(gt_surface)

        threshold_results = {}
        for thresh in distance_thresholds:
            # Precision: fraction of place points within 'thresh' of GT surface
            p_dists, _ = gt_tree.query(nearby_pts, k=1)
            precision = float(np.mean(p_dists < thresh))

            # Recall: fraction of GT surface samples within 'thresh' of place points
            g_dists, _ = place_tree.query(gt_surface, k=1)
            recall = float(np.mean(g_dists < thresh))

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            threshold_results[str(thresh)] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

        results.append({
            "task": obj["task"],
            "thresholds": threshold_results,
            "n_nearby_place_points": len(nearby_pts),
            "obj_center": obj["center"].tolist(),
            "obj_extents": obj["extents"].tolist(),
        })

    # Print summary
    for thresh in distance_thresholds:
        precs = [r["thresholds"][str(thresh)]["precision"] for r in results]
        recalls = [r["thresholds"][str(thresh)]["recall"] for r in results]
        f1s = [r["thresholds"][str(thresh)]["f1"] for r in results]
        print(f"\n=== Surface Coverage @ {thresh}m ===")
        print(f"  Precision: mean={np.mean(precs):.4f}, median={np.median(precs):.4f}")
        print(f"  Recall:    mean={np.mean(recalls):.4f}, median={np.median(recalls):.4f}")
        print(f"  F1:        mean={np.mean(f1s):.4f}, median={np.median(f1s):.4f}")
        nz = sum(1 for f in f1s if f > 0)
        print(f"  F1 > 0:    {nz}/{len(f1s)}")

    return results


# ---------------------------------------------------------------------------
# NEW: Room assignment with distance-based filter for unannotated regions
# ---------------------------------------------------------------------------

def evaluate_room_assignment_filtered(places, gt_rooms,
                                       filter_unassigned=True,
                                       max_room_dist=3.0):
    """Room assignment with optional filtering of unannotated regions.

    When filter_unassigned=True, unassigned places that are farther than
    max_room_dist from any GT room centroid are classified as "unannotated"
    (e.g., corridors, hallways) rather than "unassigned", giving a more
    accurate picture of algorithm performance within annotated regions.

    Returns:
        assignments, unassigned, unannotated
    """
    assignments = defaultdict(list)
    unassigned = []
    unannotated = []

    # Compute GT room centroids for distance check
    room_centers = {}
    for room_id, bboxes in gt_rooms.items():
        centers = [bbox["center"] for bbox in bboxes]
        room_centers[room_id] = centers

    for i, place in enumerate(places):
        pos = place["position"]
        best_room = None
        for room_id, bboxes in gt_rooms.items():
            for bbox in bboxes:
                if is_point_in_bbox(pos, bbox["corners"]):
                    best_room = room_id
                    break
            if best_room is not None:
                break

        if best_room is not None:
            assignments[best_room].append(i)
        elif filter_unassigned:
            # Check distance to nearest room center
            min_dist = float("inf")
            for centers in room_centers.values():
                for c in centers:
                    d = np.linalg.norm(pos - c)
                    if d < min_dist:
                        min_dist = d
            if min_dist > max_room_dist:
                unannotated.append(i)
            else:
                unassigned.append(i)
        else:
            unassigned.append(i)

    total = len(places)
    assigned = total - len(unassigned) - len(unannotated)

    print(f"\n=== Room Assignment (filtered, max_room_dist={max_room_dist}m) ===")
    print(f"Total places: {total}")
    print(f"Assigned to rooms: {assigned} ({100*assigned/total:.1f}%)" if total else "N/A")
    print(f"Unassigned (near rooms): {len(unassigned)} ({100*len(unassigned)/total:.1f}%)" if total else "N/A")
    print(f"Unannotated (far from rooms): {len(unannotated)} ({100*len(unannotated)/total:.1f}%)" if total else "N/A")
    print(f"Rooms covered: {len(assignments)}/{len(gt_rooms)}")

    # Effective coverage (excluding unannotated)
    effective_total = total - len(unannotated)
    if effective_total > 0:
        print(f"Effective room assignment (excl. unannotated): "
              f"{assigned}/{effective_total} ({100*assigned/effective_total:.1f}%)")

    for room_id in sorted(gt_rooms.keys()):
        count = len(assignments.get(room_id, []))
        flag = "" if count > 0 else " (MISSED)"
        print(f"  Room {room_id}: {count} places{flag}")

    return dict(assignments), unassigned, unannotated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Clio DSG Evaluation — surface-aware metrics for 2D place nodes")
    parser.add_argument("--dsg", required=True, help="Path to dsg.json")
    parser.add_argument("--rooms", default=None, help="Path to rooms_<scene>.yaml")
    parser.add_argument("--tasks", default=None, help="Path to tasks_<scene>.yaml")
    parser.add_argument("--output", default=None, help="Path to save evaluation results JSON")
    parser.add_argument("--mc_samples", type=int, default=50000,
                        help="Monte Carlo samples for 3D IoU (default: 50000)")
    parser.add_argument("--max_room_dist", type=float, default=3.0,
                        help="Max distance (m) for room unannotated filtering (default: 3.0)")
    args = parser.parse_args()

    dsg_data, places, objects, rooms, mesh_pts, mesh_faces = load_dsg(args.dsg)

    results = {
        "dsg_path": args.dsg,
        "num_places": len(places),
        "num_objects": len(objects),
        "num_dsg_rooms": len(rooms),
        "num_mesh_points": len(mesh_pts),
        "num_mesh_faces": len(mesh_faces),
    }

    # Place statistics & surface type classification
    results["place_stats"] = compute_place_statistics(places, mesh_pts)
    results["place_surface_types"] = classify_places_by_normal(places)

    # ---- Room evaluation ----
    if args.rooms and Path(args.rooms).exists():
        gt_rooms = load_gt_rooms(args.rooms)

        # Original room assignment
        room_assignments_raw, unassigned_raw = evaluate_room_assignment(places, gt_rooms)
        results["room_assignment_raw"] = {
            "per_room": {str(k): len(v) for k, v in room_assignments_raw.items()},
            "unassigned_count": len(unassigned_raw),
            "rooms_covered": len(room_assignments_raw),
            "total_rooms": len(gt_rooms),
        }

        # Filtered room assignment (separates unannotated regions)
        room_assignments, unassigned, unannotated = evaluate_room_assignment_filtered(
            places, gt_rooms, filter_unassigned=True,
            max_room_dist=args.max_room_dist)
        results["room_assignment"] = {
            "per_room": {str(k): len(v) for k, v in room_assignments.items()},
            "unassigned_count": len(unassigned),
            "unannotated_count": len(unannotated),
            "rooms_covered": len(room_assignments),
            "total_rooms": len(gt_rooms),
            "effective_total": len(places) - len(unannotated),
            "filter_distance_m": args.max_room_dist,
        }

        room_centroids = evaluate_room_centroid_distance(places, gt_rooms)
        results["room_centroid_distances"] = room_centroids

        room_3d_iou = evaluate_3d_iou_rooms(places, gt_rooms, mesh_pts,
                                            num_samples=args.mc_samples)
        results["room_3d_iou"] = room_3d_iou
    else:
        print(f"Rooms file not found: {args.rooms}")

    # ---- Task object evaluation ----
    if args.tasks and Path(args.tasks).exists():
        gt_objects = load_gt_objects(args.tasks)

        # 1. Object proximity (distance-based)
        obj_distances, proximity_stats = evaluate_object_proximity(places, gt_objects)
        results["object_proximity"] = {
            "distances": obj_distances,
            "stats": proximity_stats,
        }

        # 2. Chamfer distance (place mesh <-> GT bbox surface)
        chamfer = evaluate_chamfer_distance(places, gt_objects)
        results["object_chamfer_distance"] = chamfer

        # 3. Surface coverage (replaces 3D IoU for 2D surface places)
        surface_cov = evaluate_surface_coverage(places, gt_objects)
        results["object_surface_coverage"] = surface_cov

        # 4. Point-in-bbox precision/recall (legacy)
        pt_coverage = evaluate_point_coverage(places, gt_objects)
        results["object_point_coverage"] = pt_coverage

        # 5. 3D IoU (place OBB vs GT) — KEPT WITH CAVEAT
        #    Note: expected to be very low because places are 2D surfaces,
        #    their OBBs have near-zero volume along the surface normal.
        obj_3d_iou = evaluate_3d_iou_objects(places, gt_objects, mesh_pts,
                                             num_samples=args.mc_samples)
        results["object_3d_iou"] = obj_3d_iou
    else:
        print(f"Tasks file not found: {args.tasks}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  Evaluation Summary: {Path(args.dsg).parent.name}")
    print(f"{'='*60}")

    if "object_proximity" in results:
        stats = results["object_proximity"]["stats"]
        if stats.get("max", 0) > 0:
            print(f"  Obj Proximity:  mean={stats['mean']:.3f}m, "
                  f"<1m={stats.get('lt_1m','N/A')}")

    if "object_surface_coverage" in results:
        f1s_025 = [r["thresholds"]["0.25"]["f1"] for r in results["object_surface_coverage"]]
        print(f"  Surface F1@0.25m: mean={np.mean(f1s_025):.4f}, "
              f">0={sum(1 for f in f1s_025 if f>0)}/{len(f1s_025)}")

    if "object_chamfer_distance" in results:
        chamfers = [r["chamfer"] for r in results["object_chamfer_distance"]
                    if r["chamfer"] is not None]
        if chamfers:
            print(f"  Chamfer (m):    mean={np.mean(chamfers):.3f}")

    if "room_assignment" in results:
        ra = results["room_assignment"]
        print(f"  Room Coverage:  {ra['rooms_covered']}/{ra['total_rooms']}")
        if "unannotated_count" in ra:
            print(f"  Unannotated:    {ra['unannotated_count']} places "
                  f"(>{ra['filter_distance_m']}m from rooms)")
            print(f"  Effective Cov:  {ra['rooms_covered']}/{ra['total_rooms']} rooms, "
                  f"{ra['effective_total'] - ra['unassigned_count']}/{ra['effective_total']} "
                  f"places assigned (excl. unannotated)")

    if "room_3d_iou" in results:
        ious = [r["best_3d_iou"] for r in results["room_3d_iou"]]
        nz = [i for i in ious if i > 0]
        print(f"  Room 3D IoU:    >0={len(nz)}/{len(ious)}, "
              f"max={max(ious):.4f}" if ious else "N/A")

    # Save
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.floating):
                    return float(o)
                return super().default(o)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
