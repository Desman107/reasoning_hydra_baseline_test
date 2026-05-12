#!/usr/bin/env python3
"""Clio dataset evaluation script.

Evaluates DSG scene graph output against Clio ground-truth annotations:
- Room assignment accuracy: assign each place node to nearest GT room
- Task object proximity: distance from each GT task object to nearest place
- Room coverage: % of GT rooms containing at least one place

Usage:
    python eval/eval_clio.py --dsg /tmp/clio_output/<scene>/dsg.json \
        --rooms /data/YueChang/Clio/<scene>/rooms_<scene>.yaml \
        --tasks /data/YueChang/Clio/<scene>/tasks_<scene>.yaml
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree


def load_dsg(path):
    """Load DSG JSON and return place nodes with attributes."""
    with open(path) as f:
        dsg = json.load(f)

    # Extract mesh_places nodes (layer 20) and any other layers
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
            "color": attrs.get("color", {}),
        }
        if layer == 20:
            places.append(entry)
        elif layer == 2:
            objects.append(entry)
        elif layer == 4:
            rooms.append(entry)

    print(f"DSG loaded: {len(places)} places, {len(objects)} objects, "
          f"{len(rooms)} rooms, {len(dsg['mesh']['points'])} mesh pts")
    return dsg, places, objects, rooms


def _bbox_from_center_extents_quat(center, extents, quat):
    """Convert center + extents + quaternion to oriented bbox corners (8x3)."""
    center = np.asarray(center, dtype=np.float64)
    extents = np.asarray(extents, dtype=np.float64)
    R = Rotation.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()

    # 8 corners of unit cube, scaled by extents
    signs = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                       [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    corners_local = signs * extents  # (8, 3)
    corners_world = corners_local @ R.T + center  # (8, 3)
    return corners_world


def is_point_in_bbox(point, corners):
    """Check if a 3D point is inside an oriented bounding box.

    Uses separating axis theorem approximation: check all 3 axes.
    """
    point = np.asarray(point, dtype=np.float64)
    center = corners.mean(axis=0)

    # Compute bbox axes from corners
    edges = corners[[1, 2, 4]] - corners[[0, 0, 0]]  # 3 edges from corner 0
    axes = edges / np.linalg.norm(edges, axis=1, keepdims=True)

    d = point - center
    half_extents = 0.5 * np.linalg.norm(edges, axis=1)

    for i in range(3):
        proj = np.abs(np.dot(d, axes[i]))
        if proj > half_extents[i] + 1e-6:
            return False
    return True


def load_gt_rooms(path):
    """Load GT room bounding boxes from rooms_<scene>.yaml."""
    with open(path) as f:
        data = yaml.safe_load(f)

    rooms = {}
    for room_id, bboxes in data.items():
        if not isinstance(bboxes, list):
            bboxes = [bboxes]
        rooms[room_id] = []
        for b in bboxes:
            corners = _bbox_from_center_extents_quat(
                b["center"], b["extents"], b["rotation"])
            rooms[room_id].append({
                "center": np.array(b["center"], dtype=np.float64),
                "extents": np.array(b["extents"], dtype=np.float64),
                "rotation": b["rotation"],
                "corners": corners,
            })

    print(f"GT rooms loaded: {len(rooms)} rooms")
    return rooms


def load_gt_objects(path):
    """Load GT task object bounding boxes from tasks_<scene>.yaml."""
    with open(path) as f:
        data = yaml.safe_load(f)

    objects = []
    for task_name, bboxes in data.items():
        if not isinstance(bboxes, list):
            bboxes = [bboxes]
        for b in bboxes:
            if not isinstance(b, dict) or "center" not in b:
                continue  # skip empty or malformed entries
            try:
                corners = _bbox_from_center_extents_quat(
                    b["center"], b["extents"], b["rotation"])
                objects.append({
                    "task": task_name,
                    "center": np.array(b["center"], dtype=np.float64),
                    "extents": np.array(b["extents"], dtype=np.float64),
                    "rotation": b["rotation"],
                    "corners": corners,
                })
            except (KeyError, TypeError, ValueError):
                continue

    print(f"GT objects loaded: {len(objects)} objects from {len(data)} tasks")
    return objects


def evaluate_room_assignment(places, gt_rooms):
    """Assign each place to the GT room containing its position.

    Returns:
        assignments: dict mapping room_id -> list of place indices
        unassigned: list of place indices not in any room
        confusion: dict with per-room stats
    """
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
    print(f"Assigned to rooms: {assigned} ({100*assigned/total:.1f}%)")
    print(f"Unassigned: {len(unassigned)} ({100*len(unassigned)/total:.1f}%)")
    print(f"Rooms covered: {len(assignments)}/{len(gt_rooms)}")
    for room_id in sorted(assignments.keys()):
        print(f"  Room {room_id}: {len(assignments[room_id])} places")
    for room_id in sorted(gt_rooms.keys()):
        if room_id not in assignments:
            print(f"  Room {room_id}: 0 places (MISSED)")

    return dict(assignments), unassigned


def evaluate_object_proximity(places, gt_objects):
    """Compute distance from each GT object to its nearest place.

    Returns:
        distances: list of (task_name, distance) tuples
        stats: min, max, mean, median distances
    """
    if not places:
        print("No places to evaluate object proximity against!")
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
        thresholds = [0.1, 0.25, 0.5, 1.0, 2.0]
        for t in thresholds:
            count = sum(1 for d in dists if d < t)
            print(f"  Within {t:.2f}m: {count}/{len(dists)} ({100*count/len(dists):.1f}%)")
    else:
        print("No distance data available.")

    return distances, {
        "min": float(min(dists)) if dists else 0,
        "max": float(max(dists)) if dists else 0,
        "mean": float(np.mean(dists)) if dists else 0,
        "median": float(np.median(dists)) if dists else 0,
    }


def evaluate_room_centroid_distance(places, gt_rooms):
    """For each GT room, find distance to nearest place centroid.

    This is an alternative metric when rooms don't have DSG room nodes.
    """
    if not places:
        print("No places to evaluate room centroids against!")
        return []

    place_positions = np.array([p["position"] for p in places])
    tree = cKDTree(place_positions)

    results = []
    for room_id, bboxes in gt_rooms.items():
        for j, bbox in enumerate(bboxes):
            center = bbox["center"]
            dist, idx = tree.query(center)
            results.append({
                "room_id": room_id,
                "bbox_idx": j,
                "room_center": center.tolist(),
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
        print("Per-room:")
        for r in results:
            print(f"  Room {r['room_id']}[{r['bbox_idx']}]: {r['distance']:.3f}m -> place {r['nearest_place_id']}")

    return results


def compute_2d_iou_numpy(corners1, corners2):
    """Compute approximate 2D XY-plane IoU between two oriented bounding boxes.

    Uses a sampling-based approach: discretize the overlapping bounding rectangle
    and count points falling inside both boxes.
    """
    # Project to XY plane
    xy1 = np.asarray(corners1)[:, :2]
    xy2 = np.asarray(corners2)[:, :2]

    # Get bounding rectangle of both
    all_xy = np.vstack([xy1, xy2])
    xmin, ymin = all_xy.min(axis=0) - 0.01
    xmax, ymax = all_xy.max(axis=0) + 0.01

    # Grid sampling
    grid_size = 200
    xs = np.linspace(xmin, xmax, grid_size)
    ys = np.linspace(ymin, ymax, grid_size)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    cell_area = dx * dy

    X, Y = np.meshgrid(xs, ys)
    points = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Point-in-convex-polygon test via cross product signs
    def points_in_hull(points, hull_xy):
        """Check if 2D points are inside convex polygon using cross products."""
        hull = np.asarray(hull_xy)
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
        # All cross products must have same sign (all >= 0 or all <= 0)
        all_pos = np.all(signs >= 0, axis=0)
        all_neg = np.all(signs <= 0, axis=0)
        return all_pos | all_neg

    in1 = points_in_hull(points, xy1)
    in2 = points_in_hull(points, xy2)

    intersection = np.sum(in1 & in2) * cell_area
    union = np.sum(in1 | in2) * cell_area

    return intersection / union if union > 0 else 0.0


def evaluate_place_gt_overlap(places, gt_objects, places_per_obj=3):
    """For each GT object, find nearest places and compute 2D XY IoU.

    Returns list of dicts with IoU results per object.
    """
    if not places:
        return []

    place_positions = np.array([p["position"] for p in places])
    tree = cKDTree(place_positions)

    results = []
    for obj in gt_objects:
        dists, indices = tree.query(obj["center"], k=min(places_per_obj, len(places)))
        if not isinstance(indices, np.ndarray):
            indices = np.array([indices])

        best_iou = 0.0
        best_place = None
        for idx in indices:
            place = places[idx]
            boundary = place.get("boundary")
            if boundary is None or len(boundary) < 3:
                continue
            try:
                iou = compute_2d_iou_numpy(boundary, obj["corners"])
            except Exception:
                iou = 0.0
            if iou > best_iou:
                best_iou = iou
                best_place = place

        results.append({
            "task": obj["task"],
            "best_iou": float(best_iou),
            "best_place_id": best_place["id"] if best_place else None,
            "obj_center": obj["center"].tolist(),
            "obj_extents": obj["extents"].tolist(),
        })

    ious = [r["best_iou"] for r in results]
    nonzero = [i for i in ious if i > 0]

    print(f"\n=== Place-GT Overlap (2D XY IoU) ===")
    print(f"Objects with IoU > 0: {len(nonzero)}/{len(ious)}")
    if nonzero:
        print(f"IoU stats: min={min(nonzero):.4f}, max={max(nonzero):.4f}, "
              f"mean={np.mean(nonzero):.4f}, median={np.median(nonzero):.4f}")
    else:
        print("All IoUs are 0 (places and GT objects occupy different regions)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Clio DSG against GT annotations")
    parser.add_argument("--dsg", required=True, help="Path to dsg.json")
    parser.add_argument("--rooms", default=None, help="Path to rooms_<scene>.yaml")
    parser.add_argument("--tasks", default=None, help="Path to tasks_<scene>.yaml")
    parser.add_argument("--output", default=None, help="Path to save evaluation results JSON")
    args = parser.parse_args()

    # Load DSG
    dsg_data, places, objects, rooms = load_dsg(args.dsg)

    results = {
        "dsg_path": args.dsg,
        "num_places": len(places),
        "num_objects": len(objects),
        "num_dsg_rooms": len(rooms),
        "num_mesh_points": len(dsg_data["mesh"]["points"]),
        "num_mesh_faces": len(dsg_data["mesh"]["faces"]),
    }

    # Room evaluation
    if args.rooms and Path(args.rooms).exists():
        gt_rooms = load_gt_rooms(args.rooms)
        room_assignments, unassigned = evaluate_room_assignment(places, gt_rooms)
        room_centroids = evaluate_room_centroid_distance(places, gt_rooms)
        results["room_assignment"] = {
            "per_room": {str(k): len(v) for k, v in room_assignments.items()},
            "unassigned_count": len(unassigned),
            "rooms_covered": len(room_assignments),
            "total_rooms": len(gt_rooms),
        }
        results["room_centroid_distances"] = room_centroids
    else:
        print(f"Rooms file not found or not specified: {args.rooms}")

    # Task object evaluation
    if args.tasks and Path(args.tasks).exists():
        gt_objects = load_gt_objects(args.tasks)
        obj_distances, proximity_stats = evaluate_object_proximity(places, gt_objects)
        results["object_proximity"] = {
            "distances": obj_distances,
            "stats": proximity_stats,
        }
    else:
        print(f"Tasks file not found or not specified: {args.tasks}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
