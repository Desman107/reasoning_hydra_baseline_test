#!/usr/bin/env python3
"""Summarize enhanced evaluation results across all Clio scenes.

Usage:
    python3 eval_m/summarize_evals_v2.py eval_m/
"""

import json
import sys
from pathlib import Path

import numpy as np


def _safe_val(seq, fn, default="N/A", fmt=".4f"):
    """Safely compute a stat on a sequence, returning default if empty."""
    if not seq:
        return default
    val = fn(seq)
    if isinstance(val, float) and fmt:
        return f"{val:{fmt}}"
    return str(val)


def main():
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval_m")

    scenes = ["cubicle_malloc", "cubicle", "apartment", "office", "building"]

    print("=" * 95)
    print("  Clio DSG Enhanced Evaluation Summary")
    print("=" * 95)

    all_data = {}

    for scene in scenes:
        result_path = result_dir / f"{scene}_results.json"
        if not result_path.exists():
            print(f"\n{scene}: NO RESULTS FILE")
            continue

        with open(result_path) as f:
            data = json.load(f)

        all_data[scene] = data

        print(f"\n--- {scene} ---")
        print(f"  Places: {data['num_places']}, Objects: {data['num_objects']}, "
              f"DSG Rooms: {data['num_dsg_rooms']}")
        print(f"  Mesh: {data['num_mesh_points']:,} points, {data['num_mesh_faces']:,} faces")

        # Room metrics
        if "room_assignment" in data:
            ra = data["room_assignment"]
            print(f"  Room Assignment: {ra['rooms_covered']}/{ra['total_rooms']} covered, "
                  f"{ra['unassigned_count']} unassigned")

        if "room_3d_iou" in data and data["room_3d_iou"]:
            ious = [r["best_3d_iou"] for r in data["room_3d_iou"]]
            nz = [i for i in ious if i > 0]
            print(f"  Room 3D IoU:     >0={len(nz)}/{len(ious)}, "
                  f"max={_safe_val(ious, max)}, mean(nz)={_safe_val(nz, np.mean)}")

        # Object metrics
        if "object_proximity" in data:
            stats = data["object_proximity"].get("stats", {})
            if stats and stats.get("max", 0) > 0:
                print(f"  Obj Proximity:   min={stats['min']:.3f}m, mean={stats['mean']:.3f}m, "
                      f"median={stats['median']:.3f}m")

        if "object_3d_iou" in data and data["object_3d_iou"]:
            ious = [r["best_3d_iou"] for r in data["object_3d_iou"]]
            nz = [i for i in ious if i > 0]
            print(f"  Obj 3D IoU:      >0={len(nz)}/{len(ious)}, "
                  f"max={_safe_val(ious, max)}, mean(nz)={_safe_val(nz, np.mean)}")

        if "object_chamfer_distance" in data and data["object_chamfer_distance"]:
            chamfers = [r["chamfer"] for r in data["object_chamfer_distance"]
                        if r["chamfer"] is not None]
            if chamfers:
                p2g = [r["place_to_gt_mean"] for r in data["object_chamfer_distance"]
                       if r["place_to_gt_mean"] is not None]
                g2p = [r["gt_to_place_mean"] for r in data["object_chamfer_distance"]
                       if r["gt_to_place_mean"] is not None]
                print(f"  Chamfer (m):     mean={np.mean(chamfers):.3f}, "
                      f"P->GT={np.mean(p2g):.3f}, GT->P={np.mean(g2p):.3f}")

        if "object_2d_iou_multiple" in data and data["object_2d_iou_multiple"]:
            for plane in ["xy", "xz", "yz", "best"]:
                key = f"iou_{plane}"
                if key not in data["object_2d_iou_multiple"][0]:
                    continue
                ious = [r[key] for r in data["object_2d_iou_multiple"]]
                nz = [i for i in ious if i > 0]
                if nz:
                    print(f"  Obj 2D IoU ({plane}): >0={len(nz)}/{len(ious)}, "
                          f"mean(nz)={np.mean(nz):.4f}")
                else:
                    print(f"  Obj 2D IoU ({plane}): all 0")

        if "object_point_coverage" in data and data["object_point_coverage"]:
            precs = [r["precision"] for r in data["object_point_coverage"]]
            recalls = [r["recall"] for r in data["object_point_coverage"]]
            if precs:
                print(f"  Pt Precision:    mean={np.mean(precs):.4f}")
                print(f"  Pt Recall:       mean={np.mean(recalls):.4f}")

    # Cross-scene comparison table
    if all_data:
        print(f"\n{'=' * 95}")
        print(f"  Cross-Scene Comparison")
        print(f"{'=' * 95}")
        header = (f"{'Scene':<16} {'Places':>7} {'Obj3DIoU>0':>11} {'ObjProx<1m':>11} "
                  f"{'Chamfer':>9} {'RoomCov':>10} {'Room3DIoU>0':>12}")
        print(header)
        print("-" * 95)
        for scene in scenes:
            if scene not in all_data:
                continue
            d = all_data[scene]

            obj3d_str = "N/A"
            if d.get("object_3d_iou"):
                ious = [r["best_3d_iou"] for r in d["object_3d_iou"]]
                obj3d_str = f"{sum(1 for i in ious if i>0)}/{len(ious)}"

            objprox_str = "N/A"
            if d.get("object_proximity", {}).get("distances"):
                dists = [r["distance"] for r in d["object_proximity"]["distances"]]
                if dists:
                    objprox_str = f"{sum(1 for d in dists if d<1.0)}/{len(dists)}"

            chamfer_str = "N/A"
            if d.get("object_chamfer_distance"):
                chamfers = [r["chamfer"] for r in d["object_chamfer_distance"]
                           if r["chamfer"] is not None]
                if chamfers:
                    chamfer_str = f"{np.mean(chamfers):.2f}"

            roomcov_str = "N/A"
            if "room_assignment" in d:
                roomcov_str = f"{d['room_assignment']['rooms_covered']}/{d['room_assignment']['total_rooms']}"

            room3d_str = "N/A"
            if d.get("room_3d_iou"):
                ious = [r["best_3d_iou"] for r in d["room_3d_iou"]]
                room3d_str = f"{sum(1 for i in ious if i>0)}/{len(ious)}"

            print(f"{scene:<16} {d['num_places']:>7} {obj3d_str:>11} {objprox_str:>11} "
                  f"{chamfer_str:>9} {roomcov_str:>10} {room3d_str:>12}")

    print("\nDone.")


if __name__ == "__main__":
    main()
