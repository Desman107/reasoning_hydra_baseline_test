#!/usr/bin/env python3
"""Summarize surface-aware evaluation results across all Clio scenes.

Usage:
    python3 eval_m/summarize_evals_v2.py eval_m/
"""

import json
import sys
from pathlib import Path

import numpy as np


def _safe_val(seq, fn, default="N/A", fmt=".4f"):
    if not seq:
        return default
    val = fn(seq)
    if isinstance(val, float) and fmt:
        return f"{val:{fmt}}"
    return str(val)


def main():
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval_m")

    scenes = ["cubicle_malloc", "apartment", "office", "building"]

    print("=" * 100)
    print("  Clio DSG Surface-Aware Evaluation Summary")
    print("=" * 100)

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

        # Place surface types
        if "place_surface_types" in data:
            from collections import Counter
            tc = Counter(p["type"] for p in data["place_surface_types"])
            print(f"  Place Types:  {', '.join(f'{k}:{v}' for k, v in sorted(tc.items()))}")

        # Room metrics
        if "room_assignment" in data:
            ra = data["room_assignment"]
            filt_info = ""
            if "unannotated_count" in ra:
                filt_info = (f", {ra['unannotated_count']} unannotated"
                             f" (> {ra.get('filter_distance_m', '?')}m)")
            print(f"  Room Raw:     {data.get('room_assignment_raw', ra)['rooms_covered']}"
                  f"/{ra['total_rooms']} covered, "
                  f"{ra.get('unassigned_count', '?')} unassigned{filt_info}")
            if "effective_total" in ra:
                eff = ra["effective_total"] - ra.get("unassigned_count", 0)
                print(f"  Room Effective: {eff}/{ra['effective_total']} assigned "
                      f"({100*eff/ra['effective_total']:.1f}%) excl. unannotated")

        if "room_3d_iou" in data and data["room_3d_iou"]:
            ious = [r["best_3d_iou"] for r in data["room_3d_iou"]]
            nz = [i for i in ious if i > 0]
            print(f"  Room 3D IoU:  >0={len(nz)}/{len(ious)}, "
                  f"max={_safe_val(ious, max)}, mean(nz)={_safe_val(nz, np.mean)}")

        # Object metrics
        if "object_proximity" in data:
            stats = data["object_proximity"].get("stats", {})
            if stats and stats.get("max", 0) > 0:
                dists = [r["distance"] for r in data["object_proximity"]["distances"]]
                lt1m = sum(1 for d in dists if d < 1.0)
                print(f"  Obj Proximity: mean={stats['mean']:.3f}m, median={stats['median']:.3f}m, "
                      f"<1m={lt1m}/{len(dists)}")

        if "object_surface_coverage" in data and data["object_surface_coverage"]:
            for thresh in ["0.1", "0.25", "0.5"]:
                f1s = [r["thresholds"][thresh]["f1"]
                       for r in data["object_surface_coverage"]]
                precs = [r["thresholds"][thresh]["precision"]
                        for r in data["object_surface_coverage"]]
                recalls = [r["thresholds"][thresh]["recall"]
                          for r in data["object_surface_coverage"]]
                nz = sum(1 for f in f1s if f > 0)
                if nz > 0:
                    print(f"  SurfCov@{thresh}m:  P={np.mean(precs):.3f}, R={np.mean(recalls):.3f}, "
                          f"F1={np.mean(f1s):.3f} ({nz}/{len(f1s)} >0)")

        if "object_chamfer_distance" in data and data["object_chamfer_distance"]:
            chamfers = [r["chamfer"] for r in data["object_chamfer_distance"]
                       if r["chamfer"] is not None]
            if chamfers:
                p2g = [r["place_to_gt_mean"] for r in data["object_chamfer_distance"]
                       if r["place_to_gt_mean"] is not None]
                print(f"  Chamfer (m):   mean={np.mean(chamfers):.3f}, "
                      f"P->GT={np.mean(p2g):.3f}")

    # Cross-scene comparison
    if all_data:
        print(f"\n{'=' * 100}")
        print(f"  Cross-Scene Comparison")
        print(f"{'=' * 100}")
        header = (f"{'Scene':<16} {'Places':>7} {'SurfF1@.25':>11} {'Prox<1m':>9} "
                  f"{'Chamfer':>9} {'RoomEff%':>9} {'Room3DIoU':>10}")
        print(header)
        print("-" * 100)
        for scene in scenes:
            if scene not in all_data:
                continue
            d = all_data[scene]

            # Surface F1 @ 0.25m
            sf1_str = "N/A"
            if d.get("object_surface_coverage"):
                f1s = [r["thresholds"]["0.25"]["f1"]
                       for r in d["object_surface_coverage"]]
                if f1s:
                    sf1_str = f"{np.mean(f1s):.3f}"

            # Proximity <1m
            prox_str = "N/A"
            if d.get("object_proximity", {}).get("distances"):
                dists = [r["distance"] for r in d["object_proximity"]["distances"]]
                if dists:
                    prox_str = f"{sum(1 for d in dists if d<1.0)}/{len(dists)}"

            # Chamfer
            chamfer_str = "N/A"
            if d.get("object_chamfer_distance"):
                chamfers = [r["chamfer"] for r in d["object_chamfer_distance"]
                           if r["chamfer"] is not None]
                if chamfers:
                    chamfer_str = f"{np.mean(chamfers):.2f}"

            # Room effective %
            room_str = "N/A"
            if "room_assignment" in d:
                ra = d["room_assignment"]
                if "effective_total" in ra and ra["effective_total"] > 0:
                    eff = ra["effective_total"] - ra.get("unassigned_count", 0)
                    room_str = f"{100*eff/ra['effective_total']:.0f}%"

            # Room 3D IoU
            room3d_str = "N/A"
            if d.get("room_3d_iou"):
                ious = [r["best_3d_iou"] for r in d["room_3d_iou"]]
                nz = [i for i in ious if i > 0]
                room3d_str = f"{len(nz)}/{len(ious)}"

            print(f"{scene:<16} {d['num_places']:>7} {sf1_str:>11} {prox_str:>9} "
                  f"{chamfer_str:>9} {room_str:>9} {room3d_str:>10}")

    # Overall averages
    print(f"\n{'=' * 100}")
    print(f"  Overall Averages (across valid scenes)")
    print(f"{'=' * 100}")

    all_surf_f1_025 = []
    all_surf_recall_05 = []
    all_prox = []
    all_chamfer = []
    all_p2gt = []
    all_room_iou_nz = []

    for scene, d in all_data.items():
        if d.get("object_surface_coverage"):
            for r in d["object_surface_coverage"]:
                all_surf_f1_025.append(r["thresholds"]["0.25"]["f1"])
                all_surf_recall_05.append(r["thresholds"]["0.5"]["recall"])
        if d.get("object_proximity", {}).get("distances"):
            for r in d["object_proximity"]["distances"]:
                all_prox.append(r["distance"])
        if d.get("object_chamfer_distance"):
            for r in d["object_chamfer_distance"]:
                if r["chamfer"] is not None:
                    all_chamfer.append(r["chamfer"])
                    all_p2gt.append(r["place_to_gt_mean"])
        if d.get("room_3d_iou"):
            for r in d["room_3d_iou"]:
                if r["best_3d_iou"] > 0:
                    all_room_iou_nz.append(r["best_3d_iou"])

    print(f"  Surface F1@0.25m:   mean={np.mean(all_surf_f1_025):.4f} "
          f"(n={len(all_surf_f1_025)})" if all_surf_f1_025 else "  N/A")
    print(f"  Surface Recall@0.5m: mean={np.mean(all_surf_recall_05):.4f} "
          f"(n={len(all_surf_recall_05)})" if all_surf_recall_05 else "  N/A")
    print(f"  Object Proximity:    mean={np.mean(all_prox):.3f}m, "
          f"<1m={sum(1 for d in all_prox if d<1.0)}/{len(all_prox)}"
          if all_prox else "  N/A")
    print(f"  Chamfer Distance:    mean={np.mean(all_chamfer):.3f}m "
          f"(n={len(all_chamfer)})" if all_chamfer else "  N/A")
    print(f"  P->GT mean:          {np.mean(all_p2gt):.3f}m" if all_p2gt else "  N/A")
    print(f"  Room 3D IoU (nz):    mean={np.mean(all_room_iou_nz):.4f} "
          f"(n={len(all_room_iou_nz)})" if all_room_iou_nz else "  N/A")

    print("\nDone.")


if __name__ == "__main__":
    main()
