#!/usr/bin/env python3
"""Summarize evaluation results across all Clio scenes.

Usage:
    python3 eval/summarize_evals.py /tmp/clio_output
"""

import json
import sys
from pathlib import Path


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/clio_output")

    scenes = ["cubicle", "apartment", "office", "building"]
    all_results = {}

    print("=" * 70)
    print("  Clio DSG Evaluation Summary")
    print("=" * 70)

    for scene in scenes:
        dsg_path = output_dir / scene / "dsg.json"
        eval_path = output_dir / scene / "eval_results.json"

        if not dsg_path.exists():
            print(f"\n{scene}: NO DSG OUTPUT")
            continue

        # Load DSG stats
        with open(dsg_path) as f:
            dsg = json.load(f)

        num_nodes = len(dsg["nodes"])
        num_edges = len(dsg["edges"])
        num_mesh_pts = len(dsg["mesh"]["points"])
        num_mesh_faces = len(dsg["mesh"]["faces"])

        # Count by layer
        from collections import Counter
        layer_counts = Counter(n["layer"] for n in dsg["nodes"])
        layer_names = {2: "objects", 3: "places", 4: "rooms",
                       5: "buildings", 20: "mesh_places"}

        print(f"\n--- {scene} ---")
        print(f"  Nodes: {num_nodes} ({', '.join(f'{layer_names.get(k, k)}:{v}' for k, v in sorted(layer_counts.items()))})")
        print(f"  Edges: {num_edges}")
        print(f"  Mesh: {num_mesh_pts:,} points, {num_mesh_faces:,} faces")

        # Load eval results if available
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)

            if "room_assignment" in eval_data:
                ra = eval_data["room_assignment"]
                print(f"  Room Assignment: {ra['rooms_covered']}/{ra['total_rooms']} rooms covered, "
                      f"{ra['unassigned_count']} places unassigned")

            if "room_centroid_distances" in eval_data:
                rcd = eval_data["room_centroid_distances"]
                if rcd:
                    dists = [r["distance"] for r in rcd]
                    print(f"  Room Centroid Dist: min={min(dists):.2f}m, mean={sum(dists)/len(dists):.2f}m")

            if "object_proximity" in eval_data:
                op = eval_data["object_proximity"]
                stats = op.get("stats", {})
                if stats:
                    print(f"  Object Proximity: min={stats.get('min', 0):.2f}m, "
                          f"mean={stats.get('mean', 0):.2f}m, "
                          f"median={stats.get('median', 0):.2f}m")

            all_results[scene] = {
                "dsg_nodes": num_nodes,
                "dsg_edges": num_edges,
                "mesh_points": num_mesh_pts,
                "eval": eval_data,
            }
        else:
            print("  (no eval results yet)")
            all_results[scene] = {
                "dsg_nodes": num_nodes,
                "dsg_edges": num_edges,
                "mesh_points": num_mesh_pts,
            }

    # Cross-scene comparison
    if all_results:
        print(f"\n=== Cross-Scene Comparison ===")
        print(f"{'Scene':<12} {'Nodes':>8} {'Edges':>8} {'Mesh Pts':>12} {'Mesh Faces':>12}")
        print("-" * 55)
        for scene in scenes:
            if scene in all_results:
                r = all_results[scene]
                dsg_path_check = output_dir / scene / "dsg.json"
                if dsg_path_check.exists():
                    with open(dsg_path_check) as f:
                        d = json.load(f)
                    print(f"{scene:<12} {len(d['nodes']):>8} {len(d['edges']):>8} "
                          f"{len(d['mesh']['points']):>12,} {len(d['mesh']['faces']):>12,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
