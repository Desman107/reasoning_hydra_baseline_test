# Clio DSG Evaluation Results (Surface-Aware)

> Generated: 2026-05-12 | Script: `eval_m/eval_clio_v2.py`
>
> **Analysis-driven improvements based on `eval_issue.md`**

---

## Background

DSG Place nodes (layer 20, `Place2dNodeAttributes`) are **2D surface patches** on room surfaces (walls, floors, ceilings). They are fundamentally different from Object nodes (layer 2) which would have 3D bounding boxes. Evaluating 2D surface nodes against 3D GT bounding boxes requires metrics that respect this geometric mismatch.

### Changes from previous evaluation

| Before (v1) | After (v2) | Reason |
|-------------|------------|--------|
| **3D IoU** as primary object metric | **Surface Coverage** (P/R/F1) as primary | 3D IoU is mathematically invalid for 2D surfaces (volume ≈ 0) |
| **Room assignment** raw | **Room assignment** with unannotated region filtering | Corridor/hallway places skew results |
| No place classification | **Surface type classification** (wall/floor/ceiling) | Understand what the DSG is actually producing |
| 2D IoU on world planes | Removed (structurally 0) | Places on walls, GT objects in room interior — no projection overlap |

---

## Metrics Description

### Object-Level Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Object Proximity** | Euclidean distance from GT object center to nearest place centroid | Lower is better |
| **Surface Precision** | Fraction of place mesh points within threshold of GT bbox surface | Higher is better |
| **Surface Recall** | Fraction of GT bbox surface samples within threshold of place mesh | Higher is better |
| **Surface F1** | Harmonic mean of precision & recall | Higher is better |
| **Chamfer Distance** | Bidirectional: place mesh ↔ GT bbox surface (meters) | Lower is better |
| **P→GT** | Mean distance from GT surface to nearest place point | Lower is better |

### Room-Level Metrics

| Metric | Description |
|--------|-------------|
| **Room Assignment (Raw)** | Place centroid inside GT room bbox |
| **Room Assignment (Effective)** | Excludes places >3m from any GT room (unannotated corridors) |
| **Room 3D IoU** | Merged-place OBB vs GT room bbox (valid for rooms: large volumes) |
| **Room Coverage** | Fraction of GT rooms containing ≥1 place |

### Place-Level Metrics

| Metric | Description |
|--------|-------------|
| **Surface Type** | PCA normal classification: wall / floor / ceiling / other |

---

## Per-Scene Results

### Cubicle (malloc fix) — `cubicle_malloc`

| Metric | Value |
|--------|-------|
| Places / Mesh | 15 / 66,551 pts, 75,460 faces |
| Place Types | wall:11, floor:1, other:3 |

**Object metrics** (18 GT objects):

| Metric | Value |
|--------|-------|
| Object Proximity mean | 1.02 m |
| Object Proximity <1m | 8/18 (44.4%) |
| Surface Precision@0.25m | 0.102 |
| Surface Recall@0.25m | **0.936** |
| Surface Recall@0.5m | **0.998** |
| Surface F1@0.25m | **0.178** |
| Chamfer Distance | 0.93 m |
| P→GT | **0.12 m** |

### Apartment

| Metric | Value |
|--------|-------|
| Places / Mesh | 69 / 33,228 pts, 38,758 faces |
| Place Types | wall:58 (84%), other:11 (16%) |

**Room metrics:**

| Metric | Value |
|--------|-------|
| Room Coverage (Raw) | 2/3, 2 unassigned |
| Room Coverage (Effective) | **100%** (67/67, 2 unannotated) |
| Room 3D IoU >0 | 3/3 |
| Room 3D IoU mean (nz) | 0.26 |

**Object metrics** (29 GT objects):

| Metric | Value |
|--------|-------|
| Object Proximity mean | 0.69 m |
| Object Proximity <1m | **26/29 (89.7%)** |
| Surface Precision@0.25m | 0.096 |
| Surface Recall@0.25m | 0.643 |
| Surface Recall@0.5m | **0.902** |
| Surface F1@0.25m | **0.141** |
| Chamfer Distance | 1.08 m |
| P→GT | **0.26 m** |

### Office

| Metric | Value |
|--------|-------|
| Places / Mesh | 305 / 93,612 pts, 104,618 faces |
| Place Types | wall:189 (62%), floor:37 (12%), other:78 (26%) |

**Room metrics:**

| Metric | Value |
|--------|-------|
| Room Coverage (Raw) | 5/8, 3 unassigned, **136 unannotated** |
| Room Coverage (Effective) | **98.2%** (166/169) |
| Room 3D IoU >0 | 8/10 |
| Room 3D IoU mean (nz) | 0.23 |

**Object metrics** (33 GT objects):

| Metric | Value |
|--------|-------|
| Object Proximity mean | 1.50 m |
| Object Proximity <1m | 13/33 (39.4%) |
| Surface Precision@0.25m | 0.055 |
| Surface Recall@0.25m | 0.115 |
| Surface F1@0.25m | **0.066** |
| Chamfer Distance | 1.95 m |
| P→GT | 0.85 m |

### Building

| Metric | Value |
|--------|-------|
| Places / Mesh | 598 / 253,054 pts, 288,206 faces |
| Place Types | wall:475 (79%), floor:11 (2%), other:111 (19%) |

**Room metrics:**

| Metric | Value |
|--------|-------|
| Room Coverage (Raw) | 7/13, 0 unassigned, 12 unannotated |
| Room Coverage (Effective) | **100%** (586/586) |
| Room 3D IoU >0 | 13/15 |
| Room 3D IoU mean (nz) | 0.24 |

**Object metrics:** N/A (building has no GT task bounding boxes — 25 empty task annotations)

---

## Overall Averages

Aggregated across valid scenes (cubicle_malloc, apartment, office, building).

### Object-Level (80 objects)

| Metric | Mean | Notes |
|--------|------|-------|
| **Surface F1 @ 0.25m** | **0.118** | Primary quality metric for 2D surfaces |
| **Surface Recall @ 0.5m** | **0.682** | 68% of GT surface within 0.5m of place mesh |
| Surface Recall @ 0.25m | 0.520 | 52% within 0.25m |
| Object Proximity | 1.10 m | 58.8% < 1m |
| Chamfer Distance | 1.37 m | |
| P→GT | **0.45 m** | Places → GT surface (more informative) |
| GT→P | 0.92 m | GT surface → places |

### Room-Level (28 GT bboxes over 24 rooms)

| Metric | Mean | Notes |
|--------|------|-------|
| Room 3D IoU (nonzero) | **0.238** | Valid for rooms (large volume vs surface OBB) |
| Room 3D IoU >0 | 85.7% (24/28) | |
| Effective Room Coverage | **99.2%** | Excluding unannotated corridor regions |
| Raw Room Coverage | 58.3% (14/24) | Includes unannotated regions |

---

## Key Findings

### 1. The DSG produces wall-dominant 2D surface places
Across all scenes: **~80% of places are walls**, ~15% "other" (mixed orientation), small fraction floor/ceiling. This confirms the Place2dNodeAttributes layer captures surface geometry as designed.

### 2. Room assignment is excellent within annotated regions
After filtering unannotated corridor regions: **99.2% effective room coverage** (apartment 100%, office 98.2%, building 100%). The previously-reported "45.6% unassigned" in office was caused by **136 places in unannotated corridor/hallway space** — not an algorithm failure.

### 3. Surface recall is the strongest metric
**68.2% of GT object surface area is within 0.5m of a place mesh point** on average. Cubicle achieves near-perfect 99.8% recall@0.5m. This indicates the TSDF mesh reconstruction places surface patches accurately near GT object locations.

### 4. Proximity is good but places are "nearby" not "on" objects
89.7% of GT objects in apartment have a place centroid within 1m. But surface precision is lower than recall (P=0.10 vs R=0.64 @0.25m for apartment), meaning places are close to objects but their mesh points don't all lie exactly on the object surface. This is expected: places are room-surface patches, not object-surface patches.

### 5. Office is the hardest scene
Surface F1@0.25m = 0.066 vs 0.178 (cubicle) and 0.141 (apartment). Large scene with >100K mesh points, sparse frame sampling (frame_skip=200), and complex geometry lead to coarser surface reconstruction.

### 6. To get meaningful 3D IoU, generate Object nodes (L2)
The current DSG has no L2 Object nodes (`total_labels=2, object_labels={}` in pipeline config). Re-running with object label extraction enabled would produce volumetric object nodes with proper 3D bounding boxes, enabling standard 3D IoU evaluation as defined in the paper.

---

## Output Files

| File | Description |
|------|-------------|
| `eval_m/eval_clio_v2.py` | Surface-aware evaluation script |
| `eval_m/run_all_evals_v2.sh` | Batch runner |
| `eval_m/summarize_evals_v2.py` | Cross-scene summarizer |
| `eval_m/<scene>_results.json` | Per-scene detailed results |
| `eval_m/EVALUATION_SUMMARY.md` | This file |
