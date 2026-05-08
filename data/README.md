# Clio Local Metadata Repo

This repository tracks only lightweight metadata for the local Clio dataset.

Raw scene assets are intentionally ignored:
- bag files
- RGB/depth images
- dense/sparse reconstructions
- extracted poses and summaries

Tracked content lives in:
- `annotations/`: benchmark annotations and task timelines
- `scripts/`: local dataset processing and validation scripts

Current annotation plan:
- `annotations/multi_task/<scene>.yaml` stores the per-scene multi-task timeline
- each task entry records when a task is injected and any notes needed for evaluation

The actual scene data remains in the sibling scene folders under this same root, but is not tracked by git.
