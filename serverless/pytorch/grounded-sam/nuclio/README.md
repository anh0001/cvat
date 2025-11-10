# Grounded-SAM Nuclio Function

This directory contains a self-contained Nuclio function that pairs
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) with
[Segment Anything](https://github.com/facebookresearch/segment-anything)
to create polygons for the repo's floor-item labels from text prompts.

## Usage

1. Clone the official `cvat` repo next to this project if you have not
   already:

   ```bash
   /workspace
   ├── floor-aware-coco-synthesis
   └── cvat
   ```

2. Copy (or symlink) this folder into the CVAT serverless tree:

   ```bash
   rsync -a --delete \
     /workspace/floor-aware-coco-synthesis/cvat_serverless/grounded-sam/ \
     /workspace/cvat/serverless/pytorch/grounded-sam/
   ```

   The destination must contain the `nuclio/` folder with
   `function-gpu.yaml`, `function-cpu.yaml`, and `main.py`.

3. From the CVAT repo root, deploy the function (GPU example):

   ```bash
   cd /workspace/cvat
   ./serverless/deploy_gpu.sh serverless/pytorch/grounded-sam
   ```

   For CPU-only hosts, run `deploy_cpu.sh` instead. The helper script
   wires the function into the `cvat_cvat` Docker network so CVAT can
   discover it automatically.

4. In CVAT, open *Actions -> Automatic Annotation* and select
   "Grounded-SAM (text -> masks)". Adjust the threshold slider if needed
   and run the job.

## Customisation

The function exposes a few Nuclio environment variables:

- `LABELS` (comma list) – label prompts the function will iterate over
- `BOX_THRESHOLD`, `TEXT_THRESHOLD` – GroundingDINO filters
- `MIN_MASK_AREA` – discard very small SAM masks
- `MODEL_CACHE_DIR`, `SAM_MODEL_TYPE`, `SAM_CHECKPOINT_URL` – override
  model cache path, SAM variant, or checkpoint source

Update the YAML files before deployment to tune any of these values.
