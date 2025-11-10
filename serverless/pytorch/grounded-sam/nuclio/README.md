# Grounded-SAM for CVAT

Open-vocabulary object detection using GroundingDINO + Segment Anything Model (SAM) for automated polygon annotation in CVAT.

## Overview

This serverless function combines:
- **GroundingDINO**: Text-based object detection (generates bounding boxes from label text)
- **SAM**: Converts boxes to precise polygon masks
- **Output**: COCO-compatible polygon annotations with confidence scores

## Quick Start

### GPU Deployment (Recommended)

From the CVAT repo root:

```bash
./serverless/deploy_gpu.sh serverless/pytorch/grounded-sam
```

### CPU Deployment (Fallback)

```bash
./serverless/deploy_cpu.sh serverless/pytorch/grounded-sam
```

**Note**: CPU inference is significantly slower (~10-30x) but works without NVIDIA GPU.

## Verify Deployment

```bash
# Check function status
docker exec -it nuclio-dashboard nuctl get functions

# Expected output:
# NAMESPACE | NAME                     | PROJECT | STATE | REPLICAS
# nuclio    | pytorch-grounded-sam     | cvat    | ready | 1/1

# View logs
docker logs $(docker ps -q --filter "name=nuclio-pytorch-grounded-sam")
```

## Usage in CVAT

1. Create a CVAT Task with labels: `cardboard`, `paper`, `shopping_bag`
2. Go to **Task → Automatic annotation**
3. Select **Grounded-SAM (text → masks)**
4. Adjust threshold slider (default: 0.5)
5. Click **Annotate** to run on all frames

Results appear as polygon annotations with matching labels.

## Configuration

### Environment Variables (in function-*.yaml)

- `LABELS`: Comma-separated class names (must match CVAT task labels)
- `BOX_THRESHOLD`: GroundingDINO confidence threshold (default: 0.35)
- `TEXT_THRESHOLD`: Text-to-box matching threshold (default: 0.25)

### Tuning Recommendations

**Low recall (missing objects)**:
```yaml
BOX_THRESHOLD: "0.25"
TEXT_THRESHOLD: "0.20"
```

**Too many false positives**:
```yaml
BOX_THRESHOLD: "0.45"
TEXT_THRESHOLD: "0.30"
```

Adjust these in `function-gpu.yaml` and redeploy:
```bash
./serverless/deploy_gpu.sh serverless/pytorch/grounded-sam
```

## Memory Requirements

### GPU Version
- **Minimum**: 8 GB VRAM (ViT-L SAM variant)
- **Recommended**: 12+ GB VRAM (ViT-H SAM variant, default)

To reduce VRAM usage, edit `main.py`:
```python
# Change line ~31-32 from:
sam_ckpt = hf_hub_download(repo_id="facebook/sam-vit-huge", filename="sam_vit_h_4b8939.pth")
self.sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(DEVICE)

# To:
sam_ckpt = hf_hub_download(repo_id="facebook/sam-vit-large", filename="sam_vit_l_0b3195.pth")
self.sam = sam_model_registry["vit_l"](checkpoint=sam_ckpt).to(DEVICE)
```

### CPU Version
- **Minimum**: 8 GB RAM
- **Warning**: Inference takes 10-30 seconds per image

## Troubleshooting

### Function won't deploy
- Check disk space (models download ~2.5 GB on first run)
- Verify CUDA version matches (11.8 in base image)
- Review logs: `docker logs nuclio-dashboard`

### "No functions available" in CVAT
- Verify Nuclio is running: `docker ps | grep nuclio`
- Check CVAT can reach Nuclio:
  ```bash
  docker exec cvat_server curl http://nuclio:8070/api/functions
  ```
- Restart CVAT server:
  ```bash
  docker compose restart cvat_server
  ```

### OOM errors (GPU)
- Switch to ViT-L SAM (see Memory Requirements above)
- Reduce maxWorkers to 1 in `function-gpu.yaml` (already default)
- Ensure no other GPU processes are running

### Predictions are empty
- Verify task labels match `LABELS` env var exactly
- Try lowering `BOX_THRESHOLD` and `TEXT_THRESHOLD`
- Check function logs for errors:
  ```bash
  docker logs $(docker ps -q --filter "name=nuclio-pytorch-grounded-sam") | tail -50
  ```

### Wrong label assignments
- Ensure CVAT task labels are: `cardboard`, `paper`, `shopping_bag` (exact match)
- Check `function-gpu.yaml` → `spec` annotation has correct category IDs
- GroundingDINO uses text similarity; similar label names may cause confusion

## Model Details

### GroundingDINO
- **Model**: SwinT-OGC (88M parameters)
- **Source**: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- **License**: Apache 2.0

### Segment Anything (SAM)
- **Model**: ViT-H (default) or ViT-L (low-VRAM)
- **Source**: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- **License**: Apache 2.0

## Performance Benchmarks

On NVIDIA RTX 4090 (24GB VRAM):
- **ViT-H SAM**: ~1.5-2 seconds per image
- **ViT-L SAM**: ~1-1.3 seconds per image

On CPU (32-core Xeon):
- ~15-30 seconds per image (highly variable)

## License Compliance

This function uses permissive licenses only:
- GroundingDINO: Apache 2.0
- SAM: Apache 2.0
- PyTorch: BSD-3
- OpenCV: Apache 2.0

Safe for commercial use without restrictions.

## References

See [floor-aware-coco-synthesis/CVAT.md](../../../../floor-aware-coco-synthesis/CVAT.md) section B8 for integration details with the synthesis pipeline.
