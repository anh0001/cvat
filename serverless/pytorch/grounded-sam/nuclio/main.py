import base64
import io
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable, List

import cv2
import groundingdino.datasets.transforms as T
import numpy as np
import torch
from GroundingDINO.groundingdino.util.inference import load_model, predict
from PIL import Image
from huggingface_hub import hf_hub_download
from segment_anything import SamPredictor, sam_model_registry
from torchvision.ops import box_convert

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = [label.strip() for label in os.getenv("LABELS", "cardboard,paper,shopping_bag").split(",") if label.strip()]
BOX_THRESHOLD = float(os.getenv("BOX_THRESHOLD", "0.35"))
TEXT_THRESHOLD = float(os.getenv("TEXT_THRESHOLD", "0.25"))
MIN_MASK_AREA = float(os.getenv("MIN_MASK_AREA", "200"))
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/opt/nuclio/models"))
GROUNDING_REPO_ID = os.getenv("GROUNDING_REPO_ID", "ShilongLiu/GroundingDINO")
GROUNDING_CONFIG = os.getenv("GROUNDING_CONFIG", "GroundingDINO_SwinT_OGC.py")
GROUNDING_WEIGHTS = os.getenv("GROUNDING_WEIGHTS", "groundingdino_swint_ogc.pth")
SAM_REPO_ID = os.getenv("SAM_REPO_ID")
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "sam_vit_h_4b8939.pth")
SAM_CHECKPOINT_URL = os.getenv(
    "SAM_CHECKPOINT_URL", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_h")


def _download_http(url: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as src, open(target_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return target_path


def _hf_download(repo: str, filename: str) -> Path:
    cache_dir = MODEL_CACHE_DIR / "hf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=repo, filename=filename, cache_dir=str(cache_dir)))


class ModelBundle:
    def __init__(self) -> None:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        config_path = _hf_download(GROUNDING_REPO_ID, GROUNDING_CONFIG)
        weights_path = _hf_download(GROUNDING_REPO_ID, GROUNDING_WEIGHTS)
        self.gd_model = load_model(str(config_path), str(weights_path), device=DEVICE)

        if SAM_REPO_ID:
            checkpoint_path = _hf_download(SAM_REPO_ID, SAM_CHECKPOINT)
        else:
            checkpoint_path = _download_http(SAM_CHECKPOINT_URL, MODEL_CACHE_DIR / SAM_CHECKPOINT)

        sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(checkpoint_path))
        sam_model.to(device=DEVICE)
        sam_model.eval()
        self.sam_predictor = SamPredictor(sam_model)
        self.min_area = MIN_MASK_AREA

    def _prepare_dino_inputs(self, pil_image: Image.Image):
        image_rgb = np.array(pil_image)
        tensor, _ = self.transform(pil_image, None)
        return image_rgb, tensor

    def _boxes_to_xyxy(self, boxes: torch.Tensor, height: int, width: int) -> np.ndarray:
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        scale = torch.tensor([width, height, width, height], dtype=xyxy.dtype)
        xyxy = xyxy * scale
        return xyxy.cpu().numpy()

    def run(
        self,
        pil_image: Image.Image,
        allowed_labels: Iterable[str],
        box_thr: float,
        text_thr: float,
        filter_thr: float,
    ) -> List[dict]:
        allowed = [label.strip() for label in allowed_labels if label.strip()]
        if not allowed:
            allowed = LABELS

        image_rgb, image_tensor = self._prepare_dino_inputs(pil_image)
        image_tensor = image_tensor.to(device=DEVICE)
        height, width = image_rgb.shape[:2]

        results: List[dict] = []
        self.sam_predictor.set_image(image_rgb)

        for label in allowed:
            boxes, logits, _ = predict(
                model=self.gd_model,
                image=image_tensor,
                caption=label,
                box_threshold=box_thr,
                text_threshold=text_thr,
                device=DEVICE,
            )

            if boxes is None or len(boxes) == 0:
                continue

            boxes_xyxy = self._boxes_to_xyxy(boxes, height, width)
            logits = logits.cpu().numpy()

            for box_coords, logit in zip(boxes_xyxy, logits):
                x1, y1, x2, y2 = box_coords.astype(int).tolist()
                sam_masks, sam_scores, _ = self.sam_predictor.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=True,
                )
                best_idx = int(np.argmax(sam_scores))
                best_mask = sam_masks[best_idx].astype(np.uint8)

                mask_area = float(best_mask.sum())
                if mask_area < self.min_area:
                    continue

                final_score = float(logit) * float(sam_scores[best_idx])
                if final_score < filter_thr:
                    continue

                contours, _ = cv2.findContours((best_mask * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) < self.min_area:
                    continue

                epsilon = max(1.0, 0.01 * cv2.arcLength(contour, True))
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if approx is None or len(approx) < 3:
                    continue

                points = approx.reshape(-1, 2).astype(float)
                pts_flat = [float(coord) for point in points for coord in point]
                if len(pts_flat) < 6:
                    continue

                results.append(
                    {
                        "confidence": final_score,
                        "label": label,
                        "type": "polygon",
                        "points": pts_flat,
                    }
                )

        return results


def init_context(context):
    context.logger.info("Loading GroundingDINO + SAM models...")
    context.user_data.bundle = ModelBundle()
    context.logger.info("Models ready on %s", DEVICE)


def _decode_event(event):
    data = event.body
    if isinstance(data, (bytes, bytearray)):
        data = json.loads(data.decode("utf-8"))
    return data


def handler(context, event):
    data = _decode_event(event)
    threshold = float(data.get("threshold", 0.5))
    box_thr = float(data.get("box_threshold", BOX_THRESHOLD))
    text_thr = float(data.get("text_threshold", TEXT_THRESHOLD))

    requested_labels = data.get("labels")
    allowed = LABELS
    if requested_labels:
        candidate_names: List[str] = []
        if isinstance(requested_labels, str):
            candidate_names = [requested_labels]
        elif isinstance(requested_labels, dict):
            name = requested_labels.get("name")
            if name:
                candidate_names = [name]
        elif isinstance(requested_labels, Iterable):
            for item in requested_labels:
                if isinstance(item, str):
                    candidate_names.append(item)
                elif isinstance(item, dict) and item.get("name"):
                    candidate_names.append(item["name"])
        filtered = [label for label in candidate_names if label in LABELS]
        if filtered:
            allowed = filtered

    image_bytes = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    preds = context.user_data.bundle.run(
        pil_image=image,
        allowed_labels=allowed,
        box_thr=box_thr,
        text_thr=text_thr,
        filter_thr=threshold,
    )

    return context.Response(
        body=json.dumps(preds),
        headers={"Content-Type": "application/json"},
        content_type="application/json",
        status_code=200,
    )
