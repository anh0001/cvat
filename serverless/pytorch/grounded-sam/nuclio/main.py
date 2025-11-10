import base64, io, json, os
import numpy as np
from PIL import Image
import cv2
import torch

from huggingface_hub import hf_hub_download
try:
    from groundingdino.util.inference import load_model, predict
except ImportError:
    # Fallback for groundingdino-py package
    from groundingdino import load_model as gd_load_model
    from groundingdino import predict as gd_predict
    load_model = gd_load_model
    predict = gd_predict
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = [s.strip() for s in os.getenv("LABELS", "cardboard,paper,shopping_bag").split(",") if s.strip()]
BOX_THRESHOLD = float(os.getenv("BOX_THRESHOLD", "0.35"))
TEXT_THRESHOLD = float(os.getenv("TEXT_THRESHOLD", "0.25"))


class ModelBundle:
    def __init__(self):
        # GroundingDINO config + weights
        gd_cfg = hf_hub_download(
            repo_id="ShilongLiu/GroundingDINO", filename="GroundingDINO_SwinT_OGC.py"
        )
        gd_wt = hf_hub_download(
            repo_id="ShilongLiu/GroundingDINO", filename="groundingdino_swint_ogc.pth"
        )
        self.gd_model = load_model(gd_cfg, gd_wt).to(DEVICE).eval()

        # SAM (ViT-H). For lower VRAM, switch to ViT-L:
        #   repo: "facebook/sam-vit-large", filename: "sam_vit_l_0b3195.pth"
        sam_ckpt = hf_hub_download(
            repo_id="facebook/sam-vit-huge", filename="sam_vit_h_4b8939.pth"
        )
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(DEVICE)
        self.sam_predictor = SamPredictor(self.sam)

    def run(self, pil_image, allowed_labels, box_thr, text_thr, filter_thr):
        image_np = np.array(pil_image.convert("RGB"))
        h, w = image_np.shape[:2]
        self.sam_predictor.set_image(image_np)

        results = []

        for label in allowed_labels:
            boxes, logits, phrases = predict(
                model=self.gd_model,
                image=image_np,
                caption=label,
                box_threshold=box_thr,
                text_threshold=text_thr,
                device=DEVICE,
            )

            if boxes is None or len(boxes) == 0:
                continue

            # scale to pixel coords if normalized
            if float(np.max(boxes)) <= 1.0:
                boxes = boxes * np.array([w, h, w, h])

            for b, conf in zip(boxes, logits):
                x1, y1, x2, y2 = [int(v) for v in b.tolist()]
                sam_masks, sam_scores, _ = self.sam_predictor.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=True,
                )

                best_idx = int(np.argmax(sam_scores))
                best_mask = sam_masks[best_idx].astype(np.uint8)

                # combine DINO (sigmoid) and SAM score
                final_score = float((1 / (1 + np.exp(-float(conf)))) * float(sam_scores[best_idx]))
                if final_score < float(filter_thr):
                    continue

                cnts, _ = cv2.findContours((best_mask * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                cnt = max(cnts, key=cv2.contourArea)
                eps = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True)
                pts = approx.reshape(-1, 2).astype(float).tolist()
                pts_flat = [p for xy in pts for p in xy]

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
    context.logger.info("Initializing models...")
    context.user_data.bundle = ModelBundle()
    context.logger.info("Models ready")


def handler(context, event):
    data = event.body
    threshold = float(data.get("threshold", 0.5))

    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf).convert("RGB")

    preds = context.user_data.bundle.run(
        pil_image=image,
        allowed_labels=LABELS,
        box_thr=BOX_THRESHOLD,
        text_thr=TEXT_THRESHOLD,
        filter_thr=threshold,
    )

    return context.Response(
        body=json.dumps(preds),
        headers={},
        content_type="application/json",
        status_code=200,
    )
