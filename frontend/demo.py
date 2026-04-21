"""
demo.py — local body fat estimator demo
----------------------------------------
Drag-and-drop an image, it crops the torso via YOLOv8 pose,
runs your trained ResNet-18 weights, and shows the predicted
body fat bucket.

Usage:
    python demo.py --weights best_model.pth

Requirements:
    pip install gradio torch torchvision ultralytics opencv-python pillow
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
LABEL_MAP = {
    0: "0–5%",
    1: "5–10%",
    2: "10–15%",
    3: "15–20%",
    4: "20–25%",
    5: "25–30%",
    6: "30%+",
}
NUM_CLASSES = 7
IMG_SIZE    = 224

# YOLOv8 keypoint indices
SHOULDER_IDS = [5, 6]
HIP_IDS      = [11, 12]

# ── Models (loaded once) ──────────────────────────────────────────────────────
_yolo  = None
_resnet = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def get_yolo():
    global _yolo
    if _yolo is None:
        _yolo = YOLO("yolov8n-pose.pt")
    return _yolo


def load_resnet(weights_path: str):
    global _resnet
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=_device))
    model.eval().to(_device)
    _resnet = model


# ── Torso crop (from crop_torso.py) ──────────────────────────────────────────
def crop_torso(image_pil: Image.Image, min_confidence=0.5):
    """Returns (cropped_pil, annotated_pil, error_str|None)."""
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]

    results = get_yolo()(img_bgr, verbose=False)

    if not results or results[0].keypoints is None or len(results[0].keypoints.data) == 0:
        return None, image_pil, "No pose detected in image."

    kpts = results[0].keypoints.data[0].cpu().numpy()

    def get_pts(ids):
        pts = []
        for i in ids:
            x, y, conf = kpts[i]
            if conf >= min_confidence:
                pts.append((float(x), float(y)))
        return pts

    shoulder_pts = get_pts(SHOULDER_IDS)
    hip_pts      = get_pts(HIP_IDS)

    if not shoulder_pts or not hip_pts:
        return None, image_pil, "Shoulders or hips not visible — try a clearer photo."

    all_pts    = shoulder_pts + hip_pts
    xs         = [p[0] for p in all_pts]
    ys         = [p[1] for p in all_pts]
    top_raw    = min(ys);  bottom_raw = max(ys)
    left_raw   = min(xs);  right_raw  = max(xs)
    torso_h    = bottom_raw - top_raw
    torso_w    = right_raw  - left_raw

    top    = int(max(0, top_raw    - torso_h * 0.08))
    bottom = int(min(h, bottom_raw + torso_h * 0.12))
    left   = int(max(0, left_raw   - torso_w * 0.15))
    right  = int(min(w, right_raw  + torso_w * 0.15))

    max_dim = max(bottom - top, right - left)
    cx = (left + right) // 2;  cy = (top + bottom) // 2
    half = max_dim // 2
    x1 = max(0, cx - half);  y1 = max(0, cy - half)
    x2 = min(w, cx + half);  y2 = min(h, cy + half)

    crop = img_bgr[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    crop_rgb     = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

    # Annotated preview
    debug = img_bgr.copy()
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 220, 120), 3)
    for pt in shoulder_pts + hip_pts:
        cv2.circle(debug, (int(pt[0]), int(pt[1])), 8, (0, 80, 255), -1)
    debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)

    return Image.fromarray(crop_rgb), Image.fromarray(debug_rgb), None


# ── Inference ─────────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def predict(image_pil: Image.Image):
    if _resnet is None:
        return None, None, "⚠️ Model weights not loaded. Run with --weights best_model.pth"

    cropped, annotated, err = crop_torso(image_pil)
    if err:
        return None, annotated, f"❌ {err}"

    tensor = _transform(cropped).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _resnet(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    pred_id    = int(torch.tensor(probs).argmax())
    pred_label = LABEL_MAP[pred_id]
    confidence = probs[pred_id] * 100

    # Build confidence breakdown string
    breakdown = "\n".join(
        f"{'→' if i == pred_id else '  '} {LABEL_MAP[i]:<8}  {p*100:5.1f}%"
        for i, p in enumerate(probs)
    )

    result = (
        f"Predicted body fat:  {pred_label}\n"
        f"Confidence:          {confidence:.1f}%\n\n"
        f"All scores:\n{breakdown}"
    )

    return cropped, annotated, result


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="Body Fat Estimator", theme=gr.themes.Base()) as demo:
        gr.Markdown("## Body Fat Estimator\nUpload a waist-up photo. The model crops the torso and predicts a body fat bucket.")

        with gr.Row():
            inp = gr.Image(type="pil", label="Input image")

        with gr.Row():
            cropped_out  = gr.Image(label="Cropped torso (model input)")
            annotated_out = gr.Image(label="Detected keypoints")

        result_out = gr.Textbox(label="Prediction", lines=12, max_lines=12)

        inp.change(fn=predict, inputs=inp, outputs=[cropped_out, annotated_out, result_out])

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="best_model.pth", help="Path to trained .pth weights")
    parser.add_argument("--port",    type=int, default=7860)
    args = parser.parse_args()

    load_resnet(args.weights)
    print(f"✅ Loaded weights from {args.weights}")
    print(f"🖥  Running on {_device.upper()}")

    app = build_ui()
    app.launch(server_port=args.port)