"""
crop_torso.py
-------------
Auto-crops waist-up male photos to torso only (chest → hips)
for use as CNN input in body fat estimation models.

Uses YOLOv8 Pose to detect landmarks and crops between
shoulders and hips with configurable padding.

Usage:
    python crop_torso.py --input ./data/raw/ --output ./data/processed/

Requirements:
    pip install ultralytics opencv-python Pillow tqdm
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import json
from tqdm import tqdm
from ultralytics import YOLO

# ── YOLOv8 COCO keypoint indices ────────────────────────────────────────────
# Shoulders: 5 (left), 6 (right)
# Hips:      11 (left), 12 (right)
SHOULDER_IDS = [5, 6]
HIP_IDS      = [11, 12]

# Load model once (auto-downloads yolov8n-pose.pt on first run ~6MB)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n-pose.pt")
    return _model


def crop_torso(
    image_bgr: np.ndarray,
    padding_top: float = 0.08,
    padding_bottom: float = 0.12,
    padding_sides: float = 0.15,
    output_size: int = 224,
    min_confidence: float = 0.5,
) -> tuple[np.ndarray | None, dict]:
    """
    Detect pose landmarks and return a square-cropped torso image.

    Returns
    -------
    cropped : np.ndarray (RGB, output_size x output_size) or None if failed
    info    : dict with crop coordinates and keypoint data
    """
    h, w = image_bgr.shape[:2]

    results = get_model()(image_bgr, verbose=False)

    if not results or results[0].keypoints is None or len(results[0].keypoints.data) == 0:
        return None, {"error": "no_pose_detected"}

    # Use the first (most confident) person
    kpts = results[0].keypoints.data[0].cpu().numpy()  # (17, 3): x, y, conf

    # ── Collect shoulder & hip coordinates ──────────────────────────────────
    def get_pts(ids):
        pts = []
        for i in ids:
            x, y, conf = kpts[i]
            if conf >= min_confidence:
                pts.append((float(x), float(y)))
        return pts

    shoulder_pts = get_pts(SHOULDER_IDS)
    hip_pts      = get_pts(HIP_IDS)

    if len(shoulder_pts) == 0 or len(hip_pts) == 0:
        return None, {"error": "landmarks_not_visible"}

    # ── Bounding box from landmarks ──────────────────────────────────────────
    all_pts    = shoulder_pts + hip_pts
    xs         = [p[0] for p in all_pts]
    ys         = [p[1] for p in all_pts]

    top_raw    = min(ys)
    bottom_raw = max(ys)
    left_raw   = min(xs)
    right_raw  = max(xs)

    torso_h    = bottom_raw - top_raw
    torso_w    = right_raw  - left_raw

    # ── Apply padding ────────────────────────────────────────────────────────
    top    = int(max(0, top_raw    - torso_h * padding_top))
    bottom = int(min(h, bottom_raw + torso_h * padding_bottom))
    left   = int(max(0, left_raw   - torso_w * padding_sides))
    right  = int(min(w, right_raw  + torso_w * padding_sides))

    # ── Make square (centered on crop center) ────────────────────────────────
    max_dim = max(bottom - top, right - left)
    cx      = (left + right) // 2
    cy      = (top + bottom) // 2
    half    = max_dim // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    crop = image_bgr[y1:y2, x1:x2]

    # ── Resize to CNN input size ─────────────────────────────────────────────
    crop_resized = cv2.resize(crop, (output_size, output_size),
                               interpolation=cv2.INTER_LANCZOS4)
    crop_rgb     = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

    info = {
        "bbox": [x1, y1, x2, y2],
        "torso_height_px": int(torso_h),
        "shoulder_pts": shoulder_pts,
        "hip_pts": hip_pts,
    }

    return crop_rgb, info


def process_folder(
    input_dir: str,
    output_dir: str,
    output_size: int = 224,
    padding_top: float = 0.08,
    padding_bottom: float = 0.12,
    padding_sides: float = 0.15,
    save_debug: bool = False,
) -> dict:
    """Batch-process all images in input_dir and save cropped torsos to output_dir."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in in_path.iterdir() if p.is_file() and p.suffix.lower() in exts]

    log = {"processed": [], "failed": []}

    for img_path in tqdm(images, desc="Cropping torsos"):
        try:
            if not img_path.exists():
                log["failed"].append({"file": img_path.name, "reason": "missing_file"})
                continue

            if img_path.stat().st_size == 0:
                log["failed"].append({"file": img_path.name, "reason": "empty_file"})
                continue

            img_bgr = cv2.imread(str(img_path))

            if img_bgr is None:
                log["failed"].append({"file": img_path.name, "reason": "unreadable_or_corrupt"})
                continue

        except Exception as e:
            log["failed"].append({"file": img_path.name, "reason": f"read_error: {str(e)}"})
            continue

        crop, info = crop_torso(
            img_bgr,
            padding_top=padding_top,
            padding_bottom=padding_bottom,
            padding_sides=padding_sides,
            output_size=output_size,
        )

        if crop is None:
            log["failed"].append({"file": img_path.name, "reason": info.get("error")})
            continue

        Image.fromarray(crop).save(out_path / img_path.name)
        log["processed"].append({"file": img_path.name, "bbox": info["bbox"]})

        if save_debug:
            debug_dir = out_path / "debug"
            debug_dir.mkdir(exist_ok=True)
            debug = img_bgr.copy()
            x1, y1, x2, y2 = info["bbox"]
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
            for pt in info["shoulder_pts"] + info["hip_pts"]:
                cv2.circle(debug, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
            cv2.imwrite(str(debug_dir / img_path.name), debug)

    log_path = out_path / "crop_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    total = len(images)
    ok = len(log["processed"])
    print(f"\n✅ {ok}/{total} images cropped successfully → {out_path}")
    if log["failed"]:
        print(f"⚠️ {len(log['failed'])} failed — see {log_path}")

    return log


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop torso region from waist-up male photos")
    parser.add_argument("--input",      default="./data/raw/",       help="Folder of input images")
    parser.add_argument("--output",     default="./data/processed/", help="Folder for cropped outputs")
    parser.add_argument("--size",       type=int,   default=224,  help="Output square size (default 224)")
    parser.add_argument("--pad-top",    type=float, default=0.08)
    parser.add_argument("--pad-bottom", type=float, default=0.12)
    parser.add_argument("--pad-sides",  type=float, default=0.15)
    parser.add_argument("--debug",      action="store_true", help="Save landmark overlay images")
    args = parser.parse_args()

    process_folder(
        input_dir=args.input,
        output_dir=args.output,
        output_size=args.size,
        padding_top=args.pad_top,
        padding_bottom=args.pad_bottom,
        padding_sides=args.pad_sides,
        save_debug=args.debug,
    )