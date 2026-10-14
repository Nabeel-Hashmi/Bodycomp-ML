# Body Composition Image Classifier

This repository contains an image classification prototype that predicts one of seven body fat percentage ranges from a torso photo. The demo first uses a YOLOv8 pose model to locate the shoulders and hips, crops and resizes the torso, then passes that image to a ResNet-18 classifier.

The output is a class range (`0–5%`, `5–10%`, `10–15%`, `15–20%`, `20–25%`, `25–30%`, or `30%+`) with scores for all classes. This is an experimental image classifier, not a medical measurement or a validated body composition assessment.

## Repository map

- `frontend/demo.py` — Gradio upload interface and inference pipeline. It uses both model files described below.
- `best_model.pth` — trained ResNet-18 classifier weights.
- `yolov8n-pose.pt` — YOLOv8 pose weights used to find torso landmarks.
- `model/BodyFatDataset.py` — CSV-backed PyTorch dataset, label mapping, and image transforms.
- `model/train.py` — ResNet-18 setup and validation/evaluation code. Its training loop is currently commented out; running this file loads `best_model.pth` and evaluates it.
- `data/processed/` — torso image files used by the labeled dataset and training pipeline.
- `labels.csv`, `bodyfat_labels_filled.csv`, `bodyfat_labels_filled_augmented.csv` — labeling and training CSVs. The training script reads the augmented CSV.
- `giving_class_id.py` — fills class IDs from label buckets in `labels.csv`.
- `webscraping/src/` — scripts for downloading source images, preparing image lists, and augmenting minority classes. These are optional data-preparation utilities.
- `Untitled.ipynb` — exploratory notebook.

## Set up and run the demo

The steps below use Windows PowerShell and assume the current directory is the repository root. Python 3.10 or newer is recommended.

1. Create and activate a virtual environment:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use `\.venv\Scripts\python.exe` in place of `python` in the commands below.

2. Install the Python packages used by the demo and model code:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install torch torchvision gradio ultralytics opencv-python numpy pillow pandas scikit-learn tqdm requests
   ```

   PyTorch can use CUDA when a compatible CUDA-enabled PyTorch build and GPU are installed; otherwise the demo runs on CPU. For a specific CUDA setup, use the install command recommended by the [official PyTorch installer](https://pytorch.org/get-started/locally/).

3. Confirm that `best_model.pth` and `yolov8n-pose.pt` are in the repository root. Both are currently included in this repository. The demo also accepts a different classifier checkpoint through `--weights`.

4. Start the demo:

   ```powershell
   python frontend/demo.py --weights best_model.pth
   ```

   Open `http://127.0.0.1:7860` in a browser. To use another port, for example 7861:

   ```powershell
   python frontend/demo.py --weights best_model.pth --port 7861
   ```

5. Upload a clear photo where one person's shoulders and hips are visible. The interface displays the detected crop, an image annotated with detected landmarks, the predicted range, and the model's class scores.

## Data and model workflow

The dataset expects a CSV with at least `filename` and `label_bucket` columns. `label_bucket` must be one of `0-5`, `5-10`, `10-15`, `15-20`, `20-25`, `25-30`, or `30+`. Each filename must identify an image in the image directory. `BodyFatDataset` derives numeric class IDs from `label_bucket`; it does not use the CSV's `class_id` column.

To create torso crops from source images, the crop processing code is in `webscraping/src/softwareengineering.py`. From the repository root, run:

```powershell
python webscraping/src/softwareengineering.py --input .\data\raw\ --output .\data\processed\
```

This script writes crops plus a `crop_log.json` into the output directory. The `--input` directory needs to exist and contain supported image files (`jpg`, `jpeg`, `png`, `bmp`, or `webp`). YOLO may need network access to download model weights when they are not already available locally.

The related scripts can help build labels or add horizontally mirrored examples:

```powershell
python giving_class_id.py
python webscraping/src/utils.py
```

`giving_class_id.py` reads `labels.csv` and writes `bodyfat_labels_filled.csv`. `utils.py` reads that filled CSV and writes `bodyfat_labels_filled_augmented.csv`, while also saving mirrored images into `data/processed/`. Review the labels and images before using generated data.

## Training and evaluation notes

`model/train.py` uses paths relative to the repository root (`bodyfat_labels_filled_augmented.csv`, `data/processed/`, and `best_model.pth`), so launch it from this directory. As currently written, its optimizer, training helpers, and training loop are commented out; executing it constructs the model and evaluates the checkpoint that already exists. To train a new model, the training loop must first be enabled and configured. Training also requires enough labeled images in each class for its stratified 80/20 split.

The dataset preparation scripts are prototypes. For example, image augmentation mirrors every image in a class below the target count once; it does not fill that class all the way to the target. The scraper targets Reddit and should only be used where collection and use of source images are permitted.

## Common issues

- **Missing checkpoint:** Run from the repository root or pass the correct path to `--weights`.
- **Pose or torso not detected:** Upload a well-lit image with the person facing the camera and shoulders and hips visible.
- **Missing image during evaluation/training:** Check that every CSV `filename` exists under `data/processed/`.
- **Class mapping error:** Check spelling and formatting of `label_bucket`; the dataset expects the seven values listed above.
- **PowerShell cannot activate the environment:** Invoke `.\.venv\Scripts\python.exe` directly.
