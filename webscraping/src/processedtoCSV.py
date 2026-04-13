from pathlib import Path
import csv

image_dir = Path(r"C:\dev\Bodycomp-ML\data\processed")
output_csv = Path("labels_for_excel.csv")

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
images = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename_link", "filename", "filepath", "label_bucket", "class_id"])

    for img in images:
        abs_path = str(img.resolve())
        hyperlink_formula = f'=HYPERLINK("{abs_path}","{img.name}")'
        writer.writerow([hyperlink_formula, img.name, abs_path, "", "", ""])

print(f"Created: {output_csv}")