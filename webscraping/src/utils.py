from PIL import Image, ImageOps
import pandas as pd
import os


CSV_PATH   = "./bodyfat_labels_filled.csv"
IMG_DIR    = "data/processed"



def augment_minority_classes(csv_path, img_dir, target_count=42):
    df = pd.read_csv(csv_path)
    new_rows = []

    for bucket, group in df.groupby("label_bucket"):
        if len(group) >= target_count:
            continue  # skip majority classes

        for _, row in group.iterrows():
            img_path = os.path.join(img_dir, row["filename"])
            img = Image.open(img_path).convert("RGB")
            flipped = ImageOps.mirror(img)

            new_filename = f"aug_{row['filename']}"
            flipped.save(os.path.join(img_dir, new_filename))
            new_rows.append({**row, "filename": new_filename})

    augmented_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    augmented_df.to_csv(csv_path.replace(".csv", "_augmented.csv"), index=False)


augment_minority_classes(CSV_PATH,IMG_DIR)