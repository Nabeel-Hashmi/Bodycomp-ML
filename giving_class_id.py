import csv

# Map label bucket -> class id
LABEL_TO_CLASS = {
    "0-5": 0,
    "5-10": 1,
    "10-15": 2,
    "15-20": 3,
    "20-25": 4,
    "25-30": 5,
    "30+": 6,
}

input_file = "labels.csv"
output_file = "bodyfat_labels_filled.csv"

with open(input_file, "r", newline="", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)

    # Clean field names in case there are extra spaces
    fieldnames = [name.strip() for name in reader.fieldnames]

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        # Normalize keys/values
        clean_row = {k.strip(): (v.strip() if v is not None else "") for k, v in row.items()}

        label_bucket = clean_row.get("label_bucket", "")
        if label_bucket in LABEL_TO_CLASS:
            clean_row["class_id"] = str(LABEL_TO_CLASS[label_bucket])

        writer.writerow(clean_row)

print(f"Done. Filled class_id values and saved to: {output_file}")