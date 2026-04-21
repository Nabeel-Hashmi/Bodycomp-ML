import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.model_selection import train_test_split

from BodyFatDataset import BodyFatDataset, get_train_transform, get_val_transform


# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH   = "./bodyfat_labels_filled_augmented.csv"
IMG_DIR    = "data/processed"
NUM_CLASSES = 7
BATCH_SIZE  = 32
EPOCHS      = 25
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH   = "best_model.pth"


# ── Dataset split ─────────────────────────────────────────────────────────────
full_dataset = BodyFatDataset(CSV_PATH, IMG_DIR)

indices = list(range(len(full_dataset)))
labels  = full_dataset.df["class_id"].tolist()

train_idx, val_idx = train_test_split(
    indices, test_size=0.2, stratify=labels, random_state=42
)

# Apply different transforms to each split
train_dataset = Subset(full_dataset, train_idx)
val_dataset   = Subset(full_dataset, val_idx)

# Override transforms per split
train_dataset.dataset.transform = get_train_transform()
# val keeps the default val transform; swap it safely with a shallow copy:
import copy
val_base = copy.copy(full_dataset)
val_base.transform = get_val_transform()
val_dataset = Subset(val_base, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    """ResNet-18 with the final FC replaced for our classification head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

model = build_model(NUM_CLASSES).to(DEVICE)


# ── Loss / Optimizer / Scheduler ──────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ── Training helpers ──────────────────────────────────────────────────────────
def run_epoch(loader, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


# ── Training loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.538

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss,   val_acc   = run_epoch(val_loader,   training=False)
    scheduler.step()

    print(
        f"Epoch {epoch:>3}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f} | "
        f"Val loss: {val_loss:.4f}  acc: {val_acc:.3f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")

print(f"\nTraining complete. Best val acc: {best_val_acc:.3f}")

# ── Final evaluation ──────────────────────────────────────────────────────────
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

# Map class IDs back to human-readable bucket names
id_to_label = {v: k for k, v in full_dataset.label_map.items()}

print("\nSample predictions (first 40):")
print(f"{'Actual':<10} {'Predicted':<10} {'Correct'}")
for true, pred in zip(all_labels[:], all_preds[:]):
    match = "✓" if true == pred else "✗"
    print(f"{id_to_label[true]:<10} {id_to_label[pred]:<10} {match}")