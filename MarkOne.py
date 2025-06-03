import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ======================= CONFIG ========================
BATCH_SIZE = 32
NUM_EPOCHS = 14
LEARNING_RATE = 1e-3
NUM_BEAMS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV = "Dataset-main/scenario5_dev_train.csv"
VAL_CSV = "Dataset-main/scenario5_dev_val.csv"
TEST_CSV = "Dataset-main/scenario5_dev_test.csv"
BASE_DIR = "Dataset-main"

# ======================= DATASET ========================

class BeamPredictionDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.base_dir = os.path.dirname(csv_path)

    def __getitem__(self, idx):
        image_rel_path = self.data.iloc[idx]['unit1_rgb_1']
        label = int(self.data.iloc[idx]['beam_index_1'])

        # Prepend the base dir
        image_path = os.path.join(BASE_DIR, image_rel_path.strip("./"))

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data)


def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_set = BeamPredictionDataset(TRAIN_CSV, transform)
    val_set = BeamPredictionDataset(VAL_CSV, transform)
    test_set = BeamPredictionDataset(TEST_CSV, transform)

    return {
        'train': DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(val_set, batch_size=BATCH_SIZE),
        'test': DataLoader(test_set, batch_size=BATCH_SIZE)
    }

# ======================= MODEL ========================
class BeamPredictor(nn.Module):
    def __init__(self, num_beams=64):
        super(BeamPredictor, self).__init__()
        base_model = models.resnet18(pretrained=True)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_beams)
        self.model = base_model

    def forward(self, x):
        return self.model(x)

# ======================= METRICS ========================
def compute_topk_accuracy(outputs, labels, k=1):
    _, preds = outputs.topk(k, dim=1)
    return (preds == labels.unsqueeze(1)).sum().item()

# ======================= TRAINING ========================
def train_model(model, dataloaders):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(dataloaders['train']):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_acc = evaluate(model, dataloaders['val'])
        print(f"Validation Top-1 Accuracy: {val_acc:.2f}%, Loss: {running_loss:.4f}")

# ======================= VALIDATION ========================
def evaluate(model, dataloader, k=1):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.topk(k, dim=1)
            correct += (preds == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total

# ======================= TESTING ========================
def test_model(model, dataloader):
    model.eval()
    top1, top2, top3, total = 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            top1 += compute_topk_accuracy(outputs, labels, k=1)
            top2 += compute_topk_accuracy(outputs, labels, k=2)
            top3 += compute_topk_accuracy(outputs, labels, k=3)
            total += labels.size(0)

    print("\n===== TEST RESULTS =====")
    print(f"Top-1 Accuracy: {top1 / total:.2%}")
    print(f"Top-2 Accuracy: {top2 / total:.2%}")
    print(f"Top-3 Accuracy: {top3 / total:.2%}")

# ======================= MAIN ========================
def main():
    dataloaders = get_dataloaders()
    model = BeamPredictor(num_beams=NUM_BEAMS)

    train_model(model, dataloaders)
    test_model(model, dataloaders['test'])

if __name__ == "__main__":
    main()
