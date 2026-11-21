import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CLASSES, MODEL_DIR, RESULT_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

class WasteDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        image = Image.open(img_path).convert("RGB")
        
        label_name = row['label']
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label

# Same transformation used during validation/testing
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_trained_model(model_path, num_classes=len(CLASSES)):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def evaluate():
    test_csv = os.path.join(RESULT_DIR, "test.csv")
    model_path = os.path.join(MODEL_DIR, "best_transfer_cnn.pth")

    if not os.path.exists(test_csv):
        print(f"Error: Could not find test split at {test_csv}")
        return
    if not os.path.exists(model_path):
        print(f"Error: Could not find trained model weights at {model_path}")
        return

    test_dataset = WasteDataset(test_csv, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_trained_model(model_path)

    all_preds = []
    all_labels = []

    print(f"Evaluating ResNet-18 on the test set using {DEVICE}...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute final metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Macro F1:  {macro_f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()