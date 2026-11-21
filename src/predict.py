import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CLASSES, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (224, 224)

def load_trained_model():
    model_path = os.path.join(MODEL_DIR, "best_transfer_cnn.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model weights not found at {model_path}. Train the model first!")
    
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, len(CLASSES))
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Standard evaluation preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        return

    model = load_trained_model()
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = CLASSES[predicted_idx.item()]
    confidence_score = confidence.item() * 100
    
    print("\n========== BINGO PREDICTION RESULT ==========")
    print(f"Image: {image_path}")
    print(f"Predicted Waste Type: **{predicted_class.upper()}**")
    print(f"Confidence: {confidence_score:.2f}%\n")
    
    print("Class Probabilities:")
    for i, cls_name in enumerate(CLASSES):
        print(f"  - {cls_name:10s}: {probabilities[i].item() * 100:6.2f}%")
    print("=============================================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image>")
        print("Example: python src/predict.py raw/dataset-resized/plastic/plastic1.jpg")
    else:
        predict_image(sys.argv[1])