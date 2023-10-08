import torch
import config
import torchvision.transforms as transforms
from PIL import Image
from model import get_model
from dataset import to_device, get_device

def predict_image(image_path, model, transform, device):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = to_device(image, device)

    
    with torch.no_grad():
        outputs = model(image)
        _, prediction = torch.max(outputs, 1)
        prediction = prediction.item()

    return config.IDX_CLASS_LABELS[prediction]

if __name__ == "__main__":
    print("Initializing...")

    # Transformation for the input image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Setting up the device and loading the trained model
    device = get_device()
    model = get_model(load_weights=True).to(device)
    model.eval()

    # Predicting label for the given image
    image_path = config.PATH
    predicted_label = predict_image(image_path, model, transform, device)
    print(f"Predicted Label for {image_path}: {predicted_label}")
