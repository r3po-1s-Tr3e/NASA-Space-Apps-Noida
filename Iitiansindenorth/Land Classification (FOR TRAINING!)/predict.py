import os
import matplotlib.pyplot as plt

from dataset import to_device, get_device, get_image
from model import get_model
import torch
import config

def predict_image_label(image_path):
    image = get_image(image_path)
    device = get_device()
    model = get_model(load_weights=True)
    model.eval()
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    preds = model(xb)
    _, prediction = torch.max(preds.cpu().detach(), dim=1)
    print(f"Image Name: {image_path} - Predicted Class: {config.IDX_CLASS_LABELS[int(prediction)]}")
    return int(prediction)

if __name__ == "__main__":
    # Collect all image paths in config.DATA_TESTING_PATH
    all_image_paths = []
    for root, dirs, files in os.walk(config.DATA_TESTING_PATH):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):  # You can add more extensions if needed
                all_image_paths.append(os.path.join(root, file))
    
    # Predict labels for all images
    all_predictions = [predict_image_label(img_path) for img_path in all_image_paths]
    
    # Plot histogram
    plt.hist(all_predictions, bins=len(config.IDX_CLASS_LABELS), align='left', rwidth=0.7)
    plt.xticks(list(config.IDX_CLASS_LABELS.keys()), list(config.IDX_CLASS_LABELS.values()), rotation=45)
    plt.ylabel('Number of Images')
    plt.xlabel('Class Labels')
    plt.title('Distribution of Predicted Classes')
    plt.tight_layout()
    plt.show()
