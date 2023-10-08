from PIL import Image
from predict import predict_image
import os
import numpy as np
import torchvision.transforms as transforms
from model import get_model
from dataset import to_device, get_device

# Open the input image
input_image_path = r"C:\Users\hryad\Desktop\SHIT 2.0\cropped_img01.jpg"
input_image = Image.open(input_image_path)

# Determine the dimensions of the input image
image_width, image_height = input_image.size

# Define the size of each sub-image (e.g., 64x64)
sub_image_size = (64, 64)

# Calculate the number of rows and columns for the grid
grid_rows = image_height // sub_image_size[1]
grid_columns = image_width // sub_image_size[0]

# Define a color map for each label
color_map = {
    'AnnualCrop': (255, 0, 0),  # Red
    'Forest': (0, 255, 0),  # Green
    'HerbaceousVegetation': (0, 0, 255),  # Blue
    'Highway': (255, 255, 0), # Yellow
    'Industrial': (255, 0, 255), # Magenta
    'Pasture': (0, 255, 255), # Cyan
    'PermanentCrop':(128, 0, 0), # Maroon
    'Residential': (0, 128, 0), # Green (dark)
    'River': (0, 0, 128), # Navy
    'SeaLake':  (128, 128, 128), # Gray
    
}

label_matrix = []

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
device = get_device()
model = get_model(load_weights=True).to(device)


# Loop through each sub-image
for row in range(grid_rows):
    label_row = []
    for col in range(grid_columns):
        left = col * sub_image_size[0]
        upper = row * sub_image_size[1]
        right = left + sub_image_size[0]
        lower = upper + sub_image_size[1]

        # Crop the sub-image from the original image
        sub_image = input_image.crop((left, upper, right, lower))

        # Predict label for the sub-image
        sub_image_path = os.path.join("temp_sub_image.png")
        sub_image.save(sub_image_path)
        label = predict_image(sub_image_path, model, transform, device)
        os.remove(sub_image_path)

        # Append the predicted label to the row list
        label_row.append(label)
    
    # Append the row list to the matrix
    label_matrix.append(label_row)

# Create an empty RGB image
output_image = Image.new('RGB', (image_width, image_height))

# Fill the image with colors based on the predicted labels
for row in range(grid_rows):
    for col in range(grid_columns):
        color = color_map[label_matrix[row][col]]
        for x in range(sub_image_size[0]):
            for y in range(sub_image_size[1]):
                output_image.putpixel((col*sub_image_size[0]+x, row*sub_image_size[1]+y), color)

# Save the final labeled image
output_image_path = "labeled_image.png"
output_image.save(output_image_path)

print(f'Labeled image saved at "{output_image_path}"')
