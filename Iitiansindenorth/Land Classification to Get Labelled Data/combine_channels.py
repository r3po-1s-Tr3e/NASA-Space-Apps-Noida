import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Define the file paths for the separate bands (e.g., B2, B3, B4)
band2_path = 'blue3.jp2'
band3_path = 'green3.jp2'
band4_path = 'red3.jp2'

# Open the separate band files
with rasterio.open(band2_path) as b2, rasterio.open(band3_path) as b3, rasterio.open(band4_path) as b4:
    # Read the band data into numpy arrays
    band2 = b2.read(1)
    band3 = b3.read(1)
    band4 = b4.read(1)

# Create a false color composite (e.g., NIR, Red, Green for a natural color image)
composite = np.stack([band4, band3, band2], axis=0)

# Normalize the pixel values (optional)
composite = composite / 10000.0  # Sentinel-2 pixel values are usually in the range 0-10000

# Center crop the image to 1024x1024
crop_size = 1024
height, width = composite.shape[1], composite.shape[2]
left = (width - crop_size) // 2
top = (height - crop_size) // 2
right = (width + crop_size) // 2
bottom = (height + crop_size) // 2
composite_cropped = composite[:, top:bottom, left:right]

# Save or display the cropped composite image
plt.imshow(np.moveaxis(composite_cropped, 0, -1))
plt.axis('off')  # Optional: Hide axes
plt.show()
plt.savefig('composite_image03.png', dpi=300, bbox_inches='tight')  # Save as PNG

print("Cropped composite image saved as 'composite_image05.png'")
