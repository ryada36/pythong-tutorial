import numpy as np
from PIL import Image
import os

image_path = os.path.join(os.path.dirname(__file__), 'superman.jpeg')
destination_path = os.path.join(os.path.dirname(__file__), 'darkened_superman.jpeg')

# Load the image
image = Image.open(image_path)

# Convert the image to a NumPy array
image_array = np.array(image)

# darken the image
darkened_image_array = image_array * 0.5
darkened_image_array = darkened_image_array.astype(np.uint8)
# Convert the NumPy array back to an image
darkened_image = Image.fromarray(darkened_image_array)
# Save the darkened image
darkened_image.save(destination_path)
