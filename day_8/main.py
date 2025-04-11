import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

image_path = os.path.join(os.path.dirname(__file__), 'sample.jpeg')

# Open an image file
img = Image.open(image_path).convert('L')
img = np.array(img)

print(img.shape)  # (height, width)

img_normalized = img / 255.0
mean = np.mean(img_normalized, axis=0)
img_centered = img_normalized - mean
covariance_matrix = np.cov(img_centered, rowvar=False)

# with SVD
U, S, Vt = np.linalg.svd(covariance_matrix, full_matrices=False)
explained_variance = np.cumsum(S) / np.sum(S)
k = np.argmax(explained_variance >= 0.95) + 1
print(f"Number of components to retain 95% variance: {k}")
V_k = Vt[:k, :]
img_projected = np.dot(img_centered, V_k.T)
# Step 8: Reconstruct the image
img_reconstructed = np.dot(img_projected, V_k) + mean
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.title("Original Grayscale Image")
# plt.imshow(img, cmap='gray')
# plt.axis('off')

# plt.subplot(1,2,2)
# plt.title(f"Compressed Image with {k} components")
# plt.imshow(img_reconstructed, cmap='gray')
# plt.axis('off')
# plt.show()

# save the reconstructed image
img_reconstructed = (img_reconstructed * 255).astype(np.uint8)
img_reconstructed_path = os.path.join(os.path.dirname(__file__), 'compressed_image.jpeg')
Image.fromarray(img_reconstructed).save(img_reconstructed_path)