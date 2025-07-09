import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load binary image (text is black, background is white)
img = cv2.imread('image_ukcar.jpg', 0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# Sum vertically (axis=0 means sum along rows, resulting in horizontal projection)
vertical_proj = np.sum(binary, axis=0)

# Sum horizontally (axis=1 means sum along columns, resulting in vertical projection)
horizontal_proj = np.sum(binary, axis=1)

# Create projection image by mapping projection values to grayscale
proj_height = 100  # Height of the projection visualization
proj_width = 100   # Width of the horizontal projection visualization
proj_img = np.zeros((proj_height, len(vertical_proj)), dtype=np.uint8)
proj_img_horz = np.zeros((len(horizontal_proj), proj_width), dtype=np.uint8)

# Normalize projection values to 0-255 range
if vertical_proj.max() > 0:
    normalized_proj = (vertical_proj / vertical_proj.max() * 255).astype(np.uint8)
else:
    normalized_proj = np.zeros_like(vertical_proj, dtype=np.uint8)

# Normalize horizontal projection values to 0-255 range
if horizontal_proj.max() > 0:
    normalized_proj_horz = (horizontal_proj / horizontal_proj.max() * 255).astype(np.uint8)
else:
    normalized_proj_horz = np.zeros_like(horizontal_proj, dtype=np.uint8)

# Fill the projection image - each column represents the projection value
for i in range(len(vertical_proj)):
    # Fill from bottom to top based on projection value
    fill_height = int((vertical_proj[i] / vertical_proj.max()) * proj_height) if vertical_proj.max() > 0 else 0
    proj_img[-fill_height:, i] = normalized_proj[i]

# Fill the horizontal projection image - each row represents the projection value
for i in range(len(horizontal_proj)):
    # Fill from left to right based on projection value
    fill_width = int((horizontal_proj[i] / horizontal_proj.max()) * proj_width) if horizontal_proj.max() > 0 else 0
    proj_img_horz[i, :fill_width] = normalized_proj_horz[i]

# Alternative: Create a simpler projection image where each column is filled with the projection value
proj_img_simple = np.tile(normalized_proj, (proj_height, 1))
proj_img_horz_simple = np.tile(normalized_proj_horz.reshape(-1, 1), (1, proj_width))

# Display results
plt.figure(figsize=(18, 12))

# Original binary image
plt.subplot(3, 4, 1)
plt.imshow(binary, cmap='gray')
plt.title('Original Binary Image')
plt.axis('on')

# Vertical projection plot
plt.subplot(3, 4, 2)
plt.plot(vertical_proj)
plt.title('Vertical Projection Values')
plt.xlabel('Column Index')
plt.ylabel('Projection Value')

# Horizontal projection plot
plt.subplot(3, 4, 3)
plt.plot(horizontal_proj)
plt.title('Horizontal Projection Values')
plt.xlabel('Projection Value')
plt.ylabel('Row Index')

# Projection as bar chart
plt.subplot(3, 4, 4)
plt.bar(range(len(vertical_proj)), vertical_proj, width=1)
plt.title('Vertical Projection Bar Chart')
plt.xlabel('Column Index')
plt.ylabel('Projection Value')

# Vertical projection image (bar-like visualization)
plt.subplot(3, 4, 5)
plt.imshow(proj_img, cmap='gray')
plt.title('Vertical Projection as Gray Image (Bar Style)')
plt.xlabel('Column Index')
plt.ylabel('Projection Height')

# Horizontal projection image (bar-like visualization)
plt.subplot(3, 4, 6)
plt.imshow(proj_img_horz, cmap='gray')
plt.title('Horizontal Projection as Gray Image (Bar Style)')
plt.xlabel('Projection Width')
plt.ylabel('Row Index')

# Vertical projection image (uniform fill)
plt.subplot(3, 4, 7)
plt.imshow(proj_img_simple, cmap='gray')
plt.title('Vertical Projection as Gray Image (Uniform Fill)')
plt.xlabel('Column Index')

# Horizontal projection image (uniform fill)
plt.subplot(3, 4, 8)
plt.imshow(proj_img_horz_simple, cmap='gray')
plt.title('Horizontal Projection as Gray Image (Uniform Fill)')
plt.xlabel('Projection Width')

# Combined view - vertical
plt.subplot(3, 4, 9)
# Stack original image and projection
combined = np.vstack([binary, proj_img])
plt.imshow(combined, cmap='gray')
plt.title('Original + Vertical Projection')
plt.xlabel('Column Index')

# Combined view - horizontal
plt.subplot(3, 4, 10)
# Stack original image and horizontal projection
combined_horz = np.hstack([binary, proj_img_horz])
plt.imshow(combined_horz, cmap='gray')
plt.title('Original + Horizontal Projection')
plt.xlabel('Column Index')

# Show both projections as plots
plt.subplot(3, 4, 11)
plt.plot(vertical_proj, label='Vertical Projection')
plt.xlabel('Column Index')
plt.ylabel('Projection Value')
plt.title('Vertical Projection')
plt.legend()

plt.subplot(3, 4, 12)
plt.barh(range(len(horizontal_proj)), horizontal_proj, height=1)
plt.title('Horizontal Projection Bar Chart')
plt.xlabel('Projection Value')
plt.ylabel('Row Index')

plt.tight_layout()
plt.show()

# Find columns with zero sum (gaps)
gap_columns = np.where(vertical_proj == 0)[0]
gap_rows = np.where(horizontal_proj == 0)[0]
print("Potential character gaps at columns:", gap_columns)
print("Potential line gaps at rows:", gap_rows)
print(f"Vertical projection shape: {proj_img.shape}")
print(f"Horizontal projection shape: {proj_img_horz.shape}")
print(f"Vertical projection value range: {vertical_proj.min()} - {vertical_proj.max()}")
print(f"Horizontal projection value range: {horizontal_proj.min()} - {horizontal_proj.max()}")