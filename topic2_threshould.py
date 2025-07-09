import cv2
import matplotlib.pyplot as plt
import numpy as np

def threshold_image(image_path, threshold_value=127):
    """
    Reads an image from the specified path, converts it to grayscale,
    and applies a binary threshold to it.

    :param image_path: Path to the input image.
    :param threshold_value: Threshold value for binary thresholding.
    :return: The thresholded binary image.
    """
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Image not found or unable to read.")

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    return binary_image

def calculate_center_of_mass(binary_image):
    """
    Calculate the center of mass (centroid) of a binary image.
    
    :param binary_image: Binary image (white pixels are considered as mass)
    :return: Tuple (x, y) representing the center of mass coordinates
    """
    # Find all white pixels (value 255)
    white_pixels = np.where(binary_image == 255)
    
    if len(white_pixels[0]) == 0:
        return None  # No white pixels found
    
    # Calculate center of mass
    y_coords = white_pixels[0]  # Row indices
    x_coords = white_pixels[1]  # Column indices
    
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    
    return (center_x, center_y)

# Example usage
if __name__ == "__main__":
    input_path = "image_line_segment.jpg"  # Path to the input image
    threshold_value = 50

    binary_image = threshold_image(input_path, threshold_value)

    # Calculate center of mass
    center_of_mass = calculate_center_of_mass(binary_image)
    
    if center_of_mass:
        print(f"Center of mass: ({center_of_mass[0]:.2f}, {center_of_mass[1]:.2f})")
    else:
        print("No white pixels found in the binary image")

    # Output F(g): show the coordinate plot instead of array
    print("F(g): Displaying binary image coordinate plot")
    
    # Create a figure with subplots
    plt.figure(figsize=(15, 5))
    
    # Show original grayscale image
    plt.subplot(1, 3, 1)
    original_image = cv2.imread(input_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Grayscale Image (g)')
    plt.axis('on')
    
    # Show binary thresholded image
    plt.subplot(1, 3, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title(f'Binary Thresholded Image F(g) (threshold={threshold_value})')
    plt.axis('on')
    
    # Show binary image with center of mass marked
    plt.subplot(1, 3, 3)
    plt.imshow(binary_image, cmap='gray')
    if center_of_mass:
        plt.plot(center_of_mass[0], center_of_mass[1], 'r+', markersize=15, markeredgewidth=3, label='Center of Mass')
        plt.plot(center_of_mass[0], center_of_mass[1], 'ro', markersize=8, fillstyle='none', markeredgewidth=2)
        plt.legend()
    plt.title('Binary Image with Center of Mass')
    plt.axis('on')
    
    plt.tight_layout()
    plt.show()

    # Also show the Black/White image using OpenCV
    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()