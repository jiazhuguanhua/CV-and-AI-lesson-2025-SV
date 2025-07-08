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

# Example usage
if __name__ == "__main__":
    input_path = "image.jpg"
    threshold_value = 50

    binary_image = threshold_image(input_path, threshold_value)

    # Output F(g): show the coordinate plot instead of array
    print("F(g): Displaying binary image coordinate plot")
    
    # Create a figure with subplots
    plt.figure(figsize=(12, 5))
    
    # Show original grayscale image
    plt.subplot(1, 2, 1)
    original_image = cv2.imread(input_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Grayscale Image (g)')
    plt.axis('on')
    
    # Show binary thresholded image
    plt.subplot(1, 2, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title(f'Binary Thresholded Image F(g) (threshold={threshold_value})')
    plt.axis('on')
    
    plt.tight_layout()
    plt.show()

    # Also show the Black/White image using OpenCV
    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()