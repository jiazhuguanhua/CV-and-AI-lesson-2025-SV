import cv2
import numpy as np
import matplotlib.pyplot as plt

def line_segment_labeling(image_path, threshold_value=127, min_length=20):
    """
    Perform line segment labeling on a binary image.
    
    :param image_path: Path to the input image
    :param threshold_value: Threshold for binary conversion
    :param min_length: Minimum length for a line segment to be considered
    :return: Labeled image and line segment information
    """
    # Read and preprocess the image
    img = cv2.imread(image_path, 0)
    if img is None:
        raise ValueError("Image not found or unable to read.")
    
    # Convert to binary image (white text on black background)
    _, binary = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Invert if needed (ensure foreground is white)
    if np.sum(binary == 0) > np.sum(binary == 255):
        binary = cv2.bitwise_not(binary)
    
    return binary

def detect_horizontal_lines(binary_image, min_length=20):
    """
    Detect horizontal line segments using morphological operations.
    """
    # Create horizontal kernel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
    
    # Apply morphological operations to detect horizontal lines
    horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Label connected components using OpenCV
    num_labels, labeled_horizontal = cv2.connectedComponents(horizontal_lines)
    
    return horizontal_lines, labeled_horizontal, num_labels

def detect_vertical_lines(binary_image, min_length=20):
    """
    Detect vertical line segments using morphological operations.
    """
    # Create vertical kernel
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))
    
    # Apply morphological operations to detect vertical lines
    vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel)
    
    # Label connected components using OpenCV
    num_labels, labeled_vertical = cv2.connectedComponents(vertical_lines)
    
    return vertical_lines, labeled_vertical, num_labels

def detect_diagonal_lines(binary_image, min_length=20):
    """
    Detect diagonal line segments using Hough transform.
    """
    # Apply edge detection
    edges = cv2.Canny(binary_image, 50, 150)
    
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                           minLineLength=min_length, maxLineGap=5)
    
    # Create an image to draw detected lines
    line_image = np.zeros_like(binary_image)
    labeled_lines = np.zeros_like(binary_image, dtype=np.int32)
    
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            # Draw line
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
            cv2.line(labeled_lines, (x1, y1), (x2, y2), i+1, 2)
    
    return line_image, labeled_lines, lines

def analyze_line_segments(labeled_image):
    """
    Analyze properties of detected line segments using OpenCV.
    """
    line_info = []
    
    # Get unique labels (excluding background)
    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels > 0]
    
    for label_val in unique_labels:
        # Create mask for current label
        mask = (labeled_image == label_val).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Calculate moments
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroid = (cx, cy)
            else:
                centroid = (0, 0)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (y, x, y+h, x+w)  # (min_row, min_col, max_row, max_col)
            
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Calculate orientation (angle of the line)
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                orientation = ellipse[2]  # angle in degrees
            else:
                # For small contours, calculate angle from bounding box
                if w > h:
                    orientation = 0  # horizontal
                else:
                    orientation = 90  # vertical
            
            # Calculate length
            length = np.sqrt(w**2 + h**2)
            
            line_info.append({
                'label': int(label_val),
                'centroid': centroid,
                'bbox': bbox,
                'area': area,
                'orientation': orientation,
                'length': length
            })
    
    return line_info

def create_colored_labels(labeled_image):
    """
    Create a colored visualization of labeled line segments.
    """
    # Create a color map for different labels
    colored_image = np.zeros((*labeled_image.shape, 3), dtype=np.uint8)
    
    # Get unique labels (excluding background)
    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels > 0]
    
    # Assign colors to each label
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label_val in enumerate(unique_labels):
        mask = labeled_image == label_val
        colored_image[mask] = (colors[i][:3] * 255).astype(np.uint8)
    
    return colored_image

# Main execution
if __name__ == "__main__":
    input_path = "image_line_segment.jpg"  # Change this to your image path
    min_line_length = 15
    
    try:
        # Load and preprocess image
        binary_image = line_segment_labeling(input_path)
        
        # Detect different types of lines
        horizontal_lines, labeled_horizontal, num_h = detect_horizontal_lines(binary_image, min_line_length)
        vertical_lines, labeled_vertical, num_v = detect_vertical_lines(binary_image, min_line_length)
        diagonal_lines, labeled_diagonal, hough_lines = detect_diagonal_lines(binary_image, min_line_length)
        
        # Combine all line detections
        all_lines = horizontal_lines + vertical_lines + diagonal_lines
        all_lines = np.clip(all_lines, 0, 255).astype(np.uint8)
        
        # Create combined labeled image
        combined_labeled = np.zeros_like(binary_image, dtype=np.int32)
        offset = 0
        
        # Add horizontal lines with unique labels
        mask_h = labeled_horizontal > 0
        combined_labeled[mask_h] = labeled_horizontal[mask_h] + offset
        offset += np.max(labeled_horizontal) if np.max(labeled_horizontal) > 0 else 0
        
        # Add vertical lines with unique labels
        mask_v = labeled_vertical > 0
        combined_labeled[mask_v] = labeled_vertical[mask_v] + offset
        offset += np.max(labeled_vertical) if np.max(labeled_vertical) > 0 else 0
        
        # Add diagonal lines with unique labels
        mask_d = labeled_diagonal > 0
        combined_labeled[mask_d] = labeled_diagonal[mask_d] + offset
        
        # Analyze line segments
        line_info = analyze_line_segments(combined_labeled)
        
        # Create colored visualization
        colored_labels = create_colored_labels(combined_labeled)
        
        # Display results
        plt.figure(figsize=(20, 15))
        
        # Original image
        plt.subplot(3, 4, 1)
        original_img = cv2.imread(input_path, 0)
        plt.imshow(original_img, cmap='gray')
        plt.title('Original Image')
        plt.axis('on')
        
        # Binary image
        plt.subplot(3, 4, 2)
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binary Image')
        plt.axis('on')
        
        # Horizontal lines
        plt.subplot(3, 4, 3)
        plt.imshow(horizontal_lines, cmap='gray')
        plt.title('Horizontal Lines')
        plt.axis('on')
        
        # Vertical lines
        plt.subplot(3, 4, 4)
        plt.imshow(vertical_lines, cmap='gray')
        plt.title('Vertical Lines')
        plt.axis('on')
        
        # Diagonal lines
        plt.subplot(3, 4, 5)
        plt.imshow(diagonal_lines, cmap='gray')
        plt.title('Diagonal Lines (Hough)')
        plt.axis('on')
        
        # All lines combined
        plt.subplot(3, 4, 6)
        plt.imshow(all_lines, cmap='gray')
        plt.title('All Detected Lines')
        plt.axis('on')
        
        # Labeled horizontal lines
        plt.subplot(3, 4, 7)
        plt.imshow(labeled_horizontal, cmap='nipy_spectral')
        plt.title('Labeled Horizontal Lines')
        plt.axis('on')
        
        # Labeled vertical lines
        plt.subplot(3, 4, 8)
        plt.imshow(labeled_vertical, cmap='nipy_spectral')
        plt.title('Labeled Vertical Lines')
        plt.axis('on')
        
        # Combined labeled image
        plt.subplot(3, 4, 9)
        plt.imshow(combined_labeled, cmap='nipy_spectral')
        plt.title('All Labeled Line Segments')
        plt.axis('on')
        
        # Colored labels
        plt.subplot(3, 4, 10)
        plt.imshow(colored_labels)
        plt.title('Colored Line Segments')
        plt.axis('on')
        
        # Original with overlay
        plt.subplot(3, 4, 11)
        overlay = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB), 0.7, 
                                colored_labels, 0.3, 0)
        plt.imshow(overlay)
        plt.title('Original + Line Segments Overlay')
        plt.axis('on')
        
        # Statistics plot
        plt.subplot(3, 4, 12)
        if line_info:
            orientations = [info['orientation'] for info in line_info]
            lengths = [info['length'] for info in line_info]
            plt.scatter(orientations, lengths, alpha=0.7)
            plt.xlabel('Orientation (degrees)')
            plt.ylabel('Length (pixels)')
            plt.title('Line Segment Properties')
        else:
            plt.text(0.5, 0.5, 'No lines detected', ha='center', va='center')
            plt.title('Line Segment Properties')
        
        plt.tight_layout()
        plt.show()
        
        # Print line segment information
        print(f"\nDetected {len(line_info)} line segments:")
        print("-" * 80)
        for i, info in enumerate(line_info):
            print(f"Segment {i+1}:")
            print(f"  Label: {info['label']}")
            print(f"  Centroid: ({info['centroid'][0]:.1f}, {info['centroid'][1]:.1f})")
            print(f"  Length: {info['length']:.1f} pixels")
            print(f"  Orientation: {info['orientation']:.1f} degrees")
            print(f"  Area: {info['area']} pixels")
            print(f"  Bounding box: {info['bbox']}")
            print()
        
        # Summary statistics
        if line_info:
            horizontal_count = sum(1 for info in line_info if abs(info['orientation']) < 15 or abs(info['orientation']) > 165)
            vertical_count = sum(1 for info in line_info if 75 < abs(info['orientation']) < 105)
            diagonal_count = len(line_info) - horizontal_count - vertical_count
            
            print(f"Summary:")
            print(f"  Horizontal lines: {horizontal_count}")
            print(f"  Vertical lines: {vertical_count}")
            print(f"  Diagonal lines: {diagonal_count}")
            print(f"  Total line segments: {len(line_info)}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure the image file exists and is readable.")
