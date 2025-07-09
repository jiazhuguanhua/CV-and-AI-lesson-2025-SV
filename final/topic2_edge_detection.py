import cv2
import numpy as np
import matplotlib.pyplot as plt

def roberts_edge_detection(image):
    """
    Roberts边缘检测算子
    使用Roberts交叉算子进行边缘检测
    """
    # Roberts算子核
    roberts_x = np.array([[1, 0],
                         [0, -1]], dtype=np.float32)
    
    roberts_y = np.array([[0, 1],
                         [-1, 0]], dtype=np.float32)
    
    # 应用Roberts算子
    edges_x = cv2.filter2D(image, cv2.CV_32F, roberts_x)
    edges_y = cv2.filter2D(image, cv2.CV_32F, roberts_y)
    
    # 计算梯度幅值
    edges = np.sqrt(edges_x**2 + edges_y**2)
    
    # 归一化到0-255
    edges = np.uint8(np.clip(edges, 0, 255))
    
    return edges

def prewitt_edge_detection(image):
    """
    Prewitt边缘检测算子
    使用Prewitt算子进行边缘检测
    """
    # Prewitt算子核
    prewitt_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)
    
    prewitt_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]], dtype=np.float32)
    
    # 应用Prewitt算子
    edges_x = cv2.filter2D(image, cv2.CV_32F, prewitt_x)
    edges_y = cv2.filter2D(image, cv2.CV_32F, prewitt_y)
    
    # 计算梯度幅值
    edges = np.sqrt(edges_x**2 + edges_y**2)
    
    # 归一化到0-255
    edges = np.uint8(np.clip(edges, 0, 255))
    
    return edges

def sobel_edge_detection(image):
    """
    Sobel边缘检测算子（作为对比参考）
    """
    # 使用OpenCV内置的Sobel算子
    edges_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    edges_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    edges = np.sqrt(edges_x**2 + edges_y**2)
    
    # 归一化到0-255
    edges = np.uint8(np.clip(edges, 0, 255))
    
    return edges

def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Canny边缘检测
    """
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

def laplacian_edge_detection(image):
    """
    Laplacian边缘检测算子（额外的对比）
    """
    edges = cv2.Laplacian(image, cv2.CV_64F)
    edges = np.uint8(np.absolute(edges))
    return edges

def compare_edge_detectors(image_path):
    """
    比较不同的边缘检测算法
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    print("正在应用不同的边缘检测算法...")
    
    # 应用不同的边缘检测算法
    roberts_edges = roberts_edge_detection(blurred)
    prewitt_edges = prewitt_edge_detection(blurred)
    sobel_edges = sobel_edge_detection(blurred)
    canny_edges = canny_edge_detection(blurred)
    laplacian_edges = laplacian_edge_detection(blurred)
    
    # 创建多个Canny阈值的结果
    canny_low = canny_edge_detection(blurred, 30, 80)
    canny_medium = canny_edge_detection(blurred, 50, 150)
    canny_high = canny_edge_detection(blurred, 100, 200)
    
    # 显示结果
    plt.figure(figsize=(20, 15))
    
    # 第一行：原图和预处理
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(blurred, cmap='gray')
    plt.title('Gaussian Blurred')
    plt.axis('off')
    
    # 第二行：主要边缘检测算法
    plt.subplot(3, 4, 4)
    plt.imshow(roberts_edges, cmap='gray')
    plt.title('Roberts Edge Detection')
    plt.axis('off')
    
    plt.subplot(3, 4, 5)
    plt.imshow(prewitt_edges, cmap='gray')
    plt.title('Prewitt Edge Detection')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(laplacian_edges, cmap='gray')
    plt.title('Laplacian Edge Detection')
    plt.axis('off')
    
    # 第三行：不同参数的Canny检测
    plt.subplot(3, 4, 9)
    plt.imshow(canny_low, cmap='gray')
    plt.title('Canny (Low Threshold)')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(canny_medium, cmap='gray')
    plt.title('Canny (Medium Threshold)')
    plt.axis('off')
    
    plt.subplot(3, 4, 11)
    plt.imshow(canny_high, cmap='gray')
    plt.title('Canny (High Threshold)')
    plt.axis('off')
    
    # 统计分析
    plt.subplot(3, 4, 12)
    edge_counts = [
        np.sum(roberts_edges > 0),
        np.sum(prewitt_edges > 0),
        np.sum(sobel_edges > 0),
        np.sum(canny_edges > 0),
        np.sum(laplacian_edges > 0)
    ]
    
    methods = ['Roberts', 'Prewitt', 'Sobel', 'Canny', 'Laplacian']
    bars = plt.bar(methods, edge_counts, color=['red', 'green', 'blue', 'orange', 'purple'])
    plt.title('Edge Pixel Count Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Edge Pixels')
    
    # 添加数值标签
    for bar, count in zip(bars, edge_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(edge_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print("\n=== 边缘检测算法比较 ===")
    print(f"图像尺寸: {gray.shape}")
    print(f"总像素数: {gray.size}")
    print("\n边缘像素统计:")
    for method, count in zip(methods, edge_counts):
        percentage = (count / gray.size) * 100
        print(f"{method:>10}: {count:>8} pixels ({percentage:>5.2f}%)")
    
    return {
        'original': image,
        'gray': gray,
        'roberts': roberts_edges,
        'prewitt': prewitt_edges,
        'sobel': sobel_edges,
        'canny': canny_edges,
        'laplacian': laplacian_edges
    }

def detailed_analysis(results):
    """
    对边缘检测结果进行详细分析
    """
    plt.figure(figsize=(18, 12))
    
    # 梯度方向分析（以Sobel为例）
    gray = results['gray']
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度方向
    gradient_direction = np.arctan2(sobel_y, sobel_x)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    plt.subplot(3, 4, 1)
    plt.imshow(results['roberts'], cmap='gray')
    plt.title('Roberts Edges')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(results['prewitt'], cmap='gray')
    plt.title('Prewitt Edges')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(results['canny'], cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')
    
    # 边缘强度分布
    plt.subplot(3, 4, 5)
    plt.hist(results['roberts'].flatten(), bins=50, alpha=0.7, label='Roberts', color='red')
    plt.hist(results['prewitt'].flatten(), bins=50, alpha=0.7, label='Prewitt', color='green')
    plt.title('Edge Intensity Distribution')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.yscale('log')
    
    # 边缘方向分布
    plt.subplot(3, 4, 6)
    plt.hist(gradient_direction.flatten(), bins=50, alpha=0.7, color='blue')
    plt.title('Gradient Direction Distribution')
    plt.xlabel('Direction (radians)')
    plt.ylabel('Frequency')
    
    # 不同方法的边缘叠加
    plt.subplot(3, 4, 7)
    combined = np.zeros_like(results['gray'])
    combined += (results['roberts'] > 0).astype(np.uint8) * 85
    combined += (results['prewitt'] > 0).astype(np.uint8) * 85
    combined += (results['canny'] > 0).astype(np.uint8) * 85
    plt.imshow(combined, cmap='hot')
    plt.title('Combined Edge Map')
    plt.axis('off')
    
    # ROI分析（选择图像中心区域）
    h, w = results['gray'].shape
    roi_x, roi_y = w//4, h//4
    roi_w, roi_h = w//2, h//2
    
    plt.subplot(3, 4, 8)
    roi_original = results['gray'][roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    plt.imshow(roi_original, cmap='gray')
    plt.title('ROI - Original')
    plt.axis('off')
    
    plt.subplot(3, 4, 9)
    roi_roberts = results['roberts'][roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    plt.imshow(roi_roberts, cmap='gray')
    plt.title('ROI - Roberts')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    roi_prewitt = results['prewitt'][roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    plt.imshow(roi_prewitt, cmap='gray')
    plt.title('ROI - Prewitt')
    plt.axis('off')
    
    plt.subplot(3, 4, 11)
    roi_canny = results['canny'][roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    plt.imshow(roi_canny, cmap='gray')
    plt.title('ROI - Canny')
    plt.axis('off')
    
    # 性能比较雷达图
    plt.subplot(3, 4, 12)
    # 计算各种指标（简化版）
    roberts_score = np.sum(results['roberts'] > 0) / results['gray'].size
    prewitt_score = np.sum(results['prewitt'] > 0) / results['gray'].size
    canny_score = np.sum(results['canny'] > 0) / results['gray'].size
    
    methods = ['Roberts', 'Prewitt', 'Canny']
    scores = [roberts_score, prewitt_score, canny_score]
    
    plt.bar(methods, scores, color=['red', 'green', 'blue'])
    plt.title('Edge Density Comparison')
    plt.ylabel('Edge Density')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def performance_comparison():
    """
    性能比较分析
    """
    print("\n=== 算法特点比较 ===")
    
    comparison_data = {
        'Algorithm': ['Roberts', 'Prewitt', 'Sobel', 'Canny', 'Laplacian'],
        'Kernel Size': ['2x2', '3x3', '3x3', 'Multi-stage', '3x3'],
        'Noise Sensitivity': ['High', 'Medium', 'Low', 'Very Low', 'Very High'],
        'Edge Thickness': ['Thin', 'Medium', 'Medium', 'Thin', 'Thick'],
        'Computational Cost': ['Low', 'Medium', 'Medium', 'High', 'Low'],
        'Best Use Case': ['Simple edges', 'General purpose', 'Noise robustness', 'Precise edges', 'Blob detection']
    }
    
    print(f"{'Algorithm':<12} {'Kernel':<10} {'Noise Sens.':<12} {'Edge Thick.':<12} {'Cost':<10} {'Best Use'}")
    print("-" * 80)
    
    for i in range(len(comparison_data['Algorithm'])):
        print(f"{comparison_data['Algorithm'][i]:<12} "
              f"{comparison_data['Kernel Size'][i]:<10} "
              f"{comparison_data['Noise Sensitivity'][i]:<12} "
              f"{comparison_data['Edge Thickness'][i]:<12} "
              f"{comparison_data['Computational Cost'][i]:<10} "
              f"{comparison_data['Best Use Case'][i]}")

# 主程序
if __name__ == "__main__":
    # 图像路径 - 请替换为您的图像路径
    image_path = "image.jpg"  # 替换为实际的图像路径
    
    print("开始边缘检测比较分析...")
    
    try:
        # 运行比较分析
        results = compare_edge_detectors(image_path)
        
        # 详细分析
        detailed_analysis(results)
        
        # 性能比较
        performance_comparison()
        
        print("\n分析完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保图像文件存在且路径正确。")
        
        # 如果没有图像文件，创建一个测试图像
        print("\n正在创建测试图像...")
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # 绘制一些几何形状用于测试
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), 2)
        cv2.circle(test_image, (200, 100), 50, (255, 255, 255), 2)
        cv2.line(test_image, (50, 200), (250, 250), (255, 255, 255), 2)
        
        # 保存测试图像
        cv2.imwrite("test_image.jpg", test_image)
        print("测试图像已保存为 'test_image.jpg'")
        
        # 使用测试图像运行分析
        results = compare_edge_detectors("test_image.jpg")
        detailed_analysis(results)
        performance_comparison()
