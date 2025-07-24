import cv2
import numpy as np

def create_green_screen_mask(image, sensitivity_mode="medium"):
    """
    Membuat mask untuk area green screen dengan deteksi yang ditingkatkan.
    
    Args:
        image: Input image
        sensitivity_mode: "high", "medium", "low", "ultra_high"
    """
    # Try enhanced detection first
    mask = create_enhanced_green_screen_mask(image, sensitivity_mode)
    
    # Fallback to basic if enhanced fails
    if np.sum(mask) == 0:
        mask = create_basic_green_screen_mask(image, sensitivity_mode)
    
    return mask

def create_enhanced_green_screen_mask(image, sensitivity="medium"):
    """Enhanced green screen detection with multiple color spaces."""
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # HSV-based detection
    hsv_mask = create_hsv_green_mask(hsv, sensitivity)
    
    # LAB-based detection (better for certain lighting conditions)
    lab_mask = create_lab_green_mask(lab, sensitivity)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(hsv_mask, lab_mask)
    
    # Enhanced morphological operations
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)
    
    # Close small gaps
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small)
    # Remove small noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
    # Smooth edges
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large)
    
    # Gaussian blur for smoother edges
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    
    return combined_mask

def create_hsv_green_mask(hsv, sensitivity="medium"):
    """Create green mask using HSV color space."""
    if sensitivity == "ultra_high":
        # Ultra sensitive - very wide range
        lower_green1 = np.array([15, 20, 20])
        upper_green1 = np.array([105, 255, 255])
        lower_green2 = np.array([10, 15, 15])
        upper_green2 = np.array([110, 255, 255])
    elif sensitivity == "high":
        # Sensitivitas tinggi - range hijau lebih luas
        lower_green1 = np.array([20, 25, 25])
        upper_green1 = np.array([100, 255, 255])
        lower_green2 = np.array([15, 30, 30])
        upper_green2 = np.array([95, 255, 255])
    elif sensitivity == "low":
        # Sensitivitas rendah - range hijau lebih ketat
        lower_green1 = np.array([40, 50, 50])
        upper_green1 = np.array([80, 255, 255])
        lower_green2 = np.array([35, 60, 60])
        upper_green2 = np.array([85, 255, 255])
    else:  # medium
        # Sensitivitas sedang - default
        lower_green1 = np.array([35, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        lower_green2 = np.array([25, 30, 30])
        upper_green2 = np.array([95, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    
    return cv2.bitwise_or(mask1, mask2)

def create_lab_green_mask(lab, sensitivity="medium"):
    """Create green mask using LAB color space."""
    # In LAB, green is represented by negative A values
    if sensitivity == "ultra_high":
        lower_lab = np.array([20, 0, 0])
        upper_lab = np.array([255, 120, 255])
    elif sensitivity == "high":
        lower_lab = np.array([30, 0, 0])
        upper_lab = np.array([255, 110, 255])
    elif sensitivity == "low":
        lower_lab = np.array([50, 0, 0])
        upper_lab = np.array([255, 100, 255])
    else:  # medium
        lower_lab = np.array([40, 0, 0])
        upper_lab = np.array([255, 105, 255])
    
    return cv2.inRange(lab, lower_lab, upper_lab)

def create_basic_green_screen_mask(image, sensitivity="medium"):
    """Deteksi green screen basic dengan level sensitivitas."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if sensitivity == "high":
        # Sensitivitas tinggi - range hijau lebih luas
        lower_green1 = np.array([20, 25, 25])
        upper_green1 = np.array([100, 255, 255])
        lower_green2 = np.array([15, 30, 30])
        upper_green2 = np.array([95, 255, 255])
    elif sensitivity == "low":
        # Sensitivitas rendah - range hijau lebih ketat
        lower_green1 = np.array([40, 50, 50])
        upper_green1 = np.array([80, 255, 255])
        lower_green2 = np.array([35, 60, 60])
        upper_green2 = np.array([85, 255, 255])
    else:  # medium
        # Sensitivitas sedang - default
        lower_green1 = np.array([35, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        lower_green2 = np.array([25, 30, 30])
        upper_green2 = np.array([95, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Basic morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask

def find_greenscreen_area(image_path, sensitivity_mode="medium"):
    """
    Mendeteksi area hijau pada gambar dan mengembalikan kotak pembatasnya.
    """
    try:
        image = cv2.imread(image_path)
        if image is None: 
            return None
        
        mask = create_green_screen_mask(image, sensitivity_mode)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: 
            print("⚠️ No green screen area detected, trying with high sensitivity...")
            # Coba dengan sensitivitas tinggi
            mask = create_green_screen_mask(image, "high")
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        print(f"✅ Green screen area detected: {area} pixels")
        
        return cv2.boundingRect(largest_contour)
        
    except Exception as e:
        print(f"Error saat mendeteksi green screen: {e}")
        return None

def analyze_green_screen_quality(image_path):
    """Analisis kualitas green screen untuk memberikan rekomendasi."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Could not load image"
        
        # Test dengan berbagai mode
        medium_mask = create_green_screen_mask(image, "medium")
        high_mask = create_green_screen_mask(image, "high")
        
        medium_pixels = np.sum(medium_mask > 0)
        high_pixels = np.sum(high_mask > 0)
        total_pixels = image.shape[0] * image.shape[1]
        
        medium_coverage = (medium_pixels / total_pixels) * 100
        high_coverage = (high_pixels / total_pixels) * 100
        
        analysis = f"""
🎯 Green Screen Analysis:
• Medium Detection: {medium_pixels} pixels ({medium_coverage:.1f}% coverage)
• High Detection: {high_pixels} pixels ({high_coverage:.1f}% coverage)

📊 Quality Assessment:
"""
        
        if medium_coverage > 15:
            analysis += "✅ Excellent green screen quality"
        elif medium_coverage > 8:
            analysis += "✅ Good green screen quality"
        elif medium_coverage > 3:
            analysis += "⚠️ Fair green screen quality - consider using high sensitivity"
        else:
            analysis += "❌ Poor green screen quality - check template"
        
        return analysis
        
    except Exception as e:
        return f"Analysis error: {e}"