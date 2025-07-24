import cv2
import numpy as np
from .green_screen_detection import create_green_screen_mask

def fit_video_to_mask(video_frame, mask):
    """Menyesuaikan video frame dengan bentuk mask green screen."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return video_frame, (0, 0, video_frame.shape[1], video_frame.shape[0]), None
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # FIXED: Use INTER_AREA for better quality when downscaling, INTER_CUBIC for upscaling
    if w * h < video_frame.shape[0] * video_frame.shape[1]:
        # Downscaling - use INTER_AREA for better quality
        resized_video = cv2.resize(video_frame, (w, h), interpolation=cv2.INTER_AREA)
    else:
        # Upscaling - use INTER_CUBIC for smoother results
        resized_video = cv2.resize(video_frame, (w, h), interpolation=cv2.INTER_CUBIC)
    
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    adjusted_contour = largest_contour - [x, y]
    cv2.fillPoly(contour_mask, [adjusted_contour], 255)
    
    masked_video = cv2.bitwise_and(resized_video, resized_video, mask=contour_mask)
    
    alpha = contour_mask.astype(np.float32) / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)
    
    background = np.zeros_like(masked_video)
    blended = (masked_video * alpha + background * (1 - alpha)).astype(np.uint8)
    
    return blended, (x, y, w, h), contour_mask

def process_frame_with_green_screen(background_frame, video_frame, template_mask=None):
    """
    Memproses frame dengan mengganti green screen dengan video.
    PENTING: Text overlay harus ditambahkan SETELAH fungsi ini dipanggil
    agar text berada di lapisan paling depan.
    
    Args:
        background_frame: Template frame dengan green screen
        video_frame: Video frame yang akan di-overlay
        template_mask: Pre-computed mask (optional, akan dibuat jika None)
    """
    # Gunakan mask yang sudah ada atau buat baru
    if template_mask is None:
        mask = create_green_screen_mask(background_frame, "medium")
    else:
        mask = template_mask
    
    fitted_video, bbox, contour_mask = fit_video_to_mask(video_frame, mask)
    x, y, w, h = bbox
    
    result = background_frame.copy()
    
    if fitted_video is not None and contour_mask is not None:
        # Improved blending with better edge handling
        mask_3ch = cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        
        # Apply slight blur to mask edges for smoother blending
        mask_3ch = cv2.GaussianBlur(mask_3ch, (3, 3), 0)
        
        roi = result[y:y+h, x:x+w]
        blended_roi = (fitted_video * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
        result[y:y+h, x:x+w] = blended_roi
    
    # CATATAN: Text overlay TIDAK ditambahkan di sini
    # Text harus ditambahkan di lapisan terakhir agar berada di depan
    return result

def create_enhanced_mask(background_frame, sensitivity="medium"):
    """
    Buat mask dengan deteksi yang ditingkatkan.
    Wrapper function untuk backward compatibility.
    """
    return create_green_screen_mask(background_frame, sensitivity)

def analyze_template_quality(template_path):
    """Analisis kualitas template green screen."""
    from .green_screen_detection import analyze_green_screen_quality
    return analyze_green_screen_quality(template_path)