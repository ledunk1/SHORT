import cv2
import numpy as np

def create_blurred_background(frame, blur_strength=51):
    """Membuat background blur dari frame video."""
    # Pastikan blur_strength adalah angka ganjil
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
    return blurred

def crop_video_frame(frame, crop_top_percent, crop_bottom_percent):
    """Memotong frame video dari atas dan bawah berdasarkan persentase."""
    h, w = frame.shape[:2]
    
    crop_top_px = int(h * crop_top_percent / 100)
    crop_bottom_px = int(h * crop_bottom_percent / 100)
    
    # Pastikan tidak memotong seluruh frame
    if crop_top_px + crop_bottom_px >= h:
        crop_top_px = 0
        crop_bottom_px = 0
    
    cropped_frame = frame[crop_top_px:h-crop_bottom_px, :]
    return cropped_frame

def fit_video_to_9_16(cropped_frame, target_width, target_height):
    """Menyesuaikan video yang sudah di-crop ke aspect ratio 9:16."""
    h, w = cropped_frame.shape[:2]
    
    # Hitung aspect ratio target (9:16)
    target_ratio = target_width / target_height
    current_ratio = w / h
    
    if current_ratio > target_ratio:
        # Video terlalu lebar, fit berdasarkan tinggi
        new_height = target_height
        new_width = int(new_height * current_ratio)
        
        # Jika masih terlalu lebar, crop dari samping
        if new_width > target_width:
            new_width = target_width
            new_height = int(new_width / current_ratio)
    else:
        # Video terlalu tinggi, fit berdasarkan lebar
        new_width = target_width
        new_height = int(new_width / current_ratio)
        
        # Jika masih terlalu tinggi, crop dari atas/bawah
        if new_height > target_height:
            new_height = target_height
            new_width = int(new_height * current_ratio)
    
    # Resize video
    resized_video = cv2.resize(cropped_frame, (new_width, new_height))
    
    return resized_video, new_width, new_height

def process_blur_frame(original_frame, crop_top_percent, crop_bottom_percent, 
                      video_x_percent=50, video_y_percent=50,
                      target_width=1080, target_height=1920, blur_strength=51, text_overlay_frame=None):
    """
    Memproses frame dengan blur background mode dengan posisi video yang dapat diatur.
    PENTING: Text overlay harus ditambahkan SETELAH fungsi ini dipanggil
    agar text berada di lapisan paling depan.
    """
    
    # 1. Buat background blur
    blurred_bg = create_blurred_background(original_frame, blur_strength)
    blurred_bg = cv2.resize(blurred_bg, (target_width, target_height))
    
    # 2. Crop video asli
    cropped_video = crop_video_frame(original_frame, crop_top_percent, crop_bottom_percent)
    
    # 3. Fit video ke aspect ratio 9:16
    fitted_video, video_width, video_height = fit_video_to_9_16(
        cropped_video, target_width, target_height
    )
    
    # 4. Hitung posisi video berdasarkan persentase X dan Y
    max_x = target_width - video_width
    max_y = target_height - video_height
    
    # Pastikan video tidak keluar dari frame
    if max_x < 0:
        max_x = 0
        video_width = target_width
        fitted_video = cv2.resize(fitted_video, (video_width, video_height))
    
    if max_y < 0:
        max_y = 0
        video_height = target_height
        fitted_video = cv2.resize(fitted_video, (video_width, video_height))
    
    # Hitung offset berdasarkan persentase
    x_offset = int((video_x_percent / 100) * max_x) if max_x > 0 else 0
    y_offset = int((video_y_percent / 100) * max_y) if max_y > 0 else 0
    
    # Pastikan offset tidak negatif dan tidak melebihi batas
    x_offset = max(0, min(x_offset, max_x))
    y_offset = max(0, min(y_offset, max_y))
    
    # 5. Overlay video pada background blur
    result = blurred_bg.copy()
    
    # Pastikan dimensi video sesuai dengan area yang akan di-overlay
    end_y = min(y_offset + video_height, target_height)
    end_x = min(x_offset + video_width, target_width)
    
    actual_height = end_y - y_offset
    actual_width = end_x - x_offset
    
    if actual_height > 0 and actual_width > 0:
        # Resize fitted_video jika perlu untuk menyesuaikan area yang tersedia
        if actual_height != video_height or actual_width != video_width:
            fitted_video = cv2.resize(fitted_video, (actual_width, actual_height))
        
        result[y_offset:end_y, x_offset:end_x] = fitted_video
    
    # CATATAN: Text overlay TIDAK ditambahkan di sini
    # Text harus ditambahkan di lapisan terakhir agar berada di depan
    return result

def calculate_optimal_crop(video_path, target_ratio=9/16):
    """Menghitung crop optimal untuk video agar mendekati aspect ratio 9:16."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return 0, 0  # Default no crop
    
    h, w = frame.shape[:2]
    current_ratio = w / h
    
    if current_ratio > target_ratio:
        # Video terlalu lebar, tidak perlu crop atas/bawah
        return 0, 0
    else:
        # Video terlalu tinggi, hitung crop yang diperlukan
        ideal_height = w / target_ratio
        excess_height = h - ideal_height
        
        if excess_height > 0:
            # Bagi crop antara atas dan bawah
            crop_percent = (excess_height / h) * 100 / 2  # Bagi 2 untuk atas dan bawah
            crop_percent = min(crop_percent, 20)  # Maksimal 20% per sisi
            return crop_percent, crop_percent
    
    return 0, 0