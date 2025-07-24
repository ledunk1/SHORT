"""
Image Processing Module
Handles processing of static images for greenscreen and blur modes
"""

import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.green_screen_detection import create_green_screen_mask
from utils.video_processing import process_frame_with_green_screen
from utils.blur_processing import process_blur_frame
from utils.text_rendering import smart_text_wrap, render_text_with_emoji_multiline

def process_image_greenscreen(image_path, template, template_mask, output_path, text_settings, duration=5):
    """
    Process static image with green screen template - creates MP4 output (STABLE VERSION).
    Uses single pre-processed image frame to prevent glitch/shaking.
    
    Args:
        image_path: Path to input image
        template: Template frame with green screen
        template_mask: Pre-computed green screen mask
        output_path: Output MP4 path
        text_settings: Text overlay settings
        duration: Duration of output video in seconds
    """
    print(f"🖼️ Processing image with green screen: {os.path.basename(image_path)}")
    
    # Load input image
    input_image = cv2.imread(image_path)
    if input_image is None:
        print(f"❌ Could not load image: {image_path}")
        return False
    
    # FIXED: Detect green screen area from template to get exact dimensions
    green_area = detect_green_screen_area_from_template(template)
    if green_area is None:
        print("❌ No green screen area detected in template")
        return False
    
    x, y, w, h = green_area
    print(f"✅ Green screen area detected: ({x}, {y}) size {w}x{h}")
    
    # FIXED: Pre-process replacement image ONCE to exact green screen dimensions
    replacement_image = cv2.resize(input_image, (w, h), interpolation=cv2.INTER_AREA)
    print(f"✅ Image resized to green screen area: {w}x{h}")
    
    # Calculate output parameters
    fps = 30  # Standard FPS for image-to-video
    total_frames = duration * fps
    
    print(f"🎯 Creating {total_frames} frames at {fps}fps ({duration}s duration)")
    
    # FIXED: Create temporary output for audio processing later
    current_dir = os.getcwd()
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    temp_output = os.path.join(current_dir, f"temp_{base_name}_{os.getpid()}.mp4")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    template_height, template_width = template.shape[:2]
    out = cv2.VideoWriter(temp_output, fourcc, fps, (template_width, template_height))
    
    if not out.isOpened():
        print(f"❌ Could not create temp output file: {temp_output}")
        return False
    
    try:
        # FIXED: Create stable processed frame using optimized blending method
        processed_frame = create_stable_greenscreen_frame(
            template, replacement_image, green_area, template_mask
        )
        
        # Add text overlay if enabled
        if text_settings and text_settings['enabled']:
            image_name = os.path.basename(image_path)
            processed_frame = add_text_overlay_to_frame(processed_frame, image_name, text_settings)
        
        # FIXED: Resize to final output dimensions (9:16 aspect ratio)
        final_frame = cv2.resize(processed_frame, (1080, 1920), interpolation=cv2.INTER_AREA)
        
        # FIXED: Write the same final frame multiple times (no re-processing)
        for frame_index in range(total_frames):
            out.write(final_frame)
            
            if (frame_index + 1) % 60 == 0:
                progress = ((frame_index + 1) / total_frames) * 100
                print(f"📊 Progress: {frame_index + 1}/{total_frames} frames ({progress:.1f}%)")
        
        print(f"✅ Image greenscreen processing completed: {total_frames} frames")
        
        # FIXED: Return temp file path for audio processing
        return temp_output
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        out.release()

def process_image_with_video_template_duration(image_path, template_info, output_path, text_settings):
    """
    Process static image with video template - duration follows template video (STABLE VERSION).
    Pre-processes image once per template frame to prevent glitch.
    
    Args:
        image_path: Path to input image
        template_info: Template information including frames and durations
        output_path: Output MP4 path
        text_settings: Text overlay settings
    """
    print(f"🖼️ Processing image with video template (duration follows template)")
    print(f"   Image: {os.path.basename(image_path)}")
    print(f"   Template: {os.path.basename(template_info.get('path', ''))}")
    
    # Load input image
    input_image = cv2.imread(image_path)
    if input_image is None:
        print(f"❌ Could not load image: {image_path}")
        return False
    
    # FIXED: Use original template video instead of extracted frames for consistent processing
    template_path = template_info.get('path')
    if not template_path or not os.path.exists(template_path):
        print("❌ Template video path not found")
        return False
    
    # Open template video directly for full quality
    template_cap = cv2.VideoCapture(template_path)
    if not template_cap.isOpened():
        print("❌ Could not open template video")
        return False
    
    # Get template video properties
    template_fps = template_cap.get(cv2.CAP_PROP_FPS)
    template_total_frames = int(template_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    template_width = int(template_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    template_height = int(template_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Template: {template_total_frames} frames at {template_fps:.1f}fps ({template_width}x{template_height})")
    
    # FIXED: Detect green screen area from first template frame
    ret, first_template_frame = template_cap.read()
    if not ret:
        print("❌ Could not read first template frame")
        template_cap.release()
        return False
    
    green_area = detect_green_screen_area_from_template(first_template_frame)
    if green_area is None:
        print("❌ No green screen area detected in template")
        template_cap.release()
        return False
    
    x, y, w, h = green_area
    print(f"✅ Green screen area: ({x}, {y}) size {w}x{h}")
    
    # FIXED: Pre-process replacement image ONCE to exact green screen dimensions
    replacement_image = cv2.resize(input_image, (w, h), interpolation=cv2.INTER_AREA)
    print(f"✅ Image pre-processed to green screen size: {w}x{h}")
    
    # Reset template video to beginning
    template_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Calculate total duration from template video
    total_duration = template_total_frames / template_fps if template_fps > 0 else template_total_frames / 30.0
    
    # Use template FPS for output to maintain smooth playback
    output_fps = int(template_fps) if template_fps > 0 else 30
    total_output_frames = template_total_frames  # Use exact frame count from template
    
    print(f"🎯 Output: {total_output_frames} frames at {output_fps}fps ({total_duration:.2f}s duration)")
    
    # FIXED: Create temporary output for audio processing later
    current_dir = os.getcwd()
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    temp_output = os.path.join(current_dir, f"temp_{base_name}_{os.getpid()}.mp4")
    
    # Setup video writer with template FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, output_fps, (template_width, template_height))
    
    if not out.isOpened():
        print(f"❌ Could not create temp output file: {temp_output}")
        template_cap.release()
        return False
    
    try:
        print(f"🔄 Processing {total_output_frames} frames with original template quality...")
        
        frame_index = 0
        while frame_index < total_output_frames:
            # Read template frame directly from video (full quality)
            ret_template, template_frame = template_cap.read()
            if not ret_template:
                print(f"❌ Could not read template frame {frame_index}")
                break
            
            # FIXED: Create stable processed frame using optimized method
            processed_frame = create_stable_greenscreen_frame_for_template(
                template_frame, replacement_image, green_area
            )
            
            # Add text overlay if enabled
            if text_settings and text_settings['enabled']:
                image_name = os.path.basename(image_path)
                processed_frame = add_text_overlay_to_frame(processed_frame, image_name, text_settings)
            
            # FIXED: Resize to final output dimensions
            final_frame = cv2.resize(processed_frame, (1080, 1920), interpolation=cv2.INTER_AREA)
            
            out.write(final_frame)
            frame_index += 1
            
            if frame_index % 60 == 0:
                progress = (frame_index / total_output_frames) * 100
                print(f"📊 Progress: {frame_index}/{total_output_frames} frames ({progress:.1f}%)")
        
        print(f"✅ Image with video template processing completed: {frame_index} frames")
        
        # FIXED: Return temp file path for audio processing
        return temp_output
        
    except Exception as e:
        print(f"❌ Error processing image with video template: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        template_cap.release()
        out.release()
def process_image_blur(image_path, output_path, blur_settings, text_settings, duration=5):
    """
    Process static image with blur background - creates MP4 output (STABLE VERSION).
    Pre-processes image once to prevent glitch/shaking.
    
    Args:
        image_path: Path to input image
        output_path: Output MP4 path
        blur_settings: Blur processing settings
        text_settings: Text overlay settings
        duration: Duration of output video in seconds
    """
    print(f"🌀 Processing image with blur background: {os.path.basename(image_path)}")
    
    # Load input image
    input_image = cv2.imread(image_path)
    if input_image is None:
        print(f"❌ Could not load image: {image_path}")
        return False
    
    # Calculate output parameters
    fps = 30  # Standard FPS for image-to-video
    total_frames = duration * fps
    
    print(f"🎯 Creating {total_frames} frames at {fps}fps ({duration}s duration)")
    
    # FIXED: Create temporary output for audio processing later
    current_dir = os.getcwd()
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    temp_output = os.path.join(current_dir, f"temp_{base_name}_{os.getpid()}.mp4")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (1080, 1920))
    
    if not out.isOpened():
        print(f"❌ Could not create temp output file: {temp_output}")
        return False
    
    try:
        # FIXED: Pre-process the image with blur ONCE to prevent shaking/glitch
        processed_frame = process_blur_frame(
            input_image,
            blur_settings['crop_top'],
            blur_settings['crop_bottom'],
            blur_settings['video_x_position'],
            blur_settings['video_y_position'],
            1080,  # target_width
            1920   # target_height
        )
        
        # Add text overlay if enabled
        if text_settings and text_settings['enabled']:
            image_name = os.path.basename(image_path)
            processed_frame = add_text_overlay_to_frame(processed_frame, image_name, text_settings)
        
        # Ensure frame is correct size
        if processed_frame.shape[:2] != (1920, 1080):
            processed_frame = cv2.resize(processed_frame, (1080, 1920))
        
        # FIXED: Write the EXACT same processed frame multiple times (no re-processing)
        for frame_index in range(total_frames):
            out.write(processed_frame)
            
            if (frame_index + 1) % 60 == 0:
                progress = ((frame_index + 1) / total_frames) * 100
                print(f"📊 Progress: {frame_index + 1}/{total_frames} frames ({progress:.1f}%)")
        
        print(f"✅ Image blur processing completed: {total_frames} frames")
        
        # FIXED: Return temp file path for audio processing
        return temp_output
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        out.release()

def add_text_overlay_to_frame(frame, image_name, text_settings):
    """Add text overlay to a single frame."""
    try:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Load font
        try:
            font_file = get_font_file(text_settings['font'])
            font = ImageFont.truetype(font_file, text_settings['size'])
        except:
            font = ImageFont.load_default()
        
        # Prepare text (remove file extension and replace underscores)
        image_name_text = os.path.splitext(image_name)[0].replace("_", " ")
        
        # Calculate available text area
        max_text_width = 1080 - 80  # 40px margin on each side
        
        # Auto-wrap text based on frame width
        lines = smart_text_wrap(image_name_text, draw, font, max_text_width, emoji_size=80)
        
        # Calculate position based on settings
        x_percent = text_settings['x_position'] / 100
        y_percent = text_settings['y_position'] / 100
        
        # Calculate total text height
        line_height = text_settings['size'] + 10
        total_text_height = len(lines) * line_height
        
        # Y position based on percentage, with auto-adjustment
        base_y = int(y_percent * (1920 - total_text_height - 40))
        base_y = max(20, min(base_y, 1920 - total_text_height - 20))
        
        # Render multiline text with emoji
        rendered_lines = render_text_with_emoji_multiline(
            draw, lines, font, 1080, 1920, 
            base_y, emoji_size=80, line_spacing=10
        )
        
        # Get text color from settings
        text_color = text_settings.get('color', '#000000')
        
        # Draw text and emoji
        for line_data in rendered_lines:
            for item_type, item, x_offset in line_data['items']:
                if item_type == 'emoji':
                    # Position emoji adjusted to font height
                    emoji_y = line_data['y'] + (text_settings['size'] - line_data['emoji_size']) // 2
                    pil_image.paste(item, (line_data['x_start'] + x_offset, emoji_y), item)
                elif item_type == 'text':
                    # Draw text with selected color
                    draw.text((line_data['x_start'] + x_offset, line_data['y']), 
                             item, font=font, fill=text_color)
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    except Exception as e:
        print(f"⚠️ Text overlay error: {e}")
        return frame  # Return original frame if text overlay fails

def get_font_file(font_name):
    """Get font file based on font name."""
    font_mapping = {
        "Arial": "arial.ttf",
        "Times New Roman": "times.ttf",
        "Helvetica": "arial.ttf",  # fallback
        "Courier New": "cour.ttf",
        "Verdana": "verdana.ttf",
        "Georgia": "georgia.ttf",
        "Comic Sans MS": "comic.ttf",
        "Impact": "impact.ttf",
        "Trebuchet MS": "trebuc.ttf",
        "Tahoma": "tahoma.ttf"
    }
    return font_mapping.get(font_name, "arial.ttf")

def detect_green_screen_area_from_template(template_frame):
    """
    Detect green screen area from template frame using optimized method.
    Based on ref/b.py approach for stable detection.
    """
    try:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(template_frame, cv2.COLOR_BGR2HSV)
        
        # Green screen detection parameters (from ref/b.py)
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        min_area = 1000
        
        # Create mask for green color
        mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Clean up mask (smaller kernel for better performance)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest contour
        largest_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area and area > max_area:
                max_area = area
                largest_contour = contour
        
        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x, y, w, h)
        
        return None
        
    except Exception as e:
        print(f"❌ Error detecting green screen area: {e}")
        return None

def create_stable_greenscreen_frame(template_frame, replacement_image, green_area, template_mask=None):
    """
    Create stable green screen frame using optimized blending method.
    Based on ref/b.py approach to prevent glitch/shaking.
    """
    try:
        x, y, w, h = green_area
        
        # Create mask from green area (optimized method from ref/b.py)
        if template_mask is None:
            # Work only on ROI to save memory
            roi = template_frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Green screen parameters
            green_lower = np.array([35, 50, 50])
            green_upper = np.array([85, 255, 255])
            
            # Create mask
            mask_roi = cv2.inRange(hsv_roi, green_lower, green_upper)
            
            # Light morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)
            
            # Light blur for smoother edges
            mask_roi = cv2.medianBlur(mask_roi, 3)
            
            # Create full mask
            mask = np.zeros(template_frame.shape[:2], dtype=np.uint8)
            mask[y:y+h, x:x+w] = mask_roi
        else:
            mask = template_mask
        
        # Convert mask to 3D for blending
        mask_3d = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
        
        # Create positioned replacement (exact size match)
        positioned_replacement = np.zeros_like(template_frame, dtype=np.uint8)
        positioned_replacement[y:y+h, x:x+w] = replacement_image
        
        # Optimized blending (from ref/b.py)
        template_frame_f = template_frame.astype(np.float32)
        positioned_replacement_f = positioned_replacement.astype(np.float32)
        
        result = template_frame_f * (1 - mask_3d) + positioned_replacement_f * mask_3d
        result_frame = result.astype(np.uint8)
        
        return result_frame
        
    except Exception as e:
        print(f"❌ Error creating stable greenscreen frame: {e}")
        return template_frame  # Return original template as fallback

def create_stable_greenscreen_frame_for_template(template_frame, replacement_image, green_area):
    """
    Create stable green screen frame for video template processing.
    Optimized version that detects green screen area per frame.
    """
    try:
        x, y, w, h = green_area
        
        # Create optimized mask for current template frame
        roi = template_frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Green screen parameters
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        
        # Create mask
        mask_roi = cv2.inRange(hsv_roi, green_lower, green_upper)
        
        # Light morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)
        
        # Light blur for smoother edges
        mask_roi = cv2.medianBlur(mask_roi, 3)
        
        # Create full mask
        mask = np.zeros(template_frame.shape[:2], dtype=np.uint8)
        mask[y:y+h, x:x+w] = mask_roi
        
        # Convert mask to 3D for blending
        mask_3d = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
        
        # Create positioned replacement
        positioned_replacement = np.zeros_like(template_frame, dtype=np.uint8)
        positioned_replacement[y:y+h, x:x+w] = replacement_image
        
        # Optimized blending
        template_frame_f = template_frame.astype(np.float32)
        positioned_replacement_f = positioned_replacement.astype(np.float32)
        
        result = template_frame_f * (1 - mask_3d) + positioned_replacement_f * mask_3d
        result_frame = result.astype(np.uint8)
        
        return result_frame
        
    except Exception as e:
        print(f"❌ Error creating stable greenscreen frame for template: {e}")
        return template_frame  # Return original template as fallback

def is_supported_image_format(file_path):
    """Check if the file is a supported image format."""
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    return file_path.lower().endswith(supported_extensions)

def get_image_dimensions(image_path):
    """Get image dimensions without loading the full image."""
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        print(f"⚠️ Could not get image dimensions: {e}")
        return None

def resize_image_for_processing(image, target_width=1080, target_height=1920):
    """Resize image to fit processing dimensions while maintaining aspect ratio."""
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create canvas with target dimensions
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Center the resized image on canvas
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas