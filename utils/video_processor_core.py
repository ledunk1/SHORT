"""
Video Processor Core - Main processing logic
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.green_screen_detection import create_green_screen_mask
from utils.video_processing import process_frame_with_green_screen
from utils.blur_processing import process_blur_frame
from utils.file_operations import get_video_properties, add_audio_to_video
from utils.text_rendering import smart_text_wrap, render_text_with_emoji_multiline

class VideoProcessorCore:
    """Core video processing functionality."""
    
    def __init__(self, gui_manager=None):
        self.gui_manager = gui_manager
    
    def process_single_video(self, video_path, template, template_mask, output_path, 
                           text_settings, gpu_settings):
        """Process a single video with green screen."""
        print(f"🎬 Processing: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        fps, width, height = get_video_properties(video_path)
        
        # Create temporary output in current working directory
        current_dir = os.getcwd()
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        temp_output = os.path.join(current_dir, f"temp_{base_name}_{os.getpid()}.mp4")
        
        print(f"🔧 Temp file will be created at: {temp_output}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (1080, 1920))
        
        if not out.isOpened():
            print(f"❌ Could not create output file: {temp_output}")
            cap.release()
            return False
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.basename(video_path)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with green screen
                processed_frame = process_frame_with_green_screen(template, frame, template_mask)
                
                # Add text overlay if enabled (TEXT DI LAPISAN PALING DEPAN)
                if text_settings['enabled']:
                    processed_frame = self.add_text_overlay(processed_frame, video_name, text_settings)
                
                # Ensure frame is correct size
                if processed_frame.shape[:2] != (1920, 1080):
                    processed_frame = cv2.resize(processed_frame, (1080, 1920))
                
                out.write(processed_frame)
                frame_count += 1
                
                # Update progress
                if self.gui_manager and frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.gui_manager.update_progress(
                        progress, 
                        f"Processing {video_name}: {frame_count}/{total_frames} frames"
                    )
        
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            return False
        
        finally:
            cap.release()
            out.release()
        
        print(f"✅ Video processing completed: {frame_count} frames")
        return temp_output
    
    def process_single_video_blur(self, video_path, output_path, blur_settings, 
                                text_settings, gpu_settings):
        """Process a single video with blur background."""
        print(f"🌀 Processing blur: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        fps, width, height = get_video_properties(video_path)
        
        # Create temporary output in current working directory
        current_dir = os.getcwd()
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        temp_output = os.path.join(current_dir, f"temp_{base_name}_{os.getpid()}.mp4")
        
        print(f"🔧 Temp file will be created at: {temp_output}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (1080, 1920))
        
        if not out.isOpened():
            print(f"❌ Could not create output file: {temp_output}")
            cap.release()
            return False
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.basename(video_path)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with blur
                processed_frame = process_blur_frame(
                    frame,
                    blur_settings['crop_top'],
                    blur_settings['crop_bottom'],
                    blur_settings['video_x_position'],
                    blur_settings['video_y_position']
                )
                
                # Add text overlay if enabled (TEXT DI LAPISAN PALING DEPAN)
                if text_settings['enabled']:
                    processed_frame = self.add_text_overlay(processed_frame, video_name, text_settings)
                
                # Ensure frame is correct size
                if processed_frame.shape[:2] != (1920, 1080):
                    processed_frame = cv2.resize(processed_frame, (1080, 1920))
                
                out.write(processed_frame)
                frame_count += 1
                
                # Update progress
                if self.gui_manager and frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.gui_manager.update_progress(
                        progress, 
                        f"Processing {video_name}: {frame_count}/{total_frames} frames"
                    )
        
        except Exception as e:
            print(f"❌ Error processing blur video: {e}")
            return False
        
        finally:
            cap.release()
            out.release()
        
        print(f"✅ Blur processing completed: {frame_count} frames")
        return temp_output
    
    def add_text_overlay(self, frame, video_name, text_settings):
        """Add text overlay to frame."""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # Load font
            try:
                font_file = self.get_font_file(text_settings['font'])
                font = ImageFont.truetype(font_file, text_settings['size'])
            except:
                font = ImageFont.load_default()
            
            # Prepare text (remove file extension and replace underscores)
            video_name_text = os.path.splitext(video_name)[0].replace("_", " ")
            
            # Calculate available text area
            max_text_width = 1080 - 80  # 40px margin on each side
            
            # Auto-wrap text based on frame width
            lines = smart_text_wrap(video_name_text, draw, font, max_text_width, emoji_size=80)
            
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
    
    def get_font_file(self, font_name):
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