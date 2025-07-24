import tkinter as tk
from tkinter import filedialog
import os
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from utils.green_screen_detection import create_green_screen_mask
from utils.text_rendering import smart_text_wrap, render_text_with_emoji_multiline

class TemplateSection:
    """Template section for green screen mode."""
    
    def __init__(self, parent_frame, update_preview_callback):
        self.parent_frame = parent_frame
        self.update_preview_callback = update_preview_callback
        self.background_image_path = ""
        self.preview_label = None
        self.is_video_template = False
        self.video_frames = []
        self.video_durations = []
        self.current_frame_index = 0
        self.animation_job = None
        self.create_template_section()
    
    def create_template_section(self):
        """Create template section."""
        self.template_frame = tk.LabelFrame(
            self.parent_frame, 
            text="🖼️ Green Screen Template", 
            font=("Arial", 11, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50", 
            padx=10, 
            pady=8
        )
        self.template_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.background_label = tk.Label(
            self.template_frame, 
            text="No template selected", 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            fg="#7f8c8d"
        )
        self.background_label.pack(pady=5)
        
        select_btn = tk.Button(
            self.template_frame, 
            text="📁 Select Template (Image/GIF)", 
            command=self.select_background_image, 
            font=("Arial", 10, "bold"), 
            bg="#27ae60", 
            fg="white", 
            activebackground="#229954"
        )
        select_btn.pack(pady=5)
        
        self.preview_text = tk.Label(
            self.template_frame, 
            text="", 
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0"
        )
        self.preview_text.pack(pady=2)
        
        # Preview
        self.create_preview()
    
    def create_preview(self):
        """Create preview area."""
        preview_frame = tk.Frame(self.template_frame, bg="#e0e0e0", relief=tk.SUNKEN, bd=1)
        preview_frame.pack(pady=8)
        
        preview_title = tk.Label(
            preview_frame, 
            text="📱 Template Preview (9:16)", 
            font=("Arial", 10, "bold"), 
            bg="#e0e0e0"
        )
        preview_title.pack(pady=5)
        
        self.preview_label = tk.Label(
            preview_frame, 
            text="Upload template to see preview\nSupports: JPG, PNG, BMP, MP4, AVI, MOV", 
            bg="#ffffff", 
            fg="#95a5a6", 
            width=22, 
            height=14,
            relief=tk.SUNKEN, 
            bd=1
        )
        self.preview_label.pack(pady=8, padx=8)
    
    def select_background_image(self):
        """Select background image or GIF template."""
        path = filedialog.askopenfilename(
            title="Select Template with Green Screen (Image or Video)",
            filetypes=[
                ("All Supported", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov"),
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("Video Files", "*.mp4 *.avi *.mov"),
                ("All Files", "*.*")
            ]
        )
        if path:
            self.background_image_path = path
            filename = os.path.basename(self.background_image_path)
            self.background_label.config(text=f"Template: {filename}")
            
            # Stop any existing animation
            self.stop_animation()
            
            # Check if it's a video file
            self.is_video_template = path.lower().endswith(('.mp4', '.avi', '.mov'))
            
            try:
                if self.is_video_template:
                    self.load_video_template(path)
                else:
                    self.load_image_template(path)
                
                self.update_preview_callback()
            except Exception as e:
                self.preview_text.config(text=f"Error loading template: {str(e)}")
                print(f"Error loading template: {e}")
    
    def load_video_template(self, video_path):
        """Load video template and extract frames."""
        try:
            import cv2
            
            print(f"🎬 Loading video template: {os.path.basename(video_path)}")
            
            # Extract frames from video with memory optimization
            cap = cv2.VideoCapture(video_path)
            
            # Check if video opened successfully
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📊 Video info: {total_frames} frames, {fps:.1f}fps, {width}x{height}")
            
            # Memory optimization: limit frames and resize
            max_frames = min(total_frames, 30)  # Limit to 30 frames max for preview
            frame_step = max(1, total_frames // max_frames)  # Skip frames if too many
            
            frames = []
            frame_duration = int(1000 / fps) if fps > 0 else 100  # ms per frame
            frame_count = 0
            extracted_count = 0
            
            print(f"🔧 Preview optimization: extracting every {frame_step} frame(s), max {max_frames} frames")
            
            while extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every nth frame to save memory
                if frame_count % frame_step == 0:
                    # Resize frame to reduce memory usage (preview size)
                    preview_frame = cv2.resize(frame, (320, 240))  # Small preview size
                    frames.append(preview_frame)
                    extracted_count += 1
                    
                    if extracted_count % 10 == 0:
                        print(f"📊 Preview: {extracted_count}/{max_frames} frames")
                
                frame_count += 1
            
            cap.release()
            
            if frames:
                self.video_frames = frames
                self.video_durations = [frame_duration * frame_step] * len(frames)  # Adjust duration
                self.current_frame_index = 0
                
                print(f"✅ Video template loaded: {len(frames)} frames (optimized from {total_frames})")
                print(f"💡 Note: This is preview only. Full template will be used during processing.")
                
                # Test green screen detection on first frame
                first_frame = frames[0]
                mask = create_green_screen_mask(first_frame)
                green_pixels = np.sum(mask > 0)
                
                self.preview_text.config(
                    text=f"🎬 Video Template: {len(frames)} frames (from {total_frames}), Green screen: {green_pixels} pixels"
                )
                
                # Start animated preview
                self.start_animated_preview()
                
            else:
                self.preview_text.config(text="❌ Error: Could not extract preview frames")
                self.is_video_template = False
                
        except MemoryError as e:
            self.preview_text.config(text="❌ Error: Video too large for memory")
            self.is_video_template = False
            print(f"Memory error loading video template: {e}")
            
        except cv2.error as e:
            self.preview_text.config(text="❌ Error: OpenCV memory issue")
            self.is_video_template = False
            print(f"OpenCV error loading video template: {e}")
            
        except Exception as e:
            self.preview_text.config(text=f"❌ Error loading video: {str(e)}")
            self.is_video_template = False
            print(f"Error loading video template: {e}")
    
    def load_image_template(self, image_path):
        """Load static image template."""
        try:
            img = cv2.imread(image_path)
            if img is not None:
                img_resized = cv2.resize(img, (200, 150))
                mask = create_green_screen_mask(img_resized, "medium")
                green_pixels = np.sum(mask > 0)
                
                # Provide quality feedback
                total_pixels = img_resized.shape[0] * img_resized.shape[1]
                coverage = (green_pixels / total_pixels) * 100
                
                if coverage > 15:
                    quality = "Excellent"
                    emoji = "✅"
                elif coverage > 8:
                    quality = "Good"
                    emoji = "✅"
                elif coverage > 3:
                    quality = "Fair"
                    emoji = "⚠️"
                else:
                    quality = "Poor"
                    emoji = "❌"
                
                self.preview_text.config(
                    text=f"🖼️ Static Template: {emoji} {quality} quality - {green_pixels} pixels ({coverage:.1f}% coverage)"
                )
            else:
                self.preview_text.config(text="❌ Error: Could not load image")
        except Exception as e:
            self.preview_text.config(text=f"❌ Error loading image: {str(e)}")
            print(f"Error loading image template: {e}")
    
    def get_current_template_frame(self):
        """Get current template frame (for video) or static image."""
        if self.is_video_template and self.video_frames:
            return self.video_frames[self.current_frame_index]
        elif not self.is_video_template and self.background_image_path:
            img = cv2.imread(self.background_image_path)
            return img
        return None
    
    def advance_video_frame(self):
        """Advance to next video frame (for animated preview)."""
        if self.is_video_template and self.video_frames:
            self.current_frame_index = (self.current_frame_index + 1) % len(self.video_frames)
    
    def start_animated_preview(self):
        """Start animated preview for video templates."""
        if self.is_video_template and len(self.video_frames) > 1:
            # Stop any existing animation first
            self.stop_animation()
            # Start new animation
            self.animate_preview()
    
    def stop_animation(self):
        """Stop animated preview."""
        if self.animation_job:
            self.template_frame.after_cancel(self.animation_job)
            self.animation_job = None
    
    def animate_preview(self):
        """Animate the preview for video templates."""
        if not self.is_video_template or not self.video_frames:
            return
        
        try:
            # Get current text settings from parent for animated preview
            # Try to get text settings from the text section if available
            text_settings = {'enabled': False}  # Default fallback
            
            # Try to get actual text settings from parent GUI
            try:
                # Navigate up to find the main GUI manager
                parent_widget = self.parent_frame
                while parent_widget and not hasattr(parent_widget, 'master'):
                    parent_widget = getattr(parent_widget, 'parent', None)
                    if parent_widget is None:
                        break
                
                # Look for text section in the GUI manager
                if hasattr(parent_widget, 'master') and hasattr(parent_widget.master, 'text_section'):
                    text_settings = parent_widget.master.text_section.get_text_settings()
                elif hasattr(self.parent_frame, 'master') and hasattr(self.parent_frame.master, 'text_section'):
                    text_settings = self.parent_frame.master.text_section.get_text_settings()
            except:
                # If we can't get text settings, use default
                pass
            
            # Update preview with current frame and text settings
            self.update_preview_frame(text_settings)
            
            # Advance to next frame
            self.advance_video_frame()
            
            # Schedule next frame
            current_duration = self.video_durations[self.current_frame_index] if self.video_durations else 100
            frame_delay = max(100, current_duration)  # Minimum 100ms for preview
            
            self.animation_job = self.template_frame.after(frame_delay, self.animate_preview)
            
        except Exception as e:
            print(f"Animation error: {e}")
            self.stop_animation()
    
    def update_preview(self, text_settings):
        """Update preview with current settings."""
        if not self.background_image_path:
            return
        
        # Update preview for both static and video templates
        if self.is_video_template:
            # For video templates, update current frame with text
            self.update_preview_frame(text_settings)
        else:
            # For static templates, update immediately
            self.update_preview_frame(text_settings)
    
    def update_preview_frame(self, text_settings):
        """Update preview frame (used by both static and animated previews)."""
        try:
            # Get current template frame
            img = self.get_current_template_frame()
            if img is None:
                self.preview_label.config(text="❌ Error loading template")
                return
            
            preview_width = 160
            preview_height = 285
            
            h, w = img.shape[:2]
            target_ratio = preview_width / preview_height
            current_ratio = w / h
            
            # Crop to fit aspect ratio
            if current_ratio > target_ratio:
                new_width = int(h * target_ratio)
                start_x = (w - new_width) // 2
                img = img[:, start_x:start_x + new_width]
            else:
                new_height = int(w / target_ratio)
                start_y = (h - new_height) // 2
                img = img[start_y:start_y + new_height, :]
            
            img = cv2.resize(img, (preview_width, preview_height))
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Draw video area (green screen detection)
            mask = create_green_screen_mask(img, "medium")
            if np.sum(mask) > 0:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    if w > 5 and h > 5:
                        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
                        try:
                            small_font = ImageFont.truetype("arial.ttf", 10)
                        except:
                            small_font = ImageFont.load_default()
                        
                        text_bbox = draw.textbbox((0, 0), "VIDEO", font=small_font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        text_x = x + (w - text_width) // 2
                        text_y = y + (h - text_height) // 2
                        draw.text((text_x, text_y), "VIDEO", fill="red", font=small_font)
            
            # Add text overlay preview
            if text_settings and text_settings['enabled']:
                try:
                    preview_font_size = max(8, int(text_settings['size'] * 0.15))
                    
                    try:
                        font = ImageFont.truetype("arial.ttf", preview_font_size)
                    except:
                        font = ImageFont.load_default()
                except:
                    font = ImageFont.load_default()
                
                sample_text = f"Sample Text ({text_settings['font']})"
                max_width = pil_img.width - 10
                lines = smart_text_wrap(sample_text, draw, font, max_width, emoji_size=15)
                
                line_height = preview_font_size + 3
                total_height = len(lines) * line_height
                base_y = int((text_settings['y_position'] / 100) * (pil_img.height - total_height - 10))
                base_y = max(5, min(base_y, pil_img.height - total_height - 5))
                
                rendered_lines = render_text_with_emoji_multiline(
                    draw, lines, font, pil_img.width, pil_img.height, 
                    base_y, emoji_size=15, line_spacing=3
                )
                
                # Get text color from settings (default to black if not specified)
                text_color = text_settings.get('color', '#000000')
                
                for line_data in rendered_lines:
                    for item_type, item, x_offset in line_data['items']:
                        if item_type == 'emoji':
                            emoji_y = line_data['y'] + (preview_font_size - line_data['emoji_size']) // 2
                            pil_img.paste(item, (line_data['x_start'] + x_offset, emoji_y), item)
                        elif item_type == 'text':
                            draw.text((line_data['x_start'] + x_offset, line_data['y']), 
                                     item, fill=text_color, font=font)
            
            # Add template type indicator
            try:
                indicator_font = ImageFont.truetype("arial.ttf", 8)
            except:
                indicator_font = ImageFont.load_default()
            
            if self.is_video_template:
                frame_text = f"🎬 Video Frame {self.current_frame_index + 1}/{len(self.video_frames)}"
                draw.text((5, preview_height - 15), frame_text, fill="blue", font=indicator_font)
            else:
                draw.text((5, preview_height - 15), "🖼️ Static Image", fill="green", font=indicator_font)
            
            photo = ImageTk.PhotoImage(pil_img)
            self.preview_label.config(image=photo, text="", width=preview_width, height=preview_height)
            self.preview_label.image = photo
            
        except Exception as e:
            print(f"Preview error: {e}")
            self.preview_label.config(text=f"❌ Error: {str(e)}")
    
    def get_template_info(self):
        """Get template information for processing."""
        return {
            'path': self.background_image_path,
            'is_video': self.is_video_template,
            'frames': self.video_frames if self.is_video_template else None,
            'durations': self.video_durations if self.is_video_template else None,
            'frame_count': len(self.video_frames) if self.is_video_template else 1
        }
    
    def pack_forget(self):
        """Hide template section."""
        self.stop_animation()
        if hasattr(self, 'template_frame'):
            self.template_frame.pack_forget()
    
    def pack(self, **kwargs):
        """Show template section."""
        if hasattr(self, 'template_frame'):
            self.template_frame.pack(**kwargs)
            # Restart animation if it's a video and ensure text preview works
            if self.is_video_template and len(self.video_frames) > 1:
                self.start_animated_preview()
            elif self.is_video_template:
                # Single frame video or static preview
                try:
                    # Get current text settings for initial preview
                    text_settings = {'enabled': False}
                    self.update_preview_frame(text_settings)
                except:
                    pass
    
    def update_mode_info(self, mode):
        """Update template section info based on processing mode."""
        if mode == "greenscreen":
            info_text = "🎬 Video Template → MP4 with Audio | 🖼️ Static Template → MP4 with Audio"
        elif mode == "blur":
            info_text = "Create blurred background with cropped video overlay in 9:16 aspect ratio"
        elif mode == "narasi":
            info_text = "Template for narasi mode processing with concatenated videos"
        else:
            info_text = "Select template for processing"
        
        # Update info label if it exists
        for widget in self.template_frame.winfo_children():
            if isinstance(widget, tk.Label) and "Create blurred background" in widget.cget("text"):
                widget.config(text=info_text)
                break