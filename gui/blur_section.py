import tkinter as tk
from tkinter import filedialog
import os
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

class BlurSection:
    """Blur mode section of the GUI."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.create_blur_section()
    
    def create_blur_section(self):
        """Create blur mode section."""
        self.blur_frame = tk.LabelFrame(
            self.parent_frame, 
            text="🌀 Blur Background Settings", 
            font=("Arial", 11, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50", 
            padx=10, 
            pady=8
        )
        self.blur_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Info
        info_label = tk.Label(
            self.blur_frame, 
            text="Create blurred background with cropped video overlay in 9:16 aspect ratio",
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0", 
            wraplength=600
        )
        info_label.pack(pady=5)
        
        # Crop settings
        crop_frame = tk.Frame(self.blur_frame, bg="#f0f0f0")
        crop_frame.pack(pady=8, fill=tk.X)
        
        # Top crop
        top_crop_frame = tk.Frame(crop_frame, bg="#f0f0f0")
        top_crop_frame.pack(pady=3, fill=tk.X)
        
        tk.Label(top_crop_frame, text="✂️ Crop Top:", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        self.crop_top = tk.IntVar(value=0)
        top_scale = tk.Scale(
            top_crop_frame, 
            from_=0, 
            to=30, 
            orient=tk.HORIZONTAL, 
            variable=self.crop_top, 
            bg="#f0f0f0",
            length=200
        )
        top_scale.pack(side=tk.LEFT, padx=10)
        
        top_percent = tk.Label(top_crop_frame, textvariable=self.crop_top, font=("Arial", 10), bg="#f0f0f0")
        top_percent.pack(side=tk.LEFT)
        tk.Label(top_crop_frame, text="%", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        # Bottom crop
        bottom_crop_frame = tk.Frame(crop_frame, bg="#f0f0f0")
        bottom_crop_frame.pack(pady=3, fill=tk.X)
        
        tk.Label(bottom_crop_frame, text="✂️ Crop Bottom:", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        self.crop_bottom = tk.IntVar(value=0)
        bottom_scale = tk.Scale(
            bottom_crop_frame, 
            from_=0, 
            to=30, 
            orient=tk.HORIZONTAL, 
            variable=self.crop_bottom, 
            bg="#f0f0f0",
            length=200
        )
        bottom_scale.pack(side=tk.LEFT, padx=10)
        
        bottom_percent = tk.Label(bottom_crop_frame, textvariable=self.crop_bottom, font=("Arial", 10), bg="#f0f0f0")
        bottom_percent.pack(side=tk.LEFT)
        tk.Label(bottom_crop_frame, text="%", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        # Video position settings
        position_frame = tk.Frame(self.blur_frame, bg="#f0f0f0")
        position_frame.pack(pady=8, fill=tk.X)
        
        # X Position
        x_pos_frame = tk.Frame(position_frame, bg="#f0f0f0")
        x_pos_frame.pack(pady=3, fill=tk.X)
        
        tk.Label(x_pos_frame, text="↔️ Video X Position:", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        self.video_x_position = tk.IntVar(value=50)
        x_scale = tk.Scale(
            x_pos_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            variable=self.video_x_position, 
            bg="#f0f0f0",
            length=200
        )
        x_scale.pack(side=tk.LEFT, padx=10)
        
        x_percent = tk.Label(x_pos_frame, textvariable=self.video_x_position, font=("Arial", 10), bg="#f0f0f0")
        x_percent.pack(side=tk.LEFT)
        tk.Label(x_pos_frame, text="%", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        # Y Position
        y_pos_frame = tk.Frame(position_frame, bg="#f0f0f0")
        y_pos_frame.pack(pady=3, fill=tk.X)
        
        tk.Label(y_pos_frame, text="↕️ Video Y Position:", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        self.video_y_position = tk.IntVar(value=50)
        y_scale = tk.Scale(
            y_pos_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            variable=self.video_y_position, 
            bg="#f0f0f0",
            length=200
        )
        y_scale.pack(side=tk.LEFT, padx=10)
        
        y_percent = tk.Label(y_pos_frame, textvariable=self.video_y_position, font=("Arial", 10), bg="#f0f0f0")
        y_percent.pack(side=tk.LEFT)
        tk.Label(y_pos_frame, text="%", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        # Sample video selection for preview
        self.create_sample_video_section()
        
        # Visual preview
        self.create_visual_preview()
    
    def create_sample_video_section(self):
        """Create sample video selection for preview."""
        sample_frame = tk.Frame(self.blur_frame, bg="#f0f0f0")
        sample_frame.pack(pady=8, fill=tk.X)
        
        sample_title = tk.Label(
            sample_frame, 
            text="📹 Sample Video for Preview (Optional)", 
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50"
        )
        sample_title.pack(pady=(0, 5))
        
        self.sample_video_label = tk.Label(
            sample_frame, 
            text="No sample video selected - using default preview", 
            font=("Arial", 9), 
            bg="#f0f0f0", 
            fg="#7f8c8d"
        )
        self.sample_video_label.pack(pady=3)
        
        select_sample_btn = tk.Button(
            sample_frame, 
            text="📁 Select Sample Video", 
            command=self.select_sample_video, 
            font=("Arial", 9), 
            bg="#3498db", 
            fg="white", 
            activebackground="#2980b9"
        )
        select_sample_btn.pack(pady=5)
        
        self.sample_video_path = ""
    
    def create_visual_preview(self):
        """Create visual preview area."""
        preview_frame = tk.Frame(self.blur_frame, bg="#e0e0e0", relief=tk.SUNKEN, bd=1)
        preview_frame.pack(pady=8)
        
        preview_title = tk.Label(
            preview_frame, 
            text="📱 Blur Background Preview (9:16)", 
            font=("Arial", 10, "bold"), 
            bg="#e0e0e0"
        )
        preview_title.pack(pady=5)
        
        self.preview_label = tk.Label(
            preview_frame, 
            text="Select sample video to see preview\n\n"
                 "🌀 Blur Background Mode:\n"
                 "• Blurred background\n"
                 "• Cropped video overlay\n"
                 "• Adjustable position", 
            bg="#ffffff", 
            fg="#2c3e50", 
            width=22, 
            height=14,
            relief=tk.SUNKEN, 
            bd=1
        )
        self.preview_label.pack(pady=8, padx=8)
        
        # Bind settings changes to update preview
        self.crop_top.trace('w', self.update_blur_preview)
        self.crop_bottom.trace('w', self.update_blur_preview)
        self.video_x_position.trace('w', self.update_blur_preview)
        self.video_y_position.trace('w', self.update_blur_preview)
    
    def select_sample_video(self):
        """Select sample video for preview."""
        video_path = filedialog.askopenfilename(
            title="Select Sample Video for Preview",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All Files", "*.*")
            ]
        )
        
        if video_path:
            self.sample_video_path = video_path
            filename = os.path.basename(video_path)
            self.sample_video_label.config(text=f"Sample: {filename}")
            self.update_blur_preview()
    
    def update_blur_preview(self, *args, text_settings=None):
        """Update blur preview with visual representation and text position."""
        try:
            if self.sample_video_path and os.path.exists(self.sample_video_path):
                self.create_video_preview(text_settings)
            else:
                self.create_default_preview(text_settings)
        except Exception as e:
            print(f"Preview error: {e}")
            self.create_default_preview(text_settings)
    
    def create_video_preview(self, text_settings=None):
        """Create preview using actual video frame with text position overlay."""
        try:
            # Read first frame from video
            cap = cv2.VideoCapture(self.sample_video_path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                self.create_default_preview(text_settings)
                return
            
            # Process frame with current blur settings
            from utils.blur_processing import process_blur_frame
            
            processed_frame = process_blur_frame(
                frame,
                self.crop_top.get(),
                self.crop_bottom.get(),
                self.video_x_position.get(),
                self.video_y_position.get(),
                1080,  # target_width
                1920   # target_height
            )
            
            # Resize for preview
            preview_width = 160
            preview_height = 285
            preview_frame = cv2.resize(processed_frame, (preview_width, preview_height))
            
            # Convert to PIL and display
            pil_img = Image.fromarray(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB))
            
            # Add text position preview if text is enabled
            if text_settings and text_settings.get('enabled', False):
                draw = ImageDraw.Draw(pil_img)
                self.add_text_position_preview(draw, pil_img, text_settings, preview_width, preview_height)
            
            # Add settings overlay
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("arial.ttf", 8)
            except:
                font = ImageFont.load_default()
            
            # Add settings text
            settings_text = (
                f"Crop: {self.crop_top.get()}%/{self.crop_bottom.get()}%\n"
                f"Pos: {self.video_x_position.get()}%,{self.video_y_position.get()}%"
            )
            
            # Draw semi-transparent background for text
            text_bbox = draw.textbbox((0, 0), settings_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([5, 5, text_width + 15, text_height + 15], fill=(0, 0, 0, 128))
            
            pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(pil_img)
            draw.text((10, 10), settings_text, fill="white", font=font)
            
            # Display preview
            photo = ImageTk.PhotoImage(pil_img)
            self.preview_label.config(image=photo, text="", width=preview_width, height=preview_height)
            self.preview_label.image = photo
            
        except Exception as e:
            print(f"Video preview error: {e}")
            self.create_default_preview(text_settings)
    
    def create_default_preview(self, text_settings=None):
        """Create default preview when no video is selected with text position overlay."""
        try:
            # Create a simple visual representation
            preview_width = 160
            preview_height = 285
            
            # Create blank image
            img = Image.new('RGB', (preview_width, preview_height), color='#f0f0f0')
            draw = ImageDraw.Draw(img)
            
            # Draw background area (blurred representation)
            draw.rectangle([0, 0, preview_width, preview_height], fill='#e0e0e0', outline='#bdc3c7')
            
            # Calculate video area based on settings
            crop_top_px = int(preview_height * self.crop_top.get() / 100)
            crop_bottom_px = int(preview_height * self.crop_bottom.get() / 100)
            
            video_height = preview_height - crop_top_px - crop_bottom_px
            video_width = int(video_height * (16/9))  # Maintain aspect ratio
            
            if video_width > preview_width:
                video_width = preview_width
                video_height = int(video_width * (9/16))
            
            # Calculate position
            max_x = preview_width - video_width
            max_y = preview_height - video_height
            
            x_offset = int((self.video_x_position.get() / 100) * max_x) if max_x > 0 else 0
            y_offset = int((self.video_y_position.get() / 100) * max_y) if max_y > 0 else 0
            
            # Draw video area
            video_rect = [x_offset, y_offset, x_offset + video_width, y_offset + video_height]
            draw.rectangle(video_rect, fill='#3498db', outline='#2980b9', width=2)
            
            # Add text position preview if text is enabled
            if text_settings and text_settings.get('enabled', False):
                self.add_text_position_preview(draw, img, text_settings, preview_width, preview_height)
            
            # Add labels
            try:
                font = ImageFont.truetype("arial.ttf", 8)
            except:
                font = ImageFont.load_default()
            
            # Background label
            draw.text((5, 5), "Blurred BG", fill='#7f8c8d', font=font)
            
            # Video area label
            if video_width > 40 and video_height > 20:
                video_center_x = x_offset + video_width // 2
                video_center_y = y_offset + video_height // 2
                
                video_text = "VIDEO"
                text_bbox = draw.textbbox((0, 0), video_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                text_x = video_center_x - text_width // 2
                text_y = video_center_y - text_height // 2
                
                draw.text((text_x, text_y), video_text, fill='white', font=font)
            
            # Settings info
            settings_text = (
                f"Crop: T{self.crop_top.get()}% B{self.crop_bottom.get()}%\n"
                f"Pos: X{self.video_x_position.get()}% Y{self.video_y_position.get()}%"
            )
            
            # Draw settings at bottom
            draw.text((5, preview_height - 25), settings_text, fill='#2c3e50', font=font)
            
            # Display preview
            photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=photo, text="", width=preview_width, height=preview_height)
            self.preview_label.image = photo
            
        except Exception as e:
            print(f"Default preview error: {e}")
            # Fallback to text preview
            crop_top = self.crop_top.get()
            crop_bottom = self.crop_bottom.get()
            x_pos = self.video_x_position.get()
            y_pos = self.video_y_position.get()
            
            preview_text = (
                f"🌀 Blur Background Mode\n\n"
                f"• Crop Top: {crop_top}%\n"
                f"• Crop Bottom: {crop_bottom}%\n"
                f"• Video X Position: {x_pos}%\n"
                f"• Video Y Position: {y_pos}%\n"
                f"• Output: 9:16 aspect ratio"
            )
            
            self.preview_label.config(text=preview_text)
    
    def add_text_position_preview(self, draw, img, text_settings, preview_width, preview_height):
        """Add text position overlay to preview."""
        try:
            # Calculate text position based on settings
            x_percent = text_settings.get('x_position', 50) / 100
            y_percent = text_settings.get('y_position', 80) / 100
            
            # Calculate preview text size (scaled down)
            preview_font_size = max(6, int(text_settings.get('size', 60) * 0.1))
            
            try:
                font = ImageFont.truetype("arial.ttf", preview_font_size)
            except:
                font = ImageFont.load_default()
            
            # Sample text
            sample_text = "Sample Text"
            text_bbox = draw.textbbox((0, 0), sample_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Calculate position
            text_x = int(x_percent * (preview_width - text_width))
            text_y = int(y_percent * (preview_height - text_height))
            
            # Ensure text stays within bounds
            text_x = max(2, min(text_x, preview_width - text_width - 2))
            text_y = max(2, min(text_y, preview_height - text_height - 2))
            
            # Draw text background (semi-transparent)
            text_bg_rect = [
                text_x - 2, text_y - 1, 
                text_x + text_width + 2, text_y + text_height + 1
            ]
            
            # Create overlay for text background
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(text_bg_rect, fill=(255, 255, 0, 128))  # Yellow with transparency
            
            # Composite overlay
            img_rgba = img.convert('RGBA')
            img_with_overlay = Image.alpha_composite(img_rgba, overlay)
            
            # Update the original image
            img_updated = img_with_overlay.convert('RGB')
            draw = ImageDraw.Draw(img_updated)
            
            # Draw text
            text_color = text_settings.get('color', '#000000')
            draw.text((text_x, text_y), sample_text, fill=text_color, font=font)
            
            # Draw text position indicator
            draw.rectangle([text_x - 1, text_y - 1, text_x + text_width + 1, text_y + text_height + 1], 
                         outline='red', width=1)
            
            # Copy back to original image
            img.paste(img_updated)
            
        except Exception as e:
            print(f"Text position preview error: {e}")
    
    def get_blur_settings(self):
        """Get blur settings."""
        return {
            'crop_top': self.crop_top.get(),
            'crop_bottom': self.crop_bottom.get(),
            'video_x_position': self.video_x_position.get(),
            'video_y_position': self.video_y_position.get()
        }
    
    def pack_forget(self):
        """Hide blur section."""
        if hasattr(self, 'blur_frame'):
            self.blur_frame.pack_forget()
    
    def pack(self, **kwargs):
        """Show blur section."""
        if hasattr(self, 'blur_frame'):
            self.blur_frame.pack(**kwargs)