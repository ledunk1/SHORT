import tkinter as tk
from tkinter import filedialog
import os
from utils.file_operations import get_all_media_files

class VideoSelection:
    """Video selection section of the GUI."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.folder_path = ""
        self.create_video_selection()
    
    def create_video_selection(self):
        """Create video selection section."""
        self.video_frame = tk.LabelFrame(
            self.parent_frame, 
            text="📁 Media Selection (Video/GIF/Image)", 
            font=("Arial", 11, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50", 
            padx=10, 
            pady=8
        )
        self.video_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Folder selection
        self.folder_label = tk.Label(
            self.video_frame, 
            text="No folder selected", 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            fg="#7f8c8d"
        )
        self.folder_label.pack(pady=5)
        
        select_btn = tk.Button(
            self.video_frame, 
            text="📂 Select Media Folder (Video/GIF/Image)", 
            command=self.select_folder, 
            font=("Arial", 10, "bold"), 
            bg="#3498db", 
            fg="white", 
            activebackground="#2980b9"
        )
        select_btn.pack(pady=5)
        
        self.info_label = tk.Label(
            self.video_frame, 
            text="", 
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0"
        )
        self.info_label.pack(pady=2)
    
    def select_folder(self):
        """Select folder containing videos/GIFs/images."""
        folder_path = filedialog.askdirectory(title="Select Media Folder (Video/GIF/Image)")
        if folder_path:
            self.folder_path = folder_path
            folder_name = os.path.basename(folder_path)
            self.folder_label.config(text=f"Folder: {folder_name}")
            
            # Count files
            try:
                media_files = get_all_media_files(folder_path)
                # FIXED: Separate video and image counting properly
                video_count = sum(1 for f in media_files if self.is_video_file(f))
                gif_count = sum(1 for f in media_files if f.lower().endswith('.gif'))
                image_count = sum(1 for f in media_files if self.is_image_file(f))
                
                self.info_label.config(
                    text=f"Found: {video_count} videos, {gif_count} GIFs, {image_count} images"
                )
            except Exception as e:
                self.info_label.config(text=f"Error scanning folder: {str(e)}")
    
    def is_video_file(self, file_name):
        """Check if file is a video."""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
        return file_name.lower().endswith(video_extensions)
    
    def is_image_file(self, file_name):
        """Check if file is an image."""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
        return file_name.lower().endswith(image_extensions)
    
    def get_folder_path(self):
        """Get selected folder path."""
        return self.folder_path
    
    def pack_forget(self):
        """Hide video selection section."""
        if hasattr(self, 'video_frame'):
            self.video_frame.pack_forget()
    
    def pack(self, **kwargs):
        """Show video selection section."""
        if hasattr(self, 'video_frame'):
            self.video_frame.pack(**kwargs)