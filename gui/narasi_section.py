import tkinter as tk
from tkinter import filedialog
import os
from utils.file_operations import get_audio_files, get_all_media_files

class NarasiSection:
    """Narasi mode section of the GUI with bulk processing support."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.narasi_audio_folder_path = ""
        self.create_narasi_section()
    
    def create_narasi_section(self):
        """Create narasi mode section."""
        self.narasi_frame = tk.LabelFrame(
            self.parent_frame, 
            text="🎙️ Narasi Mode Settings", 
            font=("Arial", 11, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50", 
            padx=10, 
            pady=8
        )
        self.narasi_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Info
        info_label = tk.Label(
            self.narasi_frame, 
            text="Bulk process videos with matching audio files. Videos and audio files are matched by filename.",
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0", 
            wraplength=600
        )
        info_label.pack(pady=5)
        
        # Audio Folder Selection (Required)
        audio_section = tk.Frame(self.narasi_frame, bg="#f0f0f0")
        audio_section.pack(pady=8, fill=tk.X)
        
        audio_title = tk.Label(
            audio_section, 
            text="🎵 Audio Folder (Required)", 
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50"
        )
        audio_title.pack(pady=(0, 5))
        
        self.audio_folder_label = tk.Label(
            audio_section, 
            text="No audio folder selected", 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            fg="#7f8c8d"
        )
        self.audio_folder_label.pack(pady=3)
        
        select_audio_btn = tk.Button(
            audio_section, 
            text="📁 Select Audio Folder", 
            command=self.select_audio_folder, 
            font=("Arial", 10, "bold"), 
            bg="#e67e22", 
            fg="white", 
            activebackground="#d35400"
        )
        select_audio_btn.pack(pady=5)
        
        self.audio_info_label = tk.Label(
            audio_section, 
            text="", 
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0"
        )
        self.audio_info_label.pack(pady=2)
        
        # File Matching Preview
        self.matching_section = tk.Frame(self.narasi_frame, bg="#f0f0f0")
        self.matching_section.pack(pady=8, fill=tk.X)
        
        matching_title = tk.Label(
            self.matching_section, 
            text="📋 File Matching Preview", 
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50"
        )
        matching_title.pack(pady=(0, 5))
        
        self.matching_info_label = tk.Label(
            self.matching_section, 
            text="Select video and audio folders to see matching preview", 
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0",
            justify=tk.LEFT,
            wraplength=600
        )
        self.matching_info_label.pack(pady=2)
        
        # Audio Mode Selection
        audio_mode_section = tk.Frame(self.narasi_frame, bg="#f0f0f0")
        audio_mode_section.pack(pady=8, fill=tk.X)
        
        audio_mode_title = tk.Label(
            audio_mode_section, 
            text="🎵 Audio Mode Selection", 
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50"
        )
        audio_mode_title.pack(pady=(0, 5))
        
        # Audio mode radio buttons
        self.audio_mode_var = tk.StringVar(value="narasi_only")
        
        # Narasi audio only
        narasi_only_radio = tk.Radiobutton(
            audio_mode_section, 
            text="🎙️ Narasi Audio Only", 
            variable=self.audio_mode_var, 
            value="narasi_only",
            font=("Arial", 9), 
            bg="#f0f0f0",
            activebackground="#f0f0f0",
            command=self.on_audio_mode_change
        )
        narasi_only_radio.pack(anchor=tk.W, padx=10, pady=2)
        
        # Mixed audio (narasi + original)
        mixed_audio_radio = tk.Radiobutton(
            audio_mode_section, 
            text="🎭 Mix Narasi + Original Video Audio", 
            variable=self.audio_mode_var, 
            value="mixed_audio",
            font=("Arial", 9), 
            bg="#f0f0f0",
            activebackground="#f0f0f0",
            command=self.on_audio_mode_change
        )
        mixed_audio_radio.pack(anchor=tk.W, padx=10, pady=2)
        
        # Volume controls (for mixed audio mode)
        self.volume_controls_frame = tk.Frame(audio_mode_section, bg="#f0f0f0")
        self.volume_controls_frame.pack(pady=8, fill=tk.X)
        
        # Narasi audio volume
        narasi_volume_frame = tk.Frame(self.volume_controls_frame, bg="#f0f0f0")
        narasi_volume_frame.pack(pady=3, fill=tk.X)
        
        tk.Label(
            narasi_volume_frame, 
            text="🎙️ Narasi Volume:", 
            font=("Arial", 10), 
            bg="#f0f0f0"
        ).pack(side=tk.LEFT)
        
        self.narasi_volume = tk.IntVar(value=100)
        narasi_volume_scale = tk.Scale(
            narasi_volume_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            variable=self.narasi_volume, 
            bg="#f0f0f0",
            length=200
        )
        narasi_volume_scale.pack(side=tk.LEFT, padx=10)
        
        narasi_volume_percent = tk.Label(
            narasi_volume_frame, 
            textvariable=self.narasi_volume, 
            font=("Arial", 10), 
            bg="#f0f0f0"
        )
        narasi_volume_percent.pack(side=tk.LEFT)
        
        tk.Label(
            narasi_volume_frame, 
            text="%", 
            font=("Arial", 10), 
            bg="#f0f0f0"
        ).pack(side=tk.LEFT)
        
        # Original video audio volume
        original_volume_frame = tk.Frame(self.volume_controls_frame, bg="#f0f0f0")
        original_volume_frame.pack(pady=3, fill=tk.X)
        
        tk.Label(
            original_volume_frame, 
            text="🎬 Original Video Volume:", 
            font=("Arial", 10), 
            bg="#f0f0f0"
        ).pack(side=tk.LEFT)
        
        self.original_volume = tk.IntVar(value=30)
        original_volume_scale = tk.Scale(
            original_volume_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            variable=self.original_volume, 
            bg="#f0f0f0",
            length=200
        )
        original_volume_scale.pack(side=tk.LEFT, padx=10)
        
        original_volume_percent = tk.Label(
            original_volume_frame, 
            textvariable=self.original_volume, 
            font=("Arial", 10), 
            bg="#f0f0f0"
        )
        original_volume_percent.pack(side=tk.LEFT)
        
        tk.Label(
            original_volume_frame, 
            text="%", 
            font=("Arial", 10), 
            bg="#f0f0f0"
        ).pack(side=tk.LEFT)
        
        # Audio mode description
        self.audio_mode_description = tk.Label(
            audio_mode_section, 
            text="🎙️ Narasi Audio Only: Only the narasi audio will be used in the final video",
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0",
            justify=tk.LEFT,
            wraplength=600
        )
        self.audio_mode_description.pack(pady=(5, 0))
        
        # Initially update audio mode state
        self.on_audio_mode_change()
    
    def on_audio_mode_change(self):
        """Handle audio mode selection change."""
        audio_mode = self.audio_mode_var.get()
        
        if audio_mode == "narasi_only":
            # Hide volume controls
            self.volume_controls_frame.pack_forget()
            self.audio_mode_description.config(
                text="🎙️ Narasi Audio Only: Only the narasi audio will be used in the final video"
            )
        elif audio_mode == "mixed_audio":
            # Show volume controls
            self.volume_controls_frame.pack(pady=8, fill=tk.X)
            self.audio_mode_description.config(
                text="🎭 Mixed Audio: Narasi audio will be mixed with original video audio. Adjust volumes below."
            )
        
        print(f"🎵 Narasi audio mode changed to: {audio_mode}")
    
    def select_audio_folder(self):
        """Select audio folder for narasi mode."""
        audio_folder_path = filedialog.askdirectory(title="Select Audio Folder for Narasi Mode")
        
        if audio_folder_path:
            self.narasi_audio_folder_path = audio_folder_path
            folder_name = os.path.basename(audio_folder_path)
            self.audio_folder_label.config(text=f"Audio Folder: {folder_name}")
            
            # Count audio files and update info
            try:
                audio_files = get_audio_files(audio_folder_path)
                self.audio_info_label.config(
                    text=f"Found: {len(audio_files)} audio files"
                )
                
                # Update matching preview if video folder is also selected
                self.update_matching_preview()
                
            except Exception as e:
                self.audio_info_label.config(text=f"Error scanning audio folder: {str(e)}")
    
    def update_matching_preview(self):
        """Update the file matching preview."""
        if not hasattr(self, 'video_folder_path') or not self.video_folder_path:
            self.matching_info_label.config(text="Select video folder to see matching preview")
            return
        
        if not self.narasi_audio_folder_path:
            self.matching_info_label.config(text="Select audio folder to see matching preview")
            return
        
        try:
            # Get video and audio files
            video_files = get_all_media_files(self.video_folder_path)
            audio_files = get_audio_files(self.narasi_audio_folder_path)
            
            if not video_files:
                self.matching_info_label.config(text="No video files found in video folder")
                return
            
            if not audio_files:
                self.matching_info_label.config(text="No audio files found in audio folder")
                return
            
            # Simple matching info
            self.matching_info_label.config(
                text=f"✅ Ready: {len(video_files)} videos, {len(audio_files)} audio files"
            )
        
        except Exception as e:
            self.matching_info_label.config(text=f"Error creating matching preview: {str(e)}")
    
    def set_video_folder_path(self, video_folder_path):
        """Set video folder path and update matching preview."""
        self.video_folder_path = video_folder_path
        self.update_matching_preview()
    
    def create_file_matches(self, video_files, audio_files):
        """Create matches between video and audio files based on filename."""
        matches = {}
        
        # Get base names (without extensions) for audio files
        audio_base_names = {}
        for audio_file in audio_files:
            base_name = os.path.splitext(audio_file)[0].lower()
            audio_base_names[base_name] = audio_file
        
        # For each audio file, find matching videos
        for base_name, audio_file in audio_base_names.items():
            matched_videos = []
            
            # Find videos that match this audio file name
            for video_file in video_files:
                video_base_name = os.path.splitext(video_file)[0].lower()
                
                # Exact match or partial match
                if (base_name == video_base_name or 
                    base_name in video_base_name or 
                    video_base_name in base_name):
                    matched_videos.append(video_file)
            
            # If no exact matches found, use all videos (fallback)
            if not matched_videos:
                matched_videos = video_files.copy()
            
            matches[audio_file] = matched_videos
        
        return matches
    
    def get_narasi_settings(self):
        """Get narasi settings including audio folder path."""
        return {
            'audio_folder_path': self.narasi_audio_folder_path,
            'has_audio_folder': bool(self.narasi_audio_folder_path),
            'audio_mode': self.audio_mode_var.get(),
            'narasi_volume': self.narasi_volume.get(),
            'original_volume': self.original_volume.get()
        }
    
    def get_narasi_audio_folder_path(self):
        """Get selected audio folder path (for backward compatibility)."""
        return self.narasi_audio_folder_path
    
    def pack_forget(self):
        """Hide narasi section."""
        if hasattr(self, 'narasi_frame'):
            self.narasi_frame.pack_forget()
    
    def pack(self, **kwargs):
        """Show narasi section."""
        if hasattr(self, 'narasi_frame'):
            self.narasi_frame.pack(**kwargs)