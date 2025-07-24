import tkinter as tk
from tkinter import filedialog
import os
from utils.file_operations import get_audio_files

class EnhancedAudioSection:
    """Enhanced audio section for green screen and blur modes with multiple audio options."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.audio_folder_path = ""
        self.create_enhanced_audio_section()
    
    def create_enhanced_audio_section(self):
        """Create enhanced audio section with multiple options."""
        self.enhanced_audio_frame = tk.LabelFrame(
            self.parent_frame, 
            text="🎵 Enhanced Audio Settings", 
            font=("Arial", 11, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50", 
            padx=10, 
            pady=8
        )
        self.enhanced_audio_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Audio mode selection
        audio_mode_frame = tk.Frame(self.enhanced_audio_frame, bg="#f0f0f0")
        audio_mode_frame.pack(pady=8, fill=tk.X)
        
        audio_mode_title = tk.Label(
            audio_mode_frame, 
            text="🎵 Audio Mode Selection", 
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50"
        )
        audio_mode_title.pack(pady=(0, 5))
        
        # Audio mode radio buttons
        self.audio_mode_var = tk.StringVar(value="original_only")
        
        # Original audio only
        original_only_radio = tk.Radiobutton(
            audio_mode_frame, 
            text="🎤 Original Audio Only", 
            variable=self.audio_mode_var, 
            value="original_only",
            font=("Arial", 9), 
            bg="#f0f0f0",
            activebackground="#f0f0f0",
            command=self.on_audio_mode_change
        )
        original_only_radio.pack(anchor=tk.W, padx=10, pady=2)
        
        # Background music only
        background_only_radio = tk.Radiobutton(
            audio_mode_frame, 
            text="🎵 Background Music Only", 
            variable=self.audio_mode_var, 
            value="background_only",
            font=("Arial", 9), 
            bg="#f0f0f0",
            activebackground="#f0f0f0",
            command=self.on_audio_mode_change
        )
        background_only_radio.pack(anchor=tk.W, padx=10, pady=2)
        
        # Dual audio mixing
        dual_audio_radio = tk.Radiobutton(
            audio_mode_frame, 
            text="🎭 Dual Audio Mixing (Original + Background)", 
            variable=self.audio_mode_var, 
            value="dual_mixing",
            font=("Arial", 9), 
            bg="#f0f0f0",
            activebackground="#f0f0f0",
            command=self.on_audio_mode_change
        )
        dual_audio_radio.pack(anchor=tk.W, padx=10, pady=2)
        
        # Audio folder selection (for background music and dual mixing)
        self.audio_folder_section = tk.Frame(self.enhanced_audio_frame, bg="#f0f0f0")
        self.audio_folder_section.pack(pady=8, fill=tk.X)
        
        audio_folder_title = tk.Label(
            self.audio_folder_section, 
            text="🎵 Background Music Folder", 
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50"
        )
        audio_folder_title.pack(pady=(0, 5))
        
        self.audio_folder_label = tk.Label(
            self.audio_folder_section, 
            text="No audio folder selected", 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            fg="#7f8c8d"
        )
        self.audio_folder_label.pack(pady=3)
        
        select_audio_folder_btn = tk.Button(
            self.audio_folder_section, 
            text="🎵 Select Audio Folder", 
            command=self.select_audio_folder, 
            font=("Arial", 10, "bold"), 
            bg="#e67e22", 
            fg="white", 
            activebackground="#d35400"
        )
        select_audio_folder_btn.pack(pady=5)
        
        self.audio_folder_info_label = tk.Label(
            self.audio_folder_section, 
            text="", 
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0"
        )
        self.audio_folder_info_label.pack(pady=2)
        
        # Volume controls
        self.create_volume_controls()
        
        # Initially update state based on default selection
        self.on_audio_mode_change()
    
    def create_volume_controls(self):
        """Create volume control sliders."""
        self.volume_controls_frame = tk.Frame(self.enhanced_audio_frame, bg="#f0f0f0")
        self.volume_controls_frame.pack(pady=8, fill=tk.X)
        
        # Original audio volume (for dual mixing mode)
        self.original_volume_frame = tk.Frame(self.volume_controls_frame, bg="#f0f0f0")
        self.original_volume_frame.pack(pady=5, fill=tk.X)
        
        original_volume_label = tk.Label(
            self.original_volume_frame, 
            text="🎤 Original Audio Volume:", 
            font=("Arial", 10), 
            bg="#f0f0f0"
        )
        original_volume_label.pack(side=tk.LEFT)
        
        self.original_volume = tk.IntVar(value=100)
        original_volume_scale = tk.Scale(
            self.original_volume_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            variable=self.original_volume, 
            bg="#f0f0f0",
            length=200
        )
        original_volume_scale.pack(side=tk.LEFT, padx=10)
        
        original_volume_percent = tk.Label(
            self.original_volume_frame, 
            textvariable=self.original_volume, 
            font=("Arial", 10), 
            bg="#f0f0f0"
        )
        original_volume_percent.pack(side=tk.LEFT)
        
        original_percent_label = tk.Label(
            self.original_volume_frame, 
            text="%", 
            font=("Arial", 10), 
            bg="#f0f0f0"
        )
        original_percent_label.pack(side=tk.LEFT)
        
        # Background audio volume
        self.background_volume_frame = tk.Frame(self.volume_controls_frame, bg="#f0f0f0")
        self.background_volume_frame.pack(pady=5, fill=tk.X)
        
        background_volume_label = tk.Label(
            self.background_volume_frame, 
            text="🎵 Background Music Volume:", 
            font=("Arial", 10), 
            bg="#f0f0f0"
        )
        background_volume_label.pack(side=tk.LEFT)
        
        self.background_volume = tk.IntVar(value=50)
        background_volume_scale = tk.Scale(
            self.background_volume_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            variable=self.background_volume, 
            bg="#f0f0f0",
            length=200
        )
        background_volume_scale.pack(side=tk.LEFT, padx=10)
        
        background_volume_percent = tk.Label(
            self.background_volume_frame, 
            textvariable=self.background_volume, 
            font=("Arial", 10), 
            bg="#f0f0f0"
        )
        background_volume_percent.pack(side=tk.LEFT)
        
        background_percent_label = tk.Label(
            self.background_volume_frame, 
            text="%", 
            font=("Arial", 10), 
            bg="#f0f0f0"
        )
        background_percent_label.pack(side=tk.LEFT)
    
    def select_audio_folder(self):
        """Select audio folder for background music."""
        folder_path = filedialog.askdirectory(title="Select Audio Folder")
        if folder_path:
            self.audio_folder_path = folder_path
            folder_name = os.path.basename(folder_path)
            self.audio_folder_label.config(text=f"Audio Folder: {folder_name}")
            
            # Count audio files
            try:
                audio_files = get_audio_files(folder_path)
                self.audio_folder_info_label.config(
                    text=f"Found: {len(audio_files)} audio files"
                )
            except Exception as e:
                self.audio_folder_info_label.config(text=f"Error scanning folder: {str(e)}")
    
    def on_audio_mode_change(self):
        """Handle audio mode selection change."""
        audio_mode = self.audio_mode_var.get()
        
        # Show/hide audio folder section based on mode
        if audio_mode == "original_only":
            # Hide audio folder and volume controls
            self.audio_folder_section.pack_forget()
            self.volume_controls_frame.pack_forget()
        elif audio_mode == "background_only":
            # Show audio folder and background volume only
            self.audio_folder_section.pack(pady=8, fill=tk.X)
            self.volume_controls_frame.pack(pady=8, fill=tk.X)
            self.original_volume_frame.pack_forget()
            self.background_volume_frame.pack(pady=5, fill=tk.X)
        elif audio_mode == "dual_mixing":
            # Show audio folder and both volume controls
            self.audio_folder_section.pack(pady=8, fill=tk.X)
            self.volume_controls_frame.pack(pady=8, fill=tk.X)
            self.original_volume_frame.pack(pady=5, fill=tk.X)
            self.background_volume_frame.pack(pady=5, fill=tk.X)
        
        print(f"🎵 Audio mode changed to: {audio_mode}")
    
    def get_enhanced_audio_settings(self):
        """Get enhanced audio settings."""
        audio_mode = self.audio_mode_var.get()
        
        settings = {
            'mode': audio_mode,
            'folder_path': self.audio_folder_path if audio_mode != "original_only" else "",
            'original_volume': self.original_volume.get(),
            'background_volume': self.background_volume.get()
        }
        
        # Convert to format compatible with existing audio processing
        if audio_mode == "original_only":
            settings.update({
                'enabled': False,
                'dual_audio_enabled': False
            })
        elif audio_mode == "background_only":
            settings.update({
                'enabled': True,
                'dual_audio_enabled': False,
                'volume': self.background_volume.get()
            })
        elif audio_mode == "dual_mixing":
            settings.update({
                'enabled': True,
                'dual_audio_enabled': True
            })
        
        return settings
    
    def pack_forget(self):
        """Hide enhanced audio section."""
        if hasattr(self, 'enhanced_audio_frame'):
            self.enhanced_audio_frame.pack_forget()
    
    def pack(self, **kwargs):
        """Show enhanced audio section."""
        if hasattr(self, 'enhanced_audio_frame'):
            self.enhanced_audio_frame.pack(**kwargs)