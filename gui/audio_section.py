import tkinter as tk
from tkinter import filedialog
import os
from utils.file_operations import get_audio_files

class AudioSection:
    """Audio section of the GUI."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.audio_folder_path = ""
        self.create_audio_section()
    
    def create_audio_section(self):
        """Create audio section."""
        self.audio_frame = tk.LabelFrame(
            self.parent_frame, 
            text="🎵 Audio Settings", 
            font=("Arial", 11, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50", 
            padx=10, 
            pady=8
        )
        self.audio_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Audio enable checkbox
        self.audio_enabled = tk.BooleanVar()
        audio_checkbox = tk.Checkbutton(
            self.audio_frame, 
            text="🎵 Enable Background Music", 
            variable=self.audio_enabled, 
            font=("Arial", 10), 
            bg="#f0f0f0",
            command=self.on_audio_enable_change
        )
        audio_checkbox.pack(pady=5)
        
        # Audio folder selection
        self.audio_settings_frame = tk.Frame(self.audio_frame, bg="#f0f0f0")
        self.audio_settings_frame.pack(pady=8, fill=tk.X)
        
        self.audio_folder_label = tk.Label(
            self.audio_settings_frame, 
            text="No audio folder selected", 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            fg="#7f8c8d"
        )
        self.audio_folder_label.pack(pady=3)
        
        select_audio_btn = tk.Button(
            self.audio_settings_frame, 
            text="🎵 Select Audio Folder", 
            command=self.select_audio_folder, 
            font=("Arial", 10, "bold"), 
            bg="#e67e22", 
            fg="white", 
            activebackground="#d35400"
        )
        select_audio_btn.pack(pady=5)
        
        self.audio_info_label = tk.Label(
            self.audio_settings_frame, 
            text="", 
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0"
        )
        self.audio_info_label.pack(pady=2)
        
        # Volume control
        volume_frame = tk.Frame(self.audio_settings_frame, bg="#f0f0f0")
        volume_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(volume_frame, text="🔊 Volume:", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        self.volume = tk.IntVar(value=50)
        volume_scale = tk.Scale(
            volume_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            variable=self.volume, 
            bg="#f0f0f0",
            length=200
        )
        volume_scale.pack(side=tk.LEFT, padx=10)
        
        volume_percent = tk.Label(volume_frame, textvariable=self.volume, font=("Arial", 10), bg="#f0f0f0")
        volume_percent.pack(side=tk.LEFT)
        
        percent_label = tk.Label(volume_frame, text="%", font=("Arial", 10), bg="#f0f0f0")
        percent_label.pack(side=tk.LEFT)
        
        # Initially disable audio settings
        self.update_audio_settings_state()
    
    def select_audio_folder(self):
        """Select audio folder."""
        folder_path = filedialog.askdirectory(title="Select Audio Folder")
        if folder_path:
            self.audio_folder_path = folder_path
            folder_name = os.path.basename(folder_path)
            self.audio_folder_label.config(text=f"Audio Folder: {folder_name}")
            
            # Count audio files
            try:
                audio_files = get_audio_files(folder_path)
                self.audio_info_label.config(
                    text=f"Found: {len(audio_files)} audio files"
                )
            except Exception as e:
                self.audio_info_label.config(text=f"Error scanning folder: {str(e)}")
    
    def on_audio_enable_change(self):
        """Handle audio enable change."""
        self.update_audio_settings_state()
    
    def update_audio_settings_state(self):
        """Update the state of audio settings based on checkbox."""
        state = tk.NORMAL if self.audio_enabled.get() else tk.DISABLED
        
        # Update all child widgets in audio settings frame
        for widget in self.audio_settings_frame.winfo_children():
            self.update_widget_state(widget, state)
    
    def update_widget_state(self, widget, state):
        """Recursively update widget state."""
        try:
            if hasattr(widget, 'config'):
                widget.config(state=state)
        except:
            pass
        
        # Update children
        for child in widget.winfo_children():
            self.update_widget_state(child, state)
    
    def get_audio_settings(self):
        """Get audio settings."""
        return {
            'enabled': self.audio_enabled.get(),
            'folder_path': self.audio_folder_path,
            'volume': self.volume.get()
        }
    
    def pack_forget(self):
        """Hide audio section."""
        if hasattr(self, 'audio_frame'):
            self.audio_frame.pack_forget()
    
    def pack(self, **kwargs):
        """Show audio section."""
        if hasattr(self, 'audio_frame'):
            self.audio_frame.pack(**kwargs)