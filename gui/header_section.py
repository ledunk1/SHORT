import tkinter as tk
from .base_gui import BaseGUI

class HeaderSection:
    """Header section of the GUI with responsive design."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.create_header()
    
    def create_header(self):
        """Create header section with full width layout."""
        # Main header container
        header_container = tk.Frame(self.parent_frame, bg="#f0f0f0")
        header_container.pack(pady=15, padx=20, fill=tk.X)
        
        # Title section
        title_frame = tk.Frame(header_container, bg="#f0f0f0")
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_frame, 
            text="🎬 Video Editor Enhanced", 
            font=("Arial", 20, "bold"), 
            fg="#2c3e50", 
            bg="#f0f0f0"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame, 
            text="Green Screen & Blur Mode with GPU Acceleration + GIF Support + Narasi Mode",
            font=("Arial", 11, "italic"), 
            fg="#7f8c8d", 
            bg="#f0f0f0"
        )
        subtitle_label.pack(pady=(3, 0))
        
        # Version info
        version_label = tk.Label(
            title_frame, 
            text="v2.0 - Enhanced Edition", 
            font=("Arial", 9), 
            fg="#95a5a6", 
            bg="#f0f0f0"
        )
        version_label.pack(pady=(2, 0))
        
        # Add separator
        separator_frame = tk.Frame(header_container, height=2, bg="#bdc3c7")
        separator_frame.pack(fill=tk.X, pady=(10, 0))
    
    def pack_forget(self):
        """Hide header section (usually not needed)."""
        pass
    
    def pack(self, **kwargs):
        """Show header section (usually always visible)."""
        pass