import tkinter as tk

class ModeSection:
    """Mode selection section of the GUI with responsive layout."""
    
    def __init__(self, parent_frame, processing_mode, on_mode_change_callback):
        self.parent_frame = parent_frame
        self.processing_mode = processing_mode
        self.on_mode_change_callback = on_mode_change_callback
        self.create_mode_selection()
    
    def create_mode_selection(self):
        """Create mode selection section with responsive design."""
        mode_frame = tk.LabelFrame(
            self.parent_frame, 
            text="🎯 Processing Mode", 
            font=("Arial", 11, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50", 
            padx=15, 
            pady=10
        )
        mode_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Radio buttons container with responsive layout
        radio_container = tk.Frame(mode_frame, bg="#f0f0f0")
        radio_container.pack(pady=8, fill=tk.X)
        
        # Configure responsive grid
        radio_container.grid_columnconfigure(0, weight=1)
        radio_container.grid_columnconfigure(1, weight=1)
        radio_container.grid_columnconfigure(2, weight=1)
        
        # Radio buttons in a single row for better space utilization
        greenscreen_radio = tk.Radiobutton(
            radio_container, 
            text="🎬 Green Screen Mode", 
            variable=self.processing_mode, 
            value="greenscreen",
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0",
            activebackground="#f0f0f0",
            command=self.on_mode_change_callback,
            padx=10,
            pady=5
        )
        greenscreen_radio.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        blur_radio = tk.Radiobutton(
            radio_container, 
            text="🌀 Blur Background Mode", 
            variable=self.processing_mode, 
            value="blur",
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0",
            activebackground="#f0f0f0",
            command=self.on_mode_change_callback,
            padx=10,
            pady=5
        )
        blur_radio.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        
        narasi_radio = tk.Radiobutton(
            radio_container, 
            text="🎙️ Narasi Mode", 
            variable=self.processing_mode, 
            value="narasi",
            font=("Arial", 10, "bold"), 
            bg="#f0f0f0",
            activebackground="#f0f0f0",
            command=self.on_mode_change_callback,
            padx=10,
            pady=5
        )
        narasi_radio.grid(row=0, column=2, sticky="w", padx=10, pady=5)
        
        # Description with better wrapping
        description_frame = tk.Frame(mode_frame, bg="#f0f0f0")
        description_frame.pack(fill=tk.X, pady=(8, 0))
        
        self.mode_description = tk.Label(
            description_frame, 
            text="Green Screen: Replace green screen area with video content + text overlay (Supports MP4, AVI, MOV, GIF)",
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0", 
            wraplength=800,  # Increased wrap length for wider displays
            justify=tk.LEFT
        )
        self.mode_description.pack(fill=tk.X, padx=5)
    
    def update_description(self, mode):
        """Update mode description with better text."""
        descriptions = {
            "greenscreen": "Green Screen: Replace green screen area with video content + text overlay (Supports MP4, AVI, MOV, GIF, Images)",
            "blur": "Blur Background: Create blurred background with cropped video overlay in 9:16 + text overlay (Supports MP4, AVI, MOV, GIF, Images)",
            "narasi": "Narasi Mode: Concatenate multiple videos, process with green screen template, duration follows audio (Requires audio file)"
        }
        
        text = descriptions.get(mode, "Select a processing mode above")
        self.mode_description.config(text=text)
    
    def pack_forget(self):
        """Hide mode section (usually not needed)."""
        pass
    
    def pack(self, **kwargs):
        """Show mode section (usually always visible)."""
        pass