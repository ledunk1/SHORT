import tkinter as tk
from tkinter import filedialog
import os

class OutputSection:
    """Output settings section of the GUI."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.output_folder_path = ""
        self.create_output_section()
    
    def create_output_section(self):
        """Create output settings section."""
        self.output_frame = tk.LabelFrame(
            self.parent_frame, 
            text="📁 Output Settings", 
            font=("Arial", 11, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50", 
            padx=10, 
            pady=8
        )
        self.output_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Output folder selection
        self.output_label = tk.Label(
            self.output_frame, 
            text="Output will be saved in the same folder as input videos", 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            fg="#7f8c8d"
        )
        self.output_label.pack(pady=5)
        
        # Optional custom output folder
        custom_output_frame = tk.Frame(self.output_frame, bg="#f0f0f0")
        custom_output_frame.pack(pady=5, fill=tk.X)
        
        self.custom_output_enabled = tk.BooleanVar()
        custom_checkbox = tk.Checkbutton(
            custom_output_frame, 
            text="📂 Use custom output folder", 
            variable=self.custom_output_enabled, 
            font=("Arial", 10), 
            bg="#f0f0f0",
            command=self.on_custom_output_change
        )
        custom_checkbox.pack(pady=3)
        
        self.custom_output_frame = tk.Frame(self.output_frame, bg="#f0f0f0")
        self.custom_output_frame.pack(pady=5, fill=tk.X)
        
        self.custom_output_label = tk.Label(
            self.custom_output_frame, 
            text="No custom folder selected", 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            fg="#7f8c8d"
        )
        self.custom_output_label.pack(pady=3)
        
        select_output_btn = tk.Button(
            self.custom_output_frame, 
            text="📂 Select Output Folder", 
            command=self.select_output_folder, 
            font=("Arial", 10, "bold"), 
            bg="#8e44ad", 
            fg="white", 
            activebackground="#7d3c98"
        )
        select_output_btn.pack(pady=5)
        
        # Initially disable custom output
        self.update_custom_output_state()
    
    def select_output_folder(self):
        """Select custom output folder."""
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_folder_path = folder_path
            folder_name = os.path.basename(folder_path)
            self.custom_output_label.config(text=f"Output Folder: {folder_name}")
    
    def on_custom_output_change(self):
        """Handle custom output enable change."""
        self.update_custom_output_state()
    
    def update_custom_output_state(self):
        """Update custom output settings state."""
        state = tk.NORMAL if self.custom_output_enabled.get() else tk.DISABLED
        
        for widget in self.custom_output_frame.winfo_children():
            self.update_widget_state(widget, state)
    
    def update_widget_state(self, widget, state):
        """Recursively update widget state."""
        try:
            if hasattr(widget, 'config'):
                widget.config(state=state)
        except:
            pass
        
        for child in widget.winfo_children():
            self.update_widget_state(child, state)
    
    def get_output_settings(self):
        """Get output settings."""
        return {
            'custom_enabled': self.custom_output_enabled.get(),
            'custom_folder': self.output_folder_path if self.custom_output_enabled.get() else ""
        }
    
    def pack_forget(self):
        """Hide output section."""
        if hasattr(self, 'output_frame'):
            self.output_frame.pack_forget()
    
    def pack(self, **kwargs):
        """Show output section."""
        if hasattr(self, 'output_frame'):
            self.output_frame.pack(**kwargs)