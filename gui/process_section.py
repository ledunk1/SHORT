import tkinter as tk
from tkinter import ttk

class ProcessSection:
    """Process section with progress bar and controls."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.process_callback = None
        self.create_process_section()
    
    def create_process_section(self):
        """Create process section."""
        self.process_frame = tk.LabelFrame(
            self.parent_frame, 
            text="🚀 Processing", 
            font=("Arial", 11, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50", 
            padx=10, 
            pady=8
        )
        self.process_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Process button
        self.process_btn = tk.Button(
            self.process_frame, 
            text="🎬 Start Processing", 
            command=self.start_processing, 
            font=("Arial", 12, "bold"), 
            bg="#27ae60", 
            fg="white", 
            activebackground="#229954",
            height=2,
            width=20
        )
        self.process_btn.pack(pady=10)
        
        # Stop button (initially hidden)
        self.stop_btn = tk.Button(
            self.process_frame, 
            text="🛑 Stop Processing", 
            command=self.stop_processing, 
            font=("Arial", 10, "bold"), 
            bg="#e74c3c", 
            fg="white", 
            activebackground="#c0392b",
            height=1,
            width=20
        )
        # Don't pack initially - will be shown during processing
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.process_frame, 
            variable=self.progress_var, 
            maximum=100,
            length=400
        )
        self.progress_bar.pack(pady=5, fill=tk.X, padx=20)
        
        # Status label
        self.status_label = tk.Label(
            self.process_frame, 
            text="Ready to process", 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            fg="#7f8c8d"
        )
        self.status_label.pack(pady=5)
        
        # Store reference to stop callback
        self.stop_callback = None
    
    def start_processing(self):
        """Start the processing."""
        if self.process_callback:
            self.process_callback()
    
    def set_process_callback(self, callback):
        """Set the process callback function."""
        self.process_callback = callback
    
    def set_stop_callback(self, callback):
        """Set the stop callback function."""
        self.stop_callback = callback
    
    def stop_processing(self):
        """Stop the processing."""
        if self.stop_callback:
            self.stop_callback()
    
    def update_progress(self, value, status="Processing..."):
        """Update progress bar and status."""
        self.progress_var.set(value)
        self.status_label.config(text=status)
        self.parent_frame.update_idletasks()
    
    def set_processing_state(self, processing=True):
        """Enable/disable process button."""
        if processing:
            self.process_btn.config(state=tk.DISABLED, text="🔄 Processing...")
            self.process_btn.pack_forget()
            self.stop_btn.pack(pady=5)
        else:
            self.process_btn.config(state=tk.NORMAL, text="🎬 Start Processing")
            self.stop_btn.pack_forget()
            self.process_btn.pack(pady=10)
    
    def reset_progress(self):
        """Reset progress bar."""
        self.progress_var.set(0)
        self.status_label.config(text="Ready to process")
    
    def pack_forget(self):
        """Hide process section."""
        if hasattr(self, 'process_frame'):
            self.process_frame.pack_forget()
    
    def pack(self, **kwargs):
        """Show process section."""
        if hasattr(self, 'process_frame'):
            self.process_frame.pack(**kwargs)