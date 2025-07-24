import tkinter as tk
from tkinter import ttk
from utils.gpu_config import gpu_config

class GPUSection:
    """GPU settings section of the GUI."""
    
    def __init__(self, parent_frame, gpu_enabled, selected_encoder, selected_decoder):
        self.parent_frame = parent_frame
        self.gpu_enabled = gpu_enabled
        self.selected_encoder = selected_encoder
        self.selected_decoder = selected_decoder
        self.create_gpu_settings()
    
    def create_gpu_settings(self):
        """Create GPU settings section."""
        self.gpu_frame = tk.LabelFrame(
            self.parent_frame, 
            text="🎮 GPU Acceleration Settings", 
            font=("Arial", 11, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50", 
            padx=10, 
            pady=8
        )
        self.gpu_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # GPU Status
        self.create_gpu_status()
        
        # GPU Enable/Disable
        self.create_gpu_controls()
        
        # Encoder/Decoder Selection
        self.create_codec_selection()
        
        # Performance Info
        self.create_performance_info()
        
        # Update initial state
        self.on_gpu_enable_change()
    
    def create_gpu_status(self):
        """Create GPU status display."""
        gpu_status_frame = tk.Frame(self.gpu_frame, bg="#f0f0f0")
        gpu_status_frame.pack(pady=5, fill=tk.X)
        
        gpu_status_text = "✅ GPU Available" if gpu_config.GPU_AVAILABLE else "❌ GPU Not Available"
        gpu_status_color = "#27ae60" if gpu_config.GPU_AVAILABLE else "#e74c3c"
        
        gpu_status_label = tk.Label(
            gpu_status_frame, 
            text=gpu_status_text, 
            font=("Arial", 10, "bold"), 
            fg=gpu_status_color, 
            bg="#f0f0f0"
        )
        gpu_status_label.pack(side=tk.LEFT)
    
    def create_gpu_controls(self):
        """Create GPU enable/disable controls."""
        gpu_enable_frame = tk.Frame(self.gpu_frame, bg="#f0f0f0")
        gpu_enable_frame.pack(pady=5, fill=tk.X)
        
        gpu_checkbox = tk.Checkbutton(
            gpu_enable_frame, 
            text="🚀 Enable GPU Acceleration", 
            variable=self.gpu_enabled, 
            font=("Arial", 10), 
            bg="#f0f0f0", 
            command=self.on_gpu_enable_change,
            state=tk.NORMAL if gpu_config.GPU_AVAILABLE else tk.DISABLED
        )
        gpu_checkbox.pack(side=tk.LEFT)
    
    def create_codec_selection(self):
        """Create encoder/decoder selection."""
        # Encoder Selection
        encoder_frame = tk.Frame(self.gpu_frame, bg="#f0f0f0")
        encoder_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(encoder_frame, text="🎬 Video Encoder:", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        encoder_options = gpu_config.SUPPORTED_ENCODERS + ["libx264 (CPU)"]
        if not encoder_options:
            encoder_options = ["libx264 (CPU)"]
        
        self.encoder_combobox = ttk.Combobox(
            encoder_frame, 
            textvariable=self.selected_encoder, 
            values=encoder_options, 
            state="readonly", 
            width=20
        )
        self.encoder_combobox.pack(side=tk.LEFT, padx=(10, 0))
        self.encoder_combobox.bind("<<ComboboxSelected>>", self.on_encoder_change)
        
        # Decoder Selection
        decoder_frame = tk.Frame(self.gpu_frame, bg="#f0f0f0")
        decoder_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(decoder_frame, text="🎞️ Video Decoder:", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        decoder_options = gpu_config.SUPPORTED_DECODERS + ["CPU"]
        if not decoder_options:
            decoder_options = ["CPU"]
        
        self.decoder_combobox = ttk.Combobox(
            decoder_frame, 
            textvariable=self.selected_decoder, 
            values=decoder_options, 
            state="readonly", 
            width=20
        )
        self.decoder_combobox.pack(side=tk.LEFT, padx=(10, 0))
        self.decoder_combobox.bind("<<ComboboxSelected>>", self.on_decoder_change)
    
    def create_performance_info(self):
        """Create performance information display."""
        perf_info_frame = tk.Frame(self.gpu_frame, bg="#f0f0f0")
        perf_info_frame.pack(pady=5, fill=tk.X)
        
        perf_text = "💡 GPU acceleration can significantly speed up video processing"
        if gpu_config.SUPPORTED_ENCODERS:
            perf_text += f"\n🎯 Detected: {', '.join(gpu_config.SUPPORTED_ENCODERS[:3])}"
        
        perf_label = tk.Label(
            perf_info_frame, 
            text=perf_text, 
            font=("Arial", 9), 
            fg="#7f8c8d", 
            bg="#f0f0f0", 
            justify=tk.LEFT, 
            wraplength=700
        )
        perf_label.pack()
    
    def on_gpu_enable_change(self):
        """Handle GPU enable/disable change."""
        gpu_enabled = self.gpu_enabled.get() and gpu_config.GPU_AVAILABLE
        
        # Update GPU config
        gpu_config.set_gpu_enabled(gpu_enabled)
        
        # Enable/disable encoder and decoder selection
        self.encoder_combobox.config(state="readonly" if gpu_enabled else tk.DISABLED)
        self.decoder_combobox.config(state="readonly" if gpu_enabled else tk.DISABLED)
        
        # Update encoder/decoder to optimal values
        if gpu_enabled:
            self.selected_encoder.set(gpu_config.get_optimal_encoder())
            self.selected_decoder.set(gpu_config.get_optimal_decoder() or "CPU")
        else:
            self.selected_encoder.set("libx264 (CPU)")
            self.selected_decoder.set("CPU")
    
    def on_encoder_change(self, event=None):
        """Handle encoder change."""
        encoder = self.selected_encoder.get().replace(" (CPU)", "")
        gpu_config.set_encoder(encoder)
    
    def on_decoder_change(self, event=None):
        """Handle decoder change."""
        decoder = self.selected_decoder.get()
        if decoder == "CPU":
            decoder = None
        gpu_config.set_decoder(decoder)
    
    def pack_forget(self):
        """Hide GPU section."""
        if hasattr(self, 'gpu_frame'):
            self.gpu_frame.pack_forget()
    
    def pack(self, **kwargs):
        """Show GPU section."""
        if hasattr(self, 'gpu_frame'):
            self.gpu_frame.pack(**kwargs)