"""
GUI Manager - Main GUI coordination and setup
"""

import tkinter as tk
from gui.base_gui import BaseGUI
from gui.header_section import HeaderSection
from gui.mode_section import ModeSection
from gui.video_selection import VideoSelection
from gui.template_section import TemplateSection
from gui.text_section import TextSection
from gui.audio_section import AudioSection
from gui.blur_section import BlurSection
from gui.output_section import OutputSection
from gui.gpu_section import GPUSection
from gui.process_section import ProcessSection
from gui.narasi_section import NarasiSection
from gui.enhanced_audio_section import EnhancedAudioSection
from utils.gpu_config import gpu_config

class GUIManager(BaseGUI):
    """Main GUI Manager that coordinates all GUI sections."""
    
    def __init__(self, root):
        super().__init__(root)
        self.processing_mode = tk.StringVar(value="greenscreen")
        self.gpu_enabled = tk.BooleanVar(value=gpu_config.GPU_AVAILABLE)
        self.selected_encoder = tk.StringVar(value=gpu_config.get_optimal_encoder())
        self.selected_decoder = tk.StringVar(value=gpu_config.get_optimal_decoder() or "CPU")
        
        self.init_sections()
        self.setup_gui()
    
    def init_sections(self):
        """Initialize all GUI sections."""
        # Header
        self.header_section = HeaderSection(self.scrollable_frame)
        
        # Mode selection
        self.mode_section = ModeSection(
            self.scrollable_frame, 
            self.processing_mode, 
            self.on_mode_change
        )
        
        # Video selection sections
        self.video_selection = VideoSelection(self.scrollable_frame)
        
        # Template sections
        self.template_section = TemplateSection(
            self.scrollable_frame, 
            self.update_preview
        )
        
        # Text section
        self.text_section = TextSection(
            self.scrollable_frame, 
            self.update_preview
        )
        
        # Audio sections
        self.audio_section = AudioSection(self.scrollable_frame)
        self.enhanced_audio_section = EnhancedAudioSection(self.scrollable_frame)
        
        # Mode-specific sections
        self.blur_section = BlurSection(self.scrollable_frame)
        self.narasi_section = NarasiSection(self.scrollable_frame)
        
        # System sections
        self.output_section = OutputSection(self.scrollable_frame)
        self.gpu_section = GPUSection(
            self.scrollable_frame,
            self.gpu_enabled,
            self.selected_encoder,
            self.selected_decoder
        )
        
        # Process section
        self.process_section = ProcessSection(self.scrollable_frame)
    
    def setup_gui(self):
        """Setup GUI with initial mode."""
        # Set initial mode
        self.on_mode_change()
    
    def on_mode_change(self):
        """Handle mode change and show/hide appropriate sections."""
        mode = self.processing_mode.get()
        
        # Hide all mode-specific sections first
        self.hide_all_mode_sections()
        
        # Update mode description
        self.mode_section.update_description(mode)
        
        # Show sections based on selected mode
        if mode == "greenscreen":
            self.show_greenscreen_mode()
        elif mode == "blur":
            self.show_blur_mode()
        elif mode == "narasi":
            self.show_narasi_mode()
    
    def hide_all_mode_sections(self):
        """Hide all mode-specific sections."""
        sections_to_hide = [
            self.video_selection,
            self.template_section,
            self.text_section,
            self.audio_section,
            self.enhanced_audio_section,
            self.blur_section,
            self.narasi_section,
            self.output_section,
            self.process_section
        ]
        
        for section in sections_to_hide:
            if hasattr(section, 'pack_forget'):
                section.pack_forget()
        
        # Handle GPU section separately (always visible but may need repositioning)
        if hasattr(self.gpu_section, 'pack_forget'):
            self.gpu_section.pack_forget()
    
    def show_greenscreen_mode(self):
        """Show sections for green screen mode."""
        self.video_selection.pack(pady=10, padx=20, fill=tk.X)
        self.template_section.pack(pady=10, padx=20, fill=tk.X)
        self.text_section.pack(pady=10, padx=20, fill=tk.X)
        self.enhanced_audio_section.pack(pady=10, padx=20, fill=tk.X)
        self.output_section.pack(pady=10, padx=20, fill=tk.X)
        self.gpu_section.pack(pady=10, padx=20, fill=tk.X)
        self.process_section.pack(pady=10, padx=20, fill=tk.X)
    
    def show_blur_mode(self):
        """Show sections for blur mode."""
        self.video_selection.pack(pady=10, padx=20, fill=tk.X)
        self.blur_section.pack(pady=10, padx=20, fill=tk.X)
        self.text_section.pack(pady=10, padx=20, fill=tk.X)
        self.enhanced_audio_section.pack(pady=10, padx=20, fill=tk.X)
        self.output_section.pack(pady=10, padx=20, fill=tk.X)
        self.gpu_section.pack(pady=10, padx=20, fill=tk.X)
        self.process_section.pack(pady=10, padx=20, fill=tk.X)
    
    def show_narasi_mode(self):
        """Show sections for narasi mode."""
        self.video_selection.pack(pady=10, padx=20, fill=tk.X)
        self.template_section.pack(pady=10, padx=20, fill=tk.X)
        self.narasi_section.pack(pady=10, padx=20, fill=tk.X)
        self.text_section.pack(pady=10, padx=20, fill=tk.X)
        self.output_section.pack(pady=10, padx=20, fill=tk.X)
        self.gpu_section.pack(pady=10, padx=20, fill=tk.X)
        self.process_section.pack(pady=10, padx=20, fill=tk.X)
        
        # Setup narasi-specific callbacks
        self.setup_narasi_callbacks()
    
    def setup_narasi_callbacks(self):
        """Setup callbacks for narasi mode to update matching preview."""
        # Override video selection callback to update narasi matching
        original_select_folder = self.video_selection.select_folder
        
        def enhanced_select_folder():
            original_select_folder()
            # Update narasi section with new video folder
            if hasattr(self.narasi_section, 'set_video_folder_path'):
                video_folder = self.video_selection.get_folder_path()
                if video_folder:
                    self.narasi_section.set_video_folder_path(video_folder)
        
        self.video_selection.select_folder = enhanced_select_folder
    
    def show_dual_greenscreen_mode(self):
        """Show sections for dual green screen mode."""
        self.dual_video_selection.pack(pady=10, padx=20, fill=tk.X)
        self.dual_greenscreen_section.pack(pady=10, padx=20, fill=tk.X)
        self.text_section.pack(pady=10, padx=20, fill=tk.X)
        self.dual_audio_section.pack(pady=10, padx=20, fill=tk.X)
        self.output_section.pack(pady=10, padx=20, fill=tk.X)
        self.gpu_section.pack(pady=10, padx=20, fill=tk.X)
        self.process_section.pack(pady=10, padx=20, fill=tk.X)
    
    def update_preview(self):
        """Update template preview."""
        text_settings = self.text_section.get_text_settings()
        
        # Update appropriate template section
        mode = self.processing_mode.get()
        if mode in ["greenscreen", "narasi"]:
            # FIXED: Update template preview for modes that use templates
            self.template_section.update_preview(text_settings)
        elif mode == "blur":
            # Blur mode has its own preview system with text position
            if hasattr(self.blur_section, 'update_blur_preview'):
                self.blur_section.update_blur_preview(text_settings=text_settings)
    
    def set_process_callback(self, callback):
        """Set the process callback function."""
        self.process_section.set_process_callback(callback)
    
    def set_stop_callback(self, callback):
        """Set the stop callback function."""
        self.process_section.set_stop_callback(callback)
    
    def get_all_settings(self):
        """Get all GUI settings."""
        mode = self.processing_mode.get()
        
        settings = {
            'mode': mode,
            'text_settings': self.text_section.get_text_settings(),
            'output_settings': self.output_section.get_output_settings(),
            'gpu_settings': {
                'enabled': self.gpu_enabled.get(),
                'encoder': self.selected_encoder.get(),
                'decoder': self.selected_decoder.get()
            }
        }
        
        if mode == "greenscreen":
            settings.update({
                'folder_path': self.video_selection.get_folder_path(),
                'template_info': self.template_section.get_template_info(),
                'audio_settings': self.enhanced_audio_section.get_enhanced_audio_settings()
            })
        elif mode == "blur":
            settings.update({
                'folder_path': self.video_selection.get_folder_path(),
                'blur_settings': self.blur_section.get_blur_settings(),
                'audio_settings': self.enhanced_audio_section.get_enhanced_audio_settings()
            })
        elif mode == "narasi":
            settings.update({
                'folder_path': self.video_selection.get_folder_path(),
                'template_info': self.template_section.get_template_info(),
                'narasi_settings': self.narasi_section.get_narasi_settings()
            })
        
        return settings
    
    def update_progress(self, value, status="Processing..."):
        """Update progress bar."""
        self.process_section.update_progress(value, status)
    
    def set_processing_state(self, processing=True):
        """Set processing state."""
        self.process_section.set_processing_state(processing)
    
    def reset_progress(self):
        """Reset progress bar."""
        self.process_section.reset_progress()