"""
Video Processor Main - Main processor class that coordinates everything
"""

import tkinter.messagebox as messagebox
from utils.threading_manager import ThreadingManager, ProgressCallback
from utils.video_processor_core import VideoProcessorCore
from utils.video_processor_modes import VideoProcessorModes

class VideoProcessor:
    """Main video processor that coordinates all processing modes."""
    
    def __init__(self, gui_manager):
        self.gui_manager = gui_manager
        self.core = VideoProcessorCore(gui_manager)
        self.modes = VideoProcessorModes(self.core, gui_manager)
        self.threading_manager = ThreadingManager(gui_manager)
    
    def process_videos_bulk(self):
        """Main processing function called by GUI."""
        try:
            if not self.gui_manager:
                print("❌ No GUI manager available")
                return
            
            # Get all settings from GUI
            settings = self.gui_manager.get_all_settings()
            
            # Validate settings first
            is_valid, validation_message = self.validate_settings(settings)
            if not is_valid:
                messagebox.showerror("Invalid Settings", validation_message)
                return
            
            # Set processing state
            self.gui_manager.set_processing_state(True)
            self.gui_manager.update_progress(0, "Initializing...")
            
            # Start processing in background thread
            success = self.threading_manager.start_processing_thread(
                self._process_in_background, settings
            )
            
            if not success:
                self.gui_manager.set_processing_state(False)
                messagebox.showerror(
                    "Processing Error", 
                    "Could not start processing. Another process may be running."
                )
        
        except Exception as e:
            print(f"❌ Critical error in bulk processing: {e}")
            import traceback
            traceback.print_exc()
            
            if self.gui_manager:
                self.gui_manager.update_progress(0, "❌ Critical error occurred")
                messagebox.showerror(
                    "Critical Error", 
                    f"❌ A critical error occurred during processing:\n\n{str(e)}\n\n"
                    "Please check the console for detailed error information."
                )
        
        finally:
            # Reset processing state
            if self.gui_manager:
                self.gui_manager.set_processing_state(False)
    
    def _process_in_background(self, settings, progress_callback=None):
        """Process videos in background thread."""
        mode = settings['mode']
        
        print(f"🚀 Starting bulk processing in {mode} mode...")
        
        # Create progress callback if provided
        if progress_callback:
            # Inject progress callback into modes
            self.modes.progress_callback = progress_callback
        
        # Process based on mode
        if mode == "greenscreen":
            return self.modes.process_greenscreen_mode(settings)
        elif mode == "blur":
            return self.modes.process_blur_mode(settings)
        elif mode == "narasi":
            return self.modes.process_narasi_mode(settings)
        else:
            print(f"❌ Unknown processing mode: {mode}")
            return False
    
    def stop_processing(self):
        """Stop current processing."""
        self.threading_manager.stop_processing()
    
    def validate_settings(self, settings):
        """Validate settings before processing."""
        mode = settings['mode']
        
        # Common validations
        if not settings.get('text_settings'):
            return False, "Text settings not found"
        
        if not settings.get('gpu_settings'):
            return False, "GPU settings not found"
        
        # Mode-specific validations
        if mode in ["greenscreen", "blur"]:
            if not settings.get('folder_path'):
                return False, "No video folder selected"
        
        elif mode == "narasi":
            if not settings.get('folder_path'):
                return False, "No video folder selected"
            narasi_settings = settings.get('narasi_settings', {})
            if not narasi_settings.get('audio_folder_path'):
                return False, "No audio folder selected for narasi mode"
            if not settings.get('template_info', {}).get('path'):
                return False, "No template selected"
        
        return True, "Settings valid"
    
    def get_processing_summary(self, settings):
        """Get a summary of what will be processed."""
        mode = settings['mode']
        summary = f"Processing Mode: {mode.replace('_', ' ').title()}\n"
        
        if mode in ["greenscreen", "blur"]:
            folder_path = settings.get('folder_path', 'Not selected')
            summary += f"Video Folder: {folder_path}\n"
        
        elif mode == "narasi":
            folder_path = settings.get('folder_path', 'Not selected')
            narasi_settings = settings.get('narasi_settings', {})
            audio_folder = narasi_settings.get('audio_folder_path', 'Not selected')
            audio_mode = narasi_settings.get('audio_mode', 'narasi_only')
            summary += f"Video Folder: {folder_path}\n"
            summary += f"Audio Folder: {audio_folder}\n"
            summary += f"Audio Mode: {audio_mode.replace('_', ' ').title()}\n"
            if audio_mode == "mixed_audio":
                summary += f"Narasi Volume: {narasi_settings.get('narasi_volume', 100)}%\n"
                summary += f"Original Volume: {narasi_settings.get('original_volume', 30)}%\n"
        
        # Template info
        template_info = settings.get('template_info', {})
        if template_info.get('path'):
            summary += f"Template: {template_info['path']}\n"
        
        # Text overlay
        text_settings = settings.get('text_settings', {})
        if text_settings.get('enabled'):
            summary += f"Text Overlay: Enabled ({text_settings.get('font', 'Arial')})\n"
        else:
            summary += "Text Overlay: Disabled\n"
        
        # GPU acceleration
        gpu_settings = settings.get('gpu_settings', {})
        if gpu_settings.get('enabled'):
            summary += f"GPU Acceleration: Enabled ({gpu_settings.get('encoder', 'Unknown')})\n"
        else:
            summary += "GPU Acceleration: Disabled\n"
        
        return summary