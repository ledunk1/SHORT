"""
Threading Manager for Video Processing
Prevents UI freezing during long operations
"""

import threading
import queue
import time
from typing import Callable, Any, Optional

class ThreadingManager:
    """Manages background threads for video processing."""
    
    def __init__(self, gui_manager=None):
        self.gui_manager = gui_manager
        self.current_thread = None
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
    def start_processing_thread(self, target_function: Callable, *args, **kwargs):
        """Start processing in background thread."""
        if self.current_thread and self.current_thread.is_alive():
            print("⚠️ Processing already in progress")
            return False
        
        # Reset events and queues
        self.stop_event.clear()
        self._clear_queues()
        
        # Create and start thread
        self.current_thread = threading.Thread(
            target=self._thread_wrapper,
            args=(target_function, args, kwargs),
            daemon=True
        )
        
        self.current_thread.start()
        
        # Start progress monitoring
        if self.gui_manager:
            self._start_progress_monitor()
        
        return True
    
    def _thread_wrapper(self, target_function: Callable, args: tuple, kwargs: dict):
        """Wrapper for thread execution with error handling."""
        try:
            print("🚀 Starting background processing thread...")
            
            # Add progress callback to kwargs if not present
            if 'progress_callback' not in kwargs:
                kwargs['progress_callback'] = self.update_progress
            
            # Execute the target function
            result = target_function(*args, **kwargs)
            
            # Put result in queue
            self.result_queue.put(('success', result))
            print("✅ Background processing completed successfully")
            
        except Exception as e:
            print(f"❌ Background processing error: {e}")
            import traceback
            traceback.print_exc()
            self.result_queue.put(('error', str(e)))
        
        finally:
            # Signal completion
            self.progress_queue.put(('complete', None))
    
    def _start_progress_monitor(self):
        """Start monitoring progress updates from background thread."""
        if not self.gui_manager:
            return
        
        def check_progress():
            try:
                # Check for progress updates
                while not self.progress_queue.empty():
                    try:
                        msg_type, data = self.progress_queue.get_nowait()
                        
                        if msg_type == 'progress':
                            progress, status = data
                            self.gui_manager.update_progress(progress, status)
                        
                        elif msg_type == 'complete':
                            # Processing completed, check result
                            self._handle_completion()
                            return  # Stop monitoring
                    
                    except queue.Empty:
                        break
                
                # Continue monitoring if thread is still alive
                if self.current_thread and self.current_thread.is_alive():
                    self.gui_manager.root.after(100, check_progress)
                else:
                    # Thread finished, handle completion
                    self._handle_completion()
            
            except Exception as e:
                print(f"⚠️ Progress monitor error: {e}")
                self._handle_completion()
        
        # Start monitoring
        self.gui_manager.root.after(100, check_progress)
    
    def _handle_completion(self):
        """Handle processing completion."""
        if not self.gui_manager:
            return
        
        try:
            # Get result from queue
            if not self.result_queue.empty():
                result_type, result_data = self.result_queue.get_nowait()
                
                if result_type == 'success':
                    self.gui_manager.update_progress(100, "✅ Processing completed successfully!")
                    self._show_success_message()
                else:
                    self.gui_manager.update_progress(0, "❌ Processing failed")
                    self._show_error_message(result_data)
            else:
                # No result available, assume completion
                self.gui_manager.update_progress(100, "✅ Processing completed!")
                self._show_success_message()
        
        except Exception as e:
            print(f"⚠️ Completion handler error: {e}")
            self.gui_manager.update_progress(0, "❌ Processing completed with errors")
        
        finally:
            # Reset processing state
            self.gui_manager.set_processing_state(False)
    
    def _show_success_message(self):
        """Show success message to user."""
        try:
            import tkinter.messagebox as messagebox
            messagebox.showinfo(
                "Processing Complete", 
                "✅ Video processing completed successfully!\n\n"
                "Check the output folder for processed files."
            )
        except Exception as e:
            print(f"⚠️ Could not show success message: {e}")
    
    def _show_error_message(self, error_msg: str):
        """Show error message to user."""
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror(
                "Processing Failed", 
                f"❌ Video processing failed:\n\n{error_msg}\n\n"
                "Please check the console for detailed error information."
            )
        except Exception as e:
            print(f"⚠️ Could not show error message: {e}")
    
    def update_progress(self, progress: float, status: str = "Processing..."):
        """Update progress from background thread."""
        try:
            self.progress_queue.put(('progress', (progress, status)))
        except Exception as e:
            print(f"⚠️ Progress update error: {e}")
    
    def stop_processing(self):
        """Stop current processing."""
        if self.current_thread and self.current_thread.is_alive():
            print("🛑 Stopping background processing...")
            self.stop_event.set()
            
            # Wait for thread to finish (with timeout)
            self.current_thread.join(timeout=5.0)
            
            if self.current_thread.is_alive():
                print("⚠️ Thread did not stop gracefully")
            else:
                print("✅ Background processing stopped")
        
        # Reset GUI state
        if self.gui_manager:
            self.gui_manager.set_processing_state(False)
            self.gui_manager.update_progress(0, "Processing stopped")
    
    def is_processing(self) -> bool:
        """Check if processing is currently running."""
        return self.current_thread and self.current_thread.is_alive()
    
    def _clear_queues(self):
        """Clear all queues."""
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

class ProgressCallback:
    """Helper class for progress callbacks in threaded operations."""
    
    def __init__(self, threading_manager: ThreadingManager):
        self.threading_manager = threading_manager
        self.last_update_time = 0
        self.update_interval = 0.1  # Update every 100ms minimum
    
    def __call__(self, progress: float, status: str = "Processing..."):
        """Call progress update with rate limiting."""
        current_time = time.time()
        
        # Rate limit updates to prevent UI flooding
        if current_time - self.last_update_time >= self.update_interval:
            self.threading_manager.update_progress(progress, status)
            self.last_update_time = current_time