import tkinter as tk
from utils.gui_components import VideoEditorGUI
from utils.video_processor import VideoProcessor

def main():
    """Fungsi utama aplikasi."""
    root = tk.Tk()
    
    # Inisialisasi GUI
    gui = VideoEditorGUI(root)
    
    # Inisialisasi Video Processor
    processor = VideoProcessor(gui)
    
    # Set callback untuk tombol proses
    gui.set_process_callback(processor.process_videos_bulk)
    
    # Set callback untuk tombol stop
    gui.set_stop_callback(processor.stop_processing)
    
    # Jalankan aplikasi
    root.mainloop()

if __name__ == "__main__":
    main()