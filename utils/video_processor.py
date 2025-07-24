"""
Video Processor - Legacy compatibility wrapper
This file maintains backward compatibility with the old structure
"""

# Import the new VideoProcessor as the main class for backward compatibility
from utils.video_processor_main import VideoProcessor

# Re-export for backward compatibility
__all__ = ['VideoProcessor']