"""
GUI Components - Legacy compatibility wrapper
This file maintains backward compatibility with the old structure
"""

# Import the new GUIManager as VideoEditorGUI for backward compatibility
from utils.gui_manager import GUIManager as VideoEditorGUI

# Re-export for backward compatibility
__all__ = ['VideoEditorGUI']