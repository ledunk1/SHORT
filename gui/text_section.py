import tkinter as tk
from tkinter import ttk, colorchooser
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
from utils.text_rendering import smart_text_wrap, render_text_with_emoji_multiline

class TextSection:
    """Text overlay section of the GUI."""
    
    def __init__(self, parent_frame, update_preview_callback=None, mode="normal"):
        self.parent_frame = parent_frame
        self.update_preview_callback = update_preview_callback or (lambda: None)
        self.mode = mode  # Support for different modes (normal, dual, etc.)
        self.create_text_section()
    
    def create_text_section(self):
        """Create text overlay section."""
        self.text_frame = tk.LabelFrame(
            self.parent_frame, 
            text="📝 Text Overlay Settings", 
            font=("Arial", 11, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50", 
            padx=10, 
            pady=8
        )
        self.text_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Text enable checkbox
        self.text_enabled = tk.BooleanVar()
        text_checkbox = tk.Checkbutton(
            self.text_frame, 
            text="📝 Enable Text Overlay", 
            variable=self.text_enabled, 
            font=("Arial", 10), 
            bg="#f0f0f0",
            command=self.on_text_enable_change
        )
        text_checkbox.pack(pady=5)
        
        # Text settings container
        self.text_settings_frame = tk.Frame(self.text_frame, bg="#f0f0f0")
        self.text_settings_frame.pack(pady=8, fill=tk.X)
        
        # Font selection
        font_frame = tk.Frame(self.text_settings_frame, bg="#f0f0f0")
        font_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(font_frame, text="🔤 Font:", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        self.font_var = tk.StringVar(value="Arial")
        font_options = ["Arial", "Times New Roman", "Helvetica", "Courier New", 
                       "Verdana", "Georgia", "Comic Sans MS", "Impact", 
                       "Trebuchet MS", "Tahoma"]
        
        self.font_combobox = ttk.Combobox(
            font_frame, 
            textvariable=self.font_var, 
            values=font_options, 
            state="readonly", 
            width=15
        )
        self.font_combobox.pack(side=tk.LEFT, padx=(10, 0))
        self.font_combobox.bind("<<ComboboxSelected>>", self.on_text_change)
        
        # Font size
        size_frame = tk.Frame(self.text_settings_frame, bg="#f0f0f0")
        size_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(size_frame, text="📏 Size:", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        self.font_size = tk.IntVar(value=60)
        size_scale = tk.Scale(
            size_frame, 
            from_=20, 
            to=120, 
            orient=tk.HORIZONTAL, 
            variable=self.font_size, 
            bg="#f0f0f0",
            length=200,
            command=self.on_text_change
        )
        size_scale.pack(side=tk.LEFT, padx=10)
        
        size_label = tk.Label(size_frame, textvariable=self.font_size, font=("Arial", 10), bg="#f0f0f0")
        size_label.pack(side=tk.LEFT)
        
        # Text color
        color_frame = tk.Frame(self.text_settings_frame, bg="#f0f0f0")
        color_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(color_frame, text="🎨 Color:", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        self.text_color = tk.StringVar(value="#000000")
        self.color_button = tk.Button(
            color_frame, 
            text="Choose Color", 
            command=self.choose_color,
            font=("Arial", 9), 
            bg="#000000", 
            fg="white",
            width=12
        )
        self.color_button.pack(side=tk.LEFT, padx=10)
        
        # Position controls
        position_frame = tk.Frame(self.text_settings_frame, bg="#f0f0f0")
        position_frame.pack(pady=8, fill=tk.X)
        
        # X Position
        x_pos_frame = tk.Frame(position_frame, bg="#f0f0f0")
        x_pos_frame.pack(pady=3, fill=tk.X)
        
        tk.Label(x_pos_frame, text="↔️ X Position:", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        self.x_position = tk.IntVar(value=50)
        x_scale = tk.Scale(
            x_pos_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            variable=self.x_position, 
            bg="#f0f0f0",
            length=200,
            command=self.on_text_change
        )
        x_scale.pack(side=tk.LEFT, padx=10)
        
        x_percent_label = tk.Label(x_pos_frame, textvariable=self.x_position, font=("Arial", 10), bg="#f0f0f0")
        x_percent_label.pack(side=tk.LEFT)
        tk.Label(x_pos_frame, text="%", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        # Y Position
        y_pos_frame = tk.Frame(position_frame, bg="#f0f0f0")
        y_pos_frame.pack(pady=3, fill=tk.X)
        
        tk.Label(y_pos_frame, text="↕️ Y Position:", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        self.y_position = tk.IntVar(value=80)
        y_scale = tk.Scale(
            y_pos_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            variable=self.y_position, 
            bg="#f0f0f0",
            length=200,
            command=self.on_text_change
        )
        y_scale.pack(side=tk.LEFT, padx=10)
        
        y_percent_label = tk.Label(y_pos_frame, textvariable=self.y_position, font=("Arial", 10), bg="#f0f0f0")
        y_percent_label.pack(side=tk.LEFT)
        tk.Label(y_pos_frame, text="%", font=("Arial", 10), bg="#f0f0f0").pack(side=tk.LEFT)
        
        # Initially disable text settings
        self.update_text_settings_state()
    
    def choose_color(self):
        """Open color chooser dialog."""
        color = colorchooser.askcolor(title="Choose Text Color")
        if color[1]:  # color[1] is the hex value
            self.text_color.set(color[1])
            self.color_button.config(bg=color[1])
            # Determine text color for button based on background
            if self.is_dark_color(color[1]):
                self.color_button.config(fg="white")
            else:
                self.color_button.config(fg="black")
            self.on_text_change()
    
    def is_dark_color(self, hex_color):
        """Determine if a color is dark."""
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # Calculate luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return luminance < 0.5
    
    def on_text_enable_change(self):
        """Handle text enable/disable change."""
        self.update_text_settings_state()
        self.update_preview_callback()
    
    def on_text_change(self, event=None):
        """Handle text settings change."""
        # Only call preview update if callback is available and not None
        if self.update_preview_callback:
            try:
                self.update_preview_callback()
            except Exception as e:
                # Silently handle preview update errors (e.g., when no template is available)
                pass
    
    def update_text_settings_state(self):
        """Enable/disable text settings based on checkbox."""
        state = tk.NORMAL if self.text_enabled.get() else tk.DISABLED
        
        # Update all child widgets
        for widget in self.text_settings_frame.winfo_children():
            self.update_widget_state(widget, state)
    
    def update_widget_state(self, widget, state):
        """Recursively update widget state."""
        try:
            if hasattr(widget, 'config'):
                widget.config(state=state)
        except:
            pass
        
        # Update children
        for child in widget.winfo_children():
            self.update_widget_state(child, state)
    
    def get_text_settings(self):
        """Get current text settings."""
        return {
            'enabled': self.text_enabled.get(),
            'font': self.font_var.get(),
            'size': self.font_size.get(),
            'color': self.text_color.get(),
            'x_position': self.x_position.get(),
            'y_position': self.y_position.get()
        }
    
    def pack_forget(self):
        """Hide text section."""
        if hasattr(self, 'text_frame'):
            self.text_frame.pack_forget()
    
    def pack(self, **kwargs):
        """Show text section."""
        if hasattr(self, 'text_frame'):
            self.text_frame.pack(**kwargs)