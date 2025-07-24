import tkinter as tk
from tkinter import ttk
import os

class BaseGUI:
    """Base GUI class with common functionality."""
    
    def __init__(self, root):
        self.root = root
        self.setup_base_window()
        self.create_scrollable_frame()
    
    def setup_base_window(self):
        """Setup base window properties with fixed fullscreen support and custom icon."""
        self.root.title("Video Editor - Green Screen & Blur Mode with GPU Acceleration")
        
        # Set custom icon
        self.set_window_icon()
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window to use full screen dimensions
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.root.config(bg="#f0f0f0")
        self.root.resizable(True, True)
        
        # Optional: Set minimum size to prevent too small windows
        min_width = min(1200, screen_width)
        min_height = min(800, screen_height)
        self.root.minsize(min_width, min_height)
        
        # Center window if not fullscreen
        if screen_width > 1920 or screen_height > 1080:
            self.center_window()
    
    def set_window_icon(self):
        """Set custom window icon using yt.ico file."""
        try:
            # Get the path to the icon file
            icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yt.ico")
            
            # Check if icon file exists
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
                print(f"✅ Custom icon loaded: {icon_path}")
            else:
                print(f"⚠️ Icon file not found: {icon_path}")
                # Try alternative path (current directory)
                alt_icon_path = "yt.ico"
                if os.path.exists(alt_icon_path):
                    self.root.iconbitmap(alt_icon_path)
                    print(f"✅ Custom icon loaded from current directory: {alt_icon_path}")
                else:
                    print("⚠️ Icon file not found, using default icon")
        except Exception as e:
            print(f"⚠️ Error setting custom icon: {e}")
            print("Using default Tkinter icon")
    
    def center_window(self):
        """Center window on screen for large displays."""
        self.root.update_idletasks()
        
        # Use optimal size for large screens
        width = min(1400, self.root.winfo_screenwidth() - 100)
        height = min(1000, self.root.winfo_screenheight() - 100)
        
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_scrollable_frame(self):
        """Create scrollable frame that fills the entire window."""
        # Main container that fills the window
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for scrolling
        self.canvas = tk.Canvas(main_container, bg="#f0f0f0", highlightthickness=0)
        
        # Scrollbar
        self.scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=self.canvas.yview)
        
        # Scrollable frame
        self.scrollable_frame = tk.Frame(self.canvas, bg="#f0f0f0")
        
        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar to fill container
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind events for responsive behavior
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # FIXED: Bind mouse wheel events to multiple widgets for better scrolling
        self._bind_mousewheel_events()
        
        # Bind keyboard scrolling
        self.root.bind("<Up>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.root.bind("<Down>", lambda e: self.canvas.yview_scroll(1, "units"))
        self.root.bind("<Prior>", lambda e: self.canvas.yview_scroll(-10, "units"))  # Page Up
        self.root.bind("<Next>", lambda e: self.canvas.yview_scroll(10, "units"))   # Page Down
        
        # Make sure the window can receive focus for keyboard events
        self.root.focus_set()
    
    def _bind_mousewheel_events(self):
        """Bind mouse wheel events to multiple widgets for comprehensive scrolling."""
        # List of widgets to bind mouse wheel events to
        widgets_to_bind = [self.root, self.canvas, self.scrollable_frame]
        
        for widget in widgets_to_bind:
            # Windows and Linux
            widget.bind("<MouseWheel>", self._on_mousewheel)
            # Linux (alternative)
            widget.bind("<Button-4>", self._on_mousewheel_linux)
            widget.bind("<Button-5>", self._on_mousewheel_linux)
        
        # Also bind to the main container when it's created
        def bind_to_children(parent):
            """Recursively bind mouse wheel events to all child widgets."""
            try:
                parent.bind("<MouseWheel>", self._on_mousewheel)
                parent.bind("<Button-4>", self._on_mousewheel_linux)
                parent.bind("<Button-5>", self._on_mousewheel_linux)
                
                for child in parent.winfo_children():
                    bind_to_children(child)
            except:
                pass  # Some widgets might not support binding
        
        # Bind to existing children and set up for future children
        self.root.after(100, lambda: bind_to_children(self.scrollable_frame))
    
    def _on_canvas_configure(self, event):
        """Handle canvas resize to make scrollable frame fill the width."""
        # Update the scrollable frame width to match canvas width
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling with improved sensitivity."""
        # Check if content is scrollable
        try:
            # Get the scroll region
            scroll_top = self.canvas.canvasy(0)
            scroll_bottom = self.canvas.canvasy(self.canvas.winfo_height())
            scroll_region = self.canvas.cget("scrollregion").split()
            
            if len(scroll_region) >= 4:
                content_height = float(scroll_region[3])
                canvas_height = self.canvas.winfo_height()
                
                # Only scroll if content is larger than canvas
                if content_height > canvas_height:
                    # Calculate scroll amount based on event delta
                    if hasattr(event, 'delta'):
                        # Windows
                        scroll_amount = int(-1 * (event.delta / 120))
                    else:
                        # Linux/Mac fallback
                        scroll_amount = -1 if event.num == 4 else 1
                    
                    self.canvas.yview_scroll(scroll_amount, "units")
        except:
            # Fallback to simple scrolling
            try:
                if hasattr(event, 'delta'):
                    self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                else:
                    self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
            except:
                pass
    
    def _on_mousewheel_linux(self, event):
        """Handle mousewheel scrolling for Linux systems."""
        try:
            # Check if content is scrollable
            scroll_region = self.canvas.cget("scrollregion").split()
            
            if len(scroll_region) >= 4:
                content_height = float(scroll_region[3])
                canvas_height = self.canvas.winfo_height()
                
                # Only scroll if content is larger than canvas
                if content_height > canvas_height:
                    if event.num == 4:
                        self.canvas.yview_scroll(-1, "units")
                    elif event.num == 5:
                        self.canvas.yview_scroll(1, "units")
        except:
            # Fallback
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
    
    def create_label_frame(self, title, icon=""):
        """Create a labeled frame with consistent styling and full width."""
        frame = tk.LabelFrame(
            self.scrollable_frame, 
            text=f"{icon} {title}" if icon else title,
            font=("Arial", 11, "bold"), 
            bg="#f0f0f0", 
            fg="#2c3e50",
            padx=15, 
            pady=10
        )
        
        # Bind mouse wheel events to the new frame
        self._bind_mousewheel_to_widget(frame)
        
        return frame
    
    def _bind_mousewheel_to_widget(self, widget):
        """Bind mouse wheel events to a specific widget."""
        try:
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel_linux)
            widget.bind("<Button-5>", self._on_mousewheel_linux)
            
            # Also bind to all children of this widget
            def bind_to_children(parent):
                try:
                    for child in parent.winfo_children():
                        child.bind("<MouseWheel>", self._on_mousewheel)
                        child.bind("<Button-4>", self._on_mousewheel_linux)
                        child.bind("<Button-5>", self._on_mousewheel_linux)
                        bind_to_children(child)
                except:
                    pass
            
            # Bind to children after a short delay to ensure they're created
            widget.after(50, lambda: bind_to_children(widget))
        except:
            pass
    
    def create_button(self, parent, text, command, bg_color="#27ae60", **kwargs):
        """Create a styled button with consistent appearance."""
        button = tk.Button(
            parent, 
            text=text, 
            command=command,
            font=("Arial", 10, "bold"),
            bg=bg_color,
            fg="white",
            activebackground=self.darken_color(bg_color),
            relief="flat",
            borderwidth=0,
            padx=20,
            pady=8,
            cursor="hand2",
            **kwargs
        )
        
        # Bind mouse wheel events to the button
        self._bind_mousewheel_to_widget(button)
        
        return button
    
    def darken_color(self, color):
        """Darken a hex color for active state."""
        color_map = {
            "#27ae60": "#229954",
            "#3498db": "#2980b9",
            "#e67e22": "#d35400",
            "#8e44ad": "#7d3c98"
        }
        return color_map.get(color, color)
    
    def configure_responsive_grid(self, parent, columns=2):
        """Configure responsive grid layout."""
        for i in range(columns):
            parent.grid_columnconfigure(i, weight=1, uniform="column")
        
        # Bind mouse wheel events to the parent
        self._bind_mousewheel_to_widget(parent)
    
    def add_separator(self, parent):
        """Add a visual separator."""
        separator = ttk.Separator(parent, orient='horizontal')
        separator.pack(fill=tk.X, pady=10)
        
        # Bind mouse wheel events to the separator
        self._bind_mousewheel_to_widget(separator)
        
        return separator