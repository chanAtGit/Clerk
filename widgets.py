import tkinter as tk
import re
from tkinter import ttk

class ScrollableFrame(ttk.Frame):
    """A scrollable frame that can contain other widgets."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Create a canvas and a vertical scrollbar for scrolling
        self.canvas = tk.Canvas(self, width=250, height=300)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Configure scrollregion when content resizes
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Store window ID so we can match width on canvas resize
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Force inner frame to match canvas width on resize
        self.canvas.bind(
            "<Configure>", 
            lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width)
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack the canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

class SortingJobWidget(ttk.Frame):
    """Custom widget encapsulating the sorting progress UI components."""

    def __init__(self, parent, target_dir:str, sort_into_existing:bool, online:bool, recursive:bool, **kwargs):
        super().__init__(parent, **kwargs)

        # Display Variable
        self.target_dir = target_dir
        self.sort_into_existing = sort_into_existing
        self.online = online
        self.recursive = recursive

        # State Variable
        self.is_cancel = False

        def cancel_sorting():
            self.is_cancel = True
            self.status_text.config(text="Cancelling sorting...")

        # GUI Stuff
        self.configure(relief="raised", padding=5) # relief to be raised

        # 1.1 Progress Label
        folder_name = re.split(r'[/\\]', self.target_dir)[-1]
        process_label = ttk.Label(self, text=f"Sorting: {folder_name}", font=("Arial", 10, "bold"))
        process_label.pack(anchor="w", pady=(0, 2))

        # 1.2 Configuration labels
        sort_config: str = "• Sort into existing folders" if self.sort_into_existing else "• Generate new folders"
        online_config: str = "• Cloud model" if self.online else "• Local model"
        subdir_config: str = "• Create subdirectories" if self.recursive else "• Do not create subdirectories"
        tk.Label(self, text=f"{sort_config}").pack(anchor="w", pady=(0, 1))
        if not self.sort_into_existing:
            tk.Label(self, text=f"{subdir_config}").pack(anchor="w", pady=(0, 1))
        tk.Label(self, text=f"{online_config}").pack(anchor="w", pady=(0, 1))

        # 2. Frame to hold Progress Bar & Cancel Button side-by-side
        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill=tk.X, pady=5)

        self.progress_bar = ttk.Progressbar(
            controls_frame, orient="horizontal", mode="determinate"
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.cancel_btn = ttk.Button(
            controls_frame, text="Cancel", command=cancel_sorting
        )
        self.cancel_btn.pack(side=tk.RIGHT)

        # 3. Status Console Label
        self.status_text = ttk.Label(self, text="", font=("Consolas", 9))
        self.status_text.pack(fill=tk.X, pady=5)
        self.status_text.bind("<Configure>", self._auto_wrap)

    def _auto_wrap(self, event):
        """Auto-wraps long text inside the status console."""
        event.widget.config(wraplength=event.width)

    def set_status(self, text: str):
        """Helper method to update status text."""
        self.status_text.config(text=text)

    def set_progress(self, val: float):
        """Helper method to update progress bar value."""
        self.progress_bar.config(value=val)