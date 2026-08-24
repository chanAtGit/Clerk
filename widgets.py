import tkinter as tk
import re
from tkinter import ttk

class ScrollableFrame(ttk.Frame):
    """A scrollable frame that can contain other widgets."""

    def __init__(self, parent, scroll_align:str = "right", **kwargs):
        super().__init__(parent, **kwargs)

        # Create a canvas and a vertical scrollbar for scrolling
        self.canvas = tk.Canvas(self, width=200, height=250)
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
        if scroll_align == "right":
            self.canvas.pack(side="left", fill="both", expand=True)
            self.scrollbar.pack(side="right", fill="y")
        else:
            self.canvas.pack(side="right", fill="both", expand=True)
            self.scrollbar.pack(side="left", fill="y")

    def scroll_to_bottom(self):
        """Scrolls the canvas to the very bottom."""
        self.update_idletasks()  # Force Tkinter to calculate new widget sizes first
        self.canvas.yview_moveto(1.0)  # Move scroll position to 100% (bottom)

    def clear_content(self):
        """Clears all widgets inside the scrollable frame."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.update_idletasks()  # Force Tkinter to calculate new widget sizes first
        self.canvas.yview_moveto(0)  # Move scroll position to 0% (top)

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
            controls_frame, text="Cancel", command=cancel_sorting, cursor="hand2"
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

class ChatWidget:
    '''Displayed on the sidebar, the user is transported to a chat session when clicked.'''
    def __init__(self, parent, name: str, chatsession_id: str, go_to_func=None, delete_func=None):
        self.chatsession_id = chatsession_id

        # 1. Create a Style object
        style = ttk.Style()
        style.configure(
            "Custom.TButton",
            font=("Helvetica", 10, "bold"),
            anchor="w",
        )

        # 2. Create button
        chat_name_btn = ttk.Button(
            parent, 
            text=name, 
            style="Custom.TButton",
            command=lambda: go_to_func(self.chatsession_id) if go_to_func else None,
            cursor="hand2"
        )
        chat_name_btn.pack(fill=tk.X, padx=2, pady=2)

        # 3. Create right-click context menu
        context_menu = tk.Menu(chat_name_btn, tearoff=0)
        context_menu.add_command(
            label="Delete",
            command=lambda: delete_func(self.chatsession_id) if delete_func else None
        )

        def show_context_menu(event):
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()

        # 4. Bind right-click event (<Button-3> for Windows/Linux, <Button-2> for macOS)
        chat_name_btn.bind("<Button-3>", show_context_menu)
        chat_name_btn.bind("<Button-2>", show_context_menu)

class TextBubble(ttk.Frame):
    """A chat message bubble widget supporting markdown formatting, alignment, and auto-expanding multi-line text without scroll traps."""

    def __init__(self, parent, text: str, from_user: bool, **kwargs):
        super().__init__(parent, **kwargs)
        self.text = text
        self.from_user = from_user

        # Outer row frame expands horizontally to allow left/right alignment
        self.pack(fill=tk.X, padx=10, pady=5)

        # Theme & alignment configurations
        if self.from_user:
            bg_color = "#0078D4"      # User bubble background (Blue)
            fg_color = "#FFFFFF"      # User text color (White)
            bg_code = "#005A9E"       # Code background for user
            align_side = tk.RIGHT     # Pack to the right
        else:
            bg_color = "#E9E9EB"      # AI bubble background (Soft Gray)
            fg_color = "#000000"      # AI text color (Black)
            bg_code = "#D1D1D6"       # Code background for AI
            align_side = tk.LEFT      # Pack to the left

        # Inner container frame for padding & background
        bubble_frame = tk.Frame(
            self, bg=bg_color, padx=12, pady=8, highlightthickness=0
        )
        bubble_frame.pack(side=align_side, anchor="e" if self.from_user else "w")

        # Text widget for markdown rendering (takefocus=0 prevents widget focus trapping)
        self.text_widget = tk.Text(
            bubble_frame,
            bg=bg_color,
            fg=fg_color,
            font=("Arial", 10),
            wrap=tk.WORD,
            relief="flat",
            bd=0,
            highlightthickness=0,
            width=1,
            height=1,
            takefocus=0
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)

        # Render Markdown content & adjust dynamic dimensions
        self._render_markdown(bg_code, fg_color)
        self.text_widget.config(state="disabled")
        self._adjust_size()
        self._forward_scroll_events()

    def _render_markdown(self, bg_code: str, fg_code: str):
        """Parses basic Markdown formatting and applies Tkinter text tags."""
        font_family = "Arial"
        font_size = 10

        # Configure formatting tags
        self.text_widget.tag_configure("bold", font=(font_family, font_size, "bold"))
        self.text_widget.tag_configure("italic", font=(font_family, font_size, "italic"))
        self.text_widget.tag_configure("bold_italic", font=(font_family, font_size, "bold italic"))
        self.text_widget.tag_configure("h1", font=(font_family, 14, "bold"))
        self.text_widget.tag_configure("h2", font=(font_family, 12, "bold"))
        self.text_widget.tag_configure(
            "code_block",
            font=("Consolas", 9),
            background=bg_code,
            foreground=fg_code,
            lmargin1=10,
            lmargin2=10
        )
        self.text_widget.tag_configure(
            "inline_code",
            font=("Consolas", 9),
            background=bg_code,
            foreground=fg_code
        )

        lines = self.text.split("\n")
        in_code_block = False

        for i, line in enumerate(lines):
            # Code block toggle (```)
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                self.text_widget.insert(tk.END, line + "\n", "code_block")
                continue

            # Headers
            if line.startswith("# "):
                self.text_widget.insert(tk.END, line[2:] + "\n", "h1")
                continue
            elif line.startswith("## ") or line.startswith("### "):
                header_text = line.lstrip("#").strip()
                self.text_widget.insert(tk.END, header_text + "\n", "h2")
                continue

            # Bullet points
            if line.strip().startswith("- ") or line.strip().startswith("* "):
                self.text_widget.insert(tk.END, "  • ")
                line = line.strip()[2:]

            # Inline formatting (bold, italic, inline code)
            self._insert_inline_formatted(line + ("\n" if i < len(lines) - 1 else ""))

    def _insert_inline_formatted(self, line: str):
        """Splits text on markdown tokens and inserts with appropriate tags."""
        pattern = re.compile(r'(\*\*\*.*?\*\*\*|\*\*.*?\*\*|\*.*?\*|`.*?`)')
        parts = pattern.split(line)

        for part in parts:
            if not part:
                continue
            if part.startswith("***") and part.endswith("***") and len(part) >= 6:
                self.text_widget.insert(tk.END, part[3:-3], "bold_italic")
            elif part.startswith("**") and part.endswith("**") and len(part) >= 4:
                self.text_widget.insert(tk.END, part[2:-2], "bold")
            elif part.startswith("*") and part.endswith("*") and len(part) >= 2:
                self.text_widget.insert(tk.END, part[1:-1], "italic")
            elif part.startswith("`") and part.endswith("`") and len(part) >= 2:
                self.text_widget.insert(tk.END, part[1:-1], "inline_code")
            else:
                text = part.replace(r"\*", "*").replace(r"\`", "`")
                self.text_widget.insert(tk.END, text)

    def _adjust_size(self):
        """Dynamically calculates width and visual height with padding buffers."""
        lines = self.text.split("\n")
        max_line_len = max((len(l) for l in lines), default=0)

        # Cap max bubble width to 45 chars
        calc_width = min(max(max_line_len + 2, 12), 45)
        self.text_widget.config(width=calc_width)

        # Force geometry calculation
        self.update_idletasks()

        # Count visual display lines
        display_lines = self.text_widget.count("1.0", "end-1c", "displaylines")
        actual_height = display_lines[0] if display_lines else int(self.text_widget.index("end-1c").split(".")[0])

        # Buffer 1: Headers (14pt) take more pixel height than default 10pt lines
        header_count = sum(1 for line in lines if line.startswith("#"))
        
        # Buffer 2: Add +1 line padding for multi-line wrapped text or code blocks
        extra_buffer = header_count + (1 if actual_height > 1 or "```" in self.text else 0)

        self.text_widget.config(height=max(actual_height + extra_buffer, 1))

    def _forward_scroll_events(self):
        """Passes mouse wheel events to the parent ScrollableFrame canvas."""
        def _on_mousewheel(event):
            # Locate parent ScrollableFrame's canvas
            widget = self.master
            while widget and not isinstance(widget, ScrollableFrame):
                widget = widget.master
            
            if widget and hasattr(widget, "canvas"):
                if event.delta:
                    widget.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                elif event.num == 4:
                    widget.canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    widget.canvas.yview_scroll(1, "units")
                return "break"

        self.text_widget.bind("<MouseWheel>", _on_mousewheel)
        self.text_widget.bind("<Button-4>", _on_mousewheel)
        self.text_widget.bind("<Button-5>", _on_mousewheel)