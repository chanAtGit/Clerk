import os
import platform
import subprocess
import threading
import time
import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

from file_sort import AutoLabelClusters, MoveFiles, SemanticClustering, SortIntoFolders, print_groups


class GUI:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Clerk")
        self.root.geometry("950x650")

        # State Variables
        self.selected_path = None
        self.worker = None
        self.is_cancelled = False

        # Settings / Options (Using Tkinter BooleanVars for clean binding)
        self.use_online_llm = tk.BooleanVar(value=False)
        self.singular_sorting = tk.BooleanVar(value=False)
        self.generate_folder = tk.BooleanVar(value=True)

        # Main Grid Configuration
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        # Asset References (stored on self to prevent garbage collection)
        self.prev_icon = ImageTk.PhotoImage(
            Image.open("assets/prev_icon.png").resize((15, 15))
        )
        self.next_icon = ImageTk.PhotoImage(
            Image.open("assets/next_icon.png").resize((15, 15))
        )
        self.reload_icon = ImageTk.PhotoImage(
            Image.open("assets/reload_icon.png").resize((15, 15))
        )

        # Build UI Sections
        self._build_header()
        self._build_directory_frame()
        self._build_right_frame()

    def _build_header(self):
        label = tk.Label(
            self.root, text="Clerk - AI Powered File System", font=("Arial", 16)
        )
        label.grid(row=0, column=0, columnspan=2, pady=5)

    def _build_directory_frame(self):
        # COMPONENT 1: LEFT FRAME (DIRECTORY BROWSER)
        directory_frame = ttk.Frame(self.root, relief="ridge", padding=10)
        directory_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        directory_frame.rowconfigure(1, weight=1)
        directory_frame.columnconfigure(2, weight=1)

        ttk.Button(
            directory_frame, image=self.prev_icon, command=self.prev_folder
        ).grid(row=0, column=0)
        ttk.Button(
            directory_frame, image=self.next_icon, command=self.next_folder
        ).grid(row=0, column=1)

        self.path_entry = ttk.Entry(directory_frame, width=40)
        self.path_entry.grid(row=0, column=2, sticky="ew", padx=2)

        ttk.Button(
            directory_frame, image=self.reload_icon, command=self.refresh_folder
        ).grid(row=0, column=3)
        ttk.Button(
            directory_frame, text="Browse...", command=self.browse_folder
        ).grid(row=0, column=4)

        list_frame = ttk.Frame(directory_frame)
        list_frame.grid(row=1, column=0, columnspan=5, sticky="nsew", pady=10)

        self.num_of_files_label = ttk.Label(list_frame, text="Number of files: 0")
        self.num_of_files_label.pack(anchor="w")

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.file_listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set, font=("Arial", 10)
        )
        self.file_listbox.insert(tk.END, "No folder selected")
        self.file_listbox.bind("<Double-Button-1>", self.on_select_file)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.file_listbox.yview)

    def _build_right_frame(self):
        # COMPONENT 2: RIGHT FRAME (STATUS, USER CONTROLS, AND CLERKBOT)
        right_container = ttk.Frame(self.root)
        right_container.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        right_container.rowconfigure(1, weight=1)
        right_container.columnconfigure(0, weight=1)

        # --- Top Navigation Frame ---
        nav_frame = ttk.Frame(right_container)
        nav_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        btn_page1 = ttk.Button(
            nav_frame, text="File Sort", command=lambda: self.page1.tkraise()
        )
        btn_page1.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        btn_page2 = ttk.Button(
            nav_frame, text="ClerkBot", command=lambda: self.page2.tkraise()
        )
        btn_page2.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # --- Pages Container ---
        pages_container = ttk.Frame(right_container)
        pages_container.grid(row=1, column=0, sticky="nsew")
        pages_container.rowconfigure(0, weight=1)
        pages_container.columnconfigure(0, weight=1)

        # --- Page 1: File Sort ---
        self.page1 = ttk.Frame(pages_container, relief="ridge", padding=10)
        self.page1.grid(row=0, column=0, sticky="nsew")
        self.page1.grid_propagate(False)
        self.page1.rowconfigure(0, weight=0)
        self.page1.rowconfigure(1, weight=0)
        self.page1.columnconfigure(0, weight=1)

        # OPTIONS PANEL
        options_panel = ttk.Frame(self.page1)
        options_panel.grid(row=0, column=0, sticky="w", pady=5)

        mode_label = tk.Label(options_panel, text="Choose sorting mode:", font=("Helvetica", 10, "bold"))
        mode_label.pack(anchor="w")
        rb1 = tk.Radiobutton(
                options_panel,
                text="Generate and sort into new folders",
                variable=self.generate_folder,
                value=True
            )
        
        rb2 = tk.Radiobutton(
                options_panel,
                text="Sort into existing folders",
                variable=self.generate_folder,
                value=False
            )
        rb1.pack(anchor="w", padx=20)
        rb2.pack(anchor="w", padx=20)

        online_toggle_btn = tk.Checkbutton(
            options_panel,
            text="Use Gemma4:cloud (Ollama)",
            variable=self.use_online_llm,
        )
        online_toggle_btn.pack(side="top", anchor="w")

        recursive_toggle_btn = tk.Checkbutton(
            options_panel,
            text="Create no subdirectories",
            variable=self.singular_sorting,
        )
        recursive_toggle_btn.pack(side="top", anchor="w")

        # PROGRESS BAR
        progress_label = ttk.Label(self.page1, text="Sorting Progress Indicator:")
        progress_label.grid(row=1, column=0, sticky="w", pady=(10, 2))

        self.progress_bar = ttk.Progressbar(
            self.page1, orient="horizontal", mode="determinate"
        )
        self.progress_bar.grid(row=2, column=0, sticky="ew", pady=5)

        self.status_text = ttk.Label(self.page1, text="", font=("Consolas", 9))
        self.status_text.grid(row=3, column=0, sticky="ew", pady=5)
        self.status_text.bind("<Configure>", self.auto_wrap)

        # BUTTONS PANEL
        btn_panel = ttk.Frame(self.page1)
        btn_panel.grid(row=4, column=0, sticky="ew", pady=5)

        self.sort_btn = tk.Button(
            btn_panel,
            text="Sort",
            command=self.semantic_file_sort,
            width=8,
            font=("Arial", 11),
        )
        self.sort_btn.pack(side=tk.LEFT)

        self.cancel_btn = tk.Button(
            btn_panel,
            text="Cancel",
            command=self.cancel_file_sort,
            width=8,
            font=("Arial", 11),
            state=tk.DISABLED,
        )
        self.cancel_btn.pack(side=tk.RIGHT)

        # --- Page 2: ClerkBot Barebones ---
        self.page2 = ttk.Frame(pages_container, relief="ridge", padding=10)
        self.page2.grid(row=0, column=0, sticky="nsew")

        clerkbot_label = tk.Label(self.page2, text="ClerkBot", font=("Arial", 16))
        clerkbot_label.pack(expand=True)

        self.page1.tkraise()

    # --- Directory and File Helper Methods ---
    def display_files_in_dir(self, selected_dir: str):
        try:
            files = [
                f
                for f in os.listdir(selected_dir)
                if os.path.isfile(os.path.join(selected_dir, f))
            ]
            self.num_of_files_label.config(text=f"Number of files: {len(files)}")
            self.file_listbox.delete(0, tk.END)

            for item in os.listdir(selected_dir):
                full_path = os.path.join(selected_dir, item)
                if os.path.isdir(full_path):
                    self.file_listbox.insert(tk.END, f"📁{item}")

            for item in os.listdir(selected_dir):
                full_path = os.path.join(selected_dir, item)
                if os.path.isfile(full_path):
                    self.file_listbox.insert(tk.END, item)
        except Exception as e:
            self.file_listbox.delete(0, tk.END)
            self.file_listbox.insert(tk.END, f"Error: {e}")

    def browse_folder(self):
        selected_dir = filedialog.askdirectory(title="Select a Directory")
        if selected_dir:
            self.selected_path = selected_dir
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, selected_dir)
            self.display_files_in_dir(selected_dir)

    def prev_folder(self):
        if not self.path_entry.get():
            return
        previous_dir = os.path.dirname(self.path_entry.get())
        if previous_dir:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, previous_dir)
            self.display_files_in_dir(previous_dir)

    def next_folder(self):
        if not self.selected_path:
            return

        current_dir = os.path.normpath(self.path_entry.get())
        target_dir = os.path.normpath(self.selected_path)

        if current_dir == target_dir or not target_dir.startswith(current_dir):
            return

        rel_path = os.path.relpath(target_dir, current_dir)
        path_parts = rel_path.split(os.sep)

        if path_parts and path_parts[0]:
            next_step = os.path.join(current_dir, path_parts[0])
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, next_step)
            self.display_files_in_dir(next_step)

    def refresh_folder(self):
        current_dir = self.path_entry.get()
        if current_dir:
            self.display_files_in_dir(current_dir)

    def on_select_file(self, event):
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            return

        selected_file = self.file_listbox.get(selected_indices[0])
        current_dir = self.path_entry.get()

        if selected_file.startswith("📁"):
            folder_name = selected_file[1:]
            new_path = os.path.join(current_dir, folder_name)
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, new_path)
            self.selected_path = new_path
            self.display_files_in_dir(new_path)
        else:
            filename = os.path.join(current_dir, selected_file)
            if platform.system() == "Windows":
                os.startfile(filename)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", filename])
            else:
                subprocess.Popen(["xdg-open", filename])

    def open_confirm_popup(self, groups: dict) -> bool:
        """Display where the files will be sorted and allow user to confirm or reject."""
        # 1. Create a new top-level window
        popup = tk.Toplevel(self.root)
        popup.title("Confirm sorting")
        popup.geometry("450x450")

        result = False

        def confirm_sorting():
            nonlocal result
            result = True
            popup.destroy()

        def reject_sorting():
            nonlocal result
            result = False
            popup.destroy()

        # 2. Header
        title = tk.Label(popup, text="Sorting result", font=("Arial", 12, "bold"))
        title.pack(pady=(15, 5))

        # 3. Button frame placed at the bottom FIRST
        btn_frame = tk.Frame(popup)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=15)

        confirm_btn = tk.Button(btn_frame, text="Confirm", width=12, command=confirm_sorting)
        confirm_btn.pack(side=tk.LEFT, padx=40)

        reject_btn = tk.Button(btn_frame, text="Reject", width=12, command=reject_sorting)
        reject_btn.pack(side=tk.RIGHT, padx=40)

        # 4. Text widget filling remaining space (measured in lines/chars, not pixels)
        sorting_info = scrolledtext.ScrolledText(popup, height=20, width=65, font=("Arial", 10), wrap=tk.WORD)
        sorting_info.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)

        def print_line(msg: str):
            sorting_info.insert(tk.END, f"{msg}\n")

        print_groups(
            groups,
            sort_into_existing=not self.generate_folder.get(),
            print_to_widget=print_line
        )

        # Make text box uneditable
        sorting_info.config(state="disabled") 

        # 5. Control window behavior
        popup.grab_set()
        popup.wait_window(popup)

        return result

    # --- Thread-Safe Updates & Worker Logic ---
    def update_status_console(self, text):
        self.root.after(0, lambda: self.status_text.config(text=text))

    def update_progress_bar(self, val):
        self.root.after(0, lambda: self.progress_bar.config(value=val))

    def check_cancel_status(self):
        return self.is_cancelled

    def sorting_thread_worker(self, current_dir):
        start_time = time.perf_counter()
        try:
            # Step 1: Semantic Clustering
            if self.generate_folder.get(): # the sorting method generates folders and move files into them
                groups = SemanticClustering(
                    current_dir,
                    recursive=not self.singular_sorting.get(),
                    status_callback=self.update_status_console,
                    progress_callback=self.update_progress_bar,
                    check_cancel=self.check_cancel_status,
                )
            else: # the sorting method moves files into existing folders
                groups = SortIntoFolders(
                    current_dir,
                    status_callback=self.update_status_console,
                    progress_callback=self.update_progress_bar,
                    check_cancel=self.check_cancel_status,
                )

            # Step 2: Dynamic Auto-Labelling
            if groups:
                if self.generate_folder.get():
                    # Only the method which generates folders require the LLM to name the new folders
                    groups = AutoLabelClusters(
                        current_dir,
                        groups,
                        online=self.use_online_llm.get(),
                        status_callback=self.update_status_console,
                        progress_callback=self.update_progress_bar,
                        check_cancel=self.check_cancel_status,
                    )

                # Step 3: Organise Files

                # Open the custom popup to inform the user about the sorting process
                confirm: bool = self.open_confirm_popup(groups)
                
                if confirm:
                    MoveFiles(
                        current_dir,
                        groups,
                        sort_into_existing= not self.generate_folder.get(),
                        status_callback=self.update_status_console,
                        progress_callback=self.update_progress_bar,
                        check_cancel=self.check_cancel_status,
                    )
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                minutes, seconds = divmod(elapsed_time, 60)
                self.update_status_console(
                    f"Sorting completed in {int(minutes)}m {int(seconds)}s"
                )

        except InterruptedError:
            self.update_status_console("Sorting job cleanly stopped by user.")
        except Exception as e:
            self.update_status_console(f"Execution Error: {str(e)}")
        finally:
            self.update_progress_bar(100 if not self.is_cancelled else 0)
            self.sort_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)
            self.display_files_in_dir(current_dir)

    def semantic_file_sort(self):
        current_dir = self.path_entry.get()
        if not current_dir or not os.path.exists(current_dir):
            self.update_status_console("Please select a valid directory first.")
            return

        self.is_cancelled = False

        self.sort_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.update_progress_bar(0)
        self.update_status_console("Initializing semantic indexing system...")

        self.worker = threading.Thread(
            target=self.sorting_thread_worker, args=(current_dir,), daemon=True
        ) # Non blocking thread to keep GUI
        self.worker.start()

    def cancel_file_sort(self):
        self.is_cancelled = True
        self.update_status_console("Cancellation requested... Stopping execution.")
        self.cancel_btn.config(state=tk.DISABLED)

    def auto_wrap(self, event):
        event.widget.config(wraplength=event.width)

    def run(self):
        self.root.mainloop()