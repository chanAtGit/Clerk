import os
import time
import tkinter as tk
from tkinter import ttk, scrolledtext

from file_sort import AutoLabelClusters, MoveFiles, SemanticClustering, SortIntoFolders, print_groups
from widgets import SortingJobWidget, ScrollableFrame


class FileSortingPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, relief="groove", padding=10)
        self.app = app
        self._build_ui()

    def _build_ui(self):
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        # OPTIONS PANEL
        options_panel = ttk.Frame(self)
        options_panel.grid(row=0, column=0, sticky="ew", pady=5)

        tk.Label(options_panel, text="File Sort", font=("Helvetica", 15, "bold")).pack(anchor="center")

        mode_label = tk.Label(options_panel, text="Choose sorting mode:", font=("Arial", 10, "bold"))
        mode_label.pack(anchor="w")
        rb1 = tk.Radiobutton(
            options_panel,
            text="Generate and sort into new folders",
            variable=self.app.generate_folder,
            value=True
        )
        rb2 = tk.Radiobutton(
            options_panel,
            text="Sort into existing folders",
            variable=self.app.generate_folder,
            value=False
        )
        rb1.pack(anchor="w", padx=20)
        rb2.pack(anchor="w", padx=20)

        online_toggle_btn = tk.Checkbutton(
            options_panel,
            text="Use Gemma4:cloud (Ollama)",
            variable=self.app.use_online_llm,
        )
        online_toggle_btn.pack(side="top", anchor="w")

        recursive_toggle_btn = tk.Checkbutton(
            options_panel,
            text="Create no subdirectories",
            variable=self.app.singular_sorting,
        )
        recursive_toggle_btn.pack(side="top", anchor="w")

        # BUTTONS PANEL
        btn_panel = ttk.Frame(self)
        btn_panel.grid(row=1, column=0, sticky="ew", pady=5)

        self.sort_btn = tk.Button(
            btn_panel,
            text="Sort",
            command=self.semantic_file_sort,
            width=8,
            font=("Arial", 11),
            cursor="hand2"
        )
        self.sort_btn.pack(side=tk.LEFT)

        self.cancel_all_btn = tk.Button(
            btn_panel,
            text="Cancel All",
            command=self.cancel_file_sort,
            width=8,
            font=("Arial", 11),
            state=tk.DISABLED,
            cursor="hand2"
        )
        self.cancel_all_btn.pack(side=tk.RIGHT)

        sorting_jobs_frame = ttk.Frame(self, relief='solid')
        sorting_jobs_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        
        sorting_jobs_label = tk.Label(sorting_jobs_frame, text="Current sorting jobs", font=("Helvetica", 11, "underline"))
        sorting_jobs_label.pack(anchor="center", pady=5)

        self.sorting_jobs_list = ScrollableFrame(sorting_jobs_frame)
        self.sorting_jobs_list.pack(fill=tk.BOTH, expand=True, padx=2)

    def open_confirm_popup(self, directory: str, groups: dict) -> bool:
        """Display where the files will be sorted and allow user to confirm or reject."""
        popup = tk.Toplevel(self.app.root)
        popup.title(f"Sorting complete for {directory}")
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

        title = tk.Label(popup, text="Sorting result", font=("Arial", 12, "bold"))
        title.pack(pady=(15, 2))
        title = tk.Label(popup, text=f"{directory}", font=("Arial", 8, "bold"))
        title.pack(pady=(0, 5))

        btn_frame = tk.Frame(popup)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=15)

        confirm_btn = tk.Button(btn_frame, 
                                text="Confirm", 
                                width=12, 
                                command=confirm_sorting,
                                bg="green",
                                fg="white",
                                cursor="hand2")
        confirm_btn.pack(side=tk.LEFT, padx=40)

        reject_btn = tk.Button(btn_frame, 
                               text="Reject", 
                               width=12, 
                               command=reject_sorting,
                               bg="red",
                               fg="white", 
                               cursor="hand2")
        reject_btn.pack(side=tk.RIGHT, padx=40)

        sorting_info = scrolledtext.ScrolledText(popup, height=20, width=65, font=("Arial", 10), wrap=tk.WORD)
        sorting_info.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)

        def print_line(msg: str):
            sorting_info.insert(tk.END, f"{msg}\n")

        print_groups(
            groups,
            sort_into_existing=not self.app.generate_folder.get(),
            print_to_widget=print_line
        )

        sorting_info.config(state="disabled") 

        popup.grab_set()
        popup.wait_window(popup)

        return result

    def sorting_thread_worker(self, sorting_widget: SortingJobWidget):
        start_time = time.perf_counter()
        try:
            current_dir = sorting_widget.target_dir
            recursive_config: bool = sorting_widget.recursive
            online_config: bool = sorting_widget.online
            sort_into_existing_config: bool = sorting_widget.sort_into_existing

            def safe_set_status(text):
                self.app.root.after(0, lambda: sorting_widget.set_status(text))

            def safe_set_progress(val):
                self.app.root.after(0, lambda: sorting_widget.set_progress(val))

            safe_set_progress(0)
            safe_set_status("Initializing semantic indexing system...")

            def check_cancel_status() -> bool:
                cancel_status: bool = self.app.is_cancelled_all or sorting_widget.is_cancel
                if cancel_status:
                    safe_set_status("Cancelling sorting...")
                    safe_set_progress(0)
                return cancel_status

            if self.app.generate_folder.get():
                groups = SemanticClustering(
                    current_dir,
                    recursive=recursive_config,
                    status_callback=safe_set_status,
                    progress_callback=safe_set_progress,
                    check_cancel=check_cancel_status,
                )
            else:
                groups = SortIntoFolders(
                    current_dir,
                    status_callback=safe_set_status,
                    progress_callback=safe_set_progress,
                    check_cancel=check_cancel_status,
                )

            if groups:
                if self.app.generate_folder.get():
                    groups = AutoLabelClusters(
                        current_dir,
                        groups,
                        online=online_config,
                        status_callback=safe_set_status,
                        progress_callback=safe_set_progress,
                        check_cancel=check_cancel_status,
                    )

                confirm: bool = self.open_confirm_popup(current_dir, groups)
                
                if confirm:
                    MoveFiles(
                        current_dir,
                        groups,
                        sort_into_existing=sort_into_existing_config,
                        status_callback=safe_set_status,
                        progress_callback=safe_set_progress,
                        check_cancel=check_cancel_status,
                    )

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                minutes, seconds = divmod(elapsed_time, 60)
                safe_set_status(
                    f"Sorting completed in {int(minutes)}m {int(seconds)}s"
                )

        except InterruptedError:
            safe_set_status("Sorting job cleanly stopped by user.")
        except Exception as e:
            safe_set_status(f"Execution Error: {str(e)}")
        finally:
            safe_set_progress(0 if self.app.is_cancelled_all or sorting_widget.is_cancel else 100)
            if self.app.path_entry.get() == current_dir:
                self.app.display_files_in_dir(current_dir)

            time.sleep(5)
            sorting_widget.destroy()
            self.app.sorting_folders.remove(current_dir)
            if len(self.app.sorting_folders) == 0:
                self.cancel_all_btn.config(state=tk.DISABLED)

    def semantic_file_sort(self):
        current_dir = self.app.path_entry.get()
        if not current_dir or not os.path.exists(current_dir):
            print("Please select a valid directory first.")
            return

        if current_dir in self.app.sorting_folders:
            return

        self.app.is_cancelled_all = False
        self.cancel_all_btn.config(state=tk.NORMAL)
        self.app.sorting_folders.append(current_dir)
        
        recursive_config: bool = not self.app.singular_sorting.get()
        online_config: bool = self.app.use_online_llm.get()
        sort_into_existing_config: bool = not self.app.generate_folder.get()
        
        sorting_widget = SortingJobWidget(
            self.sorting_jobs_list.scrollable_frame,
            target_dir=current_dir,
            recursive=recursive_config,
            online=online_config,
            sort_into_existing=sort_into_existing_config
        )
        sorting_widget.pack(pady=5, fill=tk.X)
    
        self.app.executor.submit(self.sorting_thread_worker, sorting_widget)

    def cancel_file_sort(self):
        self.app.is_cancelled_all = True