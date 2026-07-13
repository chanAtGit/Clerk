import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import os
import platform
import subprocess
import threading 

from Semantics import semantics_init, SemanticClustering, AutoLabelClusters, MoveFiles

selected_path = None
worker = None 
is_cancelled = False 

## File and Directory Functions
def display_files_in_dir(selected_dir: str):
    num_of_files_label.config(text=f"Number of files: {len(os.listdir(selected_dir))}")
    file_listbox.delete(0, tk.END)

    try:
        for item in os.listdir(selected_dir):
            full_path = os.path.join(selected_dir, item)
            if os.path.isdir(full_path):
                file_listbox.insert(tk.END, f"📁{item}")
        
        for item in os.listdir(selected_dir):
            full_path = os.path.join(selected_dir, item)
            if os.path.isfile(full_path):
                file_listbox.insert(tk.END, item)
    except Exception as e:
        file_listbox.insert(tk.END, f"Error: {e}")

def browse_folder():
    selected_dir = filedialog.askdirectory(title="Select a Directory")
    if selected_dir:
        global selected_path
        selected_path = selected_dir
        path_entry.delete(0, tk.END)
        path_entry.insert(0, selected_dir)
        display_files_in_dir(selected_dir)

def prev_folder():
    if not path_entry.get():
        return
    previous_dir = os.path.dirname(path_entry.get())
    if previous_dir:
        path_entry.delete(0, tk.END)
        path_entry.insert(0, previous_dir)
        display_files_in_dir(previous_dir)

def next_folder():
    global selected_path
    if not selected_path:
        return

    current_dir = os.path.normpath(path_entry.get())
    target_dir = os.path.normpath(selected_path)

    if current_dir == target_dir or not target_dir.startswith(current_dir):
        return

    rel_path = os.path.relpath(target_dir, current_dir)
    path_parts = rel_path.split(os.sep)

    if path_parts and path_parts[0]:
        next_step = os.path.join(current_dir, path_parts[0])
        path_entry.delete(0, tk.END)
        path_entry.insert(0, next_step)
        display_files_in_dir(next_step)

def refresh_folder():
    current_dir = path_entry.get()
    display_files_in_dir(current_dir)

def on_select_file(event):
    selected_indices = file_listbox.curselection()
    if not selected_indices:
        return

    selected_file = file_listbox.get(selected_indices[0])
    current_dir = path_entry.get()

    if selected_file.startswith("📁"):
        folder_name = selected_file[1:]
        new_path = os.path.join(current_dir, folder_name)
        path_entry.delete(0, tk.END)
        path_entry.insert(0, new_path)
        global selected_path
        selected_path = new_path
        display_files_in_dir(new_path)
    else:
        filename = os.path.join(current_dir, selected_file)
        if platform.system() == "Windows":
            os.startfile(filename)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", filename])
        else:
            subprocess.Popen(["xdg-open", filename])

## Thread Safe Gui State Changes ##
def update_status_console(text):
    root.after(0, lambda: status_text.config(text=text))

def update_progress_bar(val):
    root.after(0, lambda: progress_bar.config(value=val))

def check_cancel_status():
    global is_cancelled
    return is_cancelled

def sorting_thread_worker(current_dir):
    global is_cancelled
    try:
        # Step 1: Semantic Clustering
        clusters = SemanticClustering(
            current_dir, 
            status_callback=update_status_console, 
            progress_callback=update_progress_bar,
            check_cancel=check_cancel_status
        )
        
        # Step 2: Dynamic Auto-Labelling
        if clusters:
            labeled_clusters = AutoLabelClusters(
                current_dir, clusters, 
                status_callback=update_status_console, 
                progress_callback=update_progress_bar,
                check_cancel=check_cancel_status
            )
            
            # Step 3: Organise Files
            MoveFiles(
                current_dir, labeled_clusters, 
                status_callback=update_status_console, 
                progress_callback=update_progress_bar,
                check_cancel=check_cancel_status
            )
            update_status_console("Sorting job completed successfully!")
        else:
            if not is_cancelled:
                update_status_console("No clusters generated. Process aborted.")
            
    except InterruptedError:
        update_status_console("Sorting job cleanly stopped by user.")
    except Exception as e:
        update_status_console(f"Execution Error: {str(e)}")
    finally:
        # Reset elements cleanly on complete or cancel
        update_progress_bar(100 if not is_cancelled else 0)
        sort_btn.config(state=tk.NORMAL)
        cancel_btn.config(state=tk.DISABLED)
        root.after(0, lambda: display_files_in_dir(current_dir))

def semantic_file_sort():
    current_dir = path_entry.get()
    if not current_dir or not os.path.exists(current_dir):
        update_status_console("Please select a valid directory first.")
        return

    global is_cancelled
    is_cancelled = False

    sort_btn.config(state=tk.DISABLED)
    cancel_btn.config(state=tk.NORMAL)
    update_progress_bar(0)
    update_status_console("Initializing semantic indexing system...")

    global worker
    worker = threading.Thread(target=sorting_thread_worker, args=(current_dir,), daemon=True)
    worker.start()

def cancel_file_sort():
    global is_cancelled
    is_cancelled = True
    update_status_console("Cancellation requested... Stopping execution.")
    cancel_btn.config(state=tk.DISABLED)

def auto_wrap(event):
    event.widget.config(wraplength=event.width)

## GUI Layout ##
if __name__ == "__main__":
    semantics_init()
    root = tk.Tk()
    root.title("Clerk")
    root.geometry("950x650")

    root.rowconfigure(0, weight=0)
    root.rowconfigure(1, weight=1)
    root.rowconfigure(2, weight=0)
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)

    prev_icon = ImageTk.PhotoImage(Image.open("assets/prev_icon.png").resize((15, 15)))
    next_icon = ImageTk.PhotoImage(Image.open("assets/next_icon.png").resize((15, 15)))
    reload_icon = ImageTk.PhotoImage(Image.open("assets/reload_icon.png").resize((15, 15)))

    label = tk.Label(root, text="Clerk - AI Powered File System", font=("Arial", 16))
    label.grid(row=0, column=0, columnspan=2, pady=5)

    # COMPONENT 1: LEFT FRAME (DIRECTORY BROWSER)
    directory_frame = ttk.Frame(root, relief="ridge", padding=10)
    directory_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    directory_frame.rowconfigure(1, weight=1)
    directory_frame.columnconfigure(2, weight=1)

    ttk.Button(directory_frame, image=prev_icon, command=prev_folder).grid(row=0, column=0)
    ttk.Button(directory_frame, image=next_icon, command=next_folder).grid(row=0, column=1)
    path_entry = ttk.Entry(directory_frame, width=40)
    path_entry.grid(row=0, column=2, sticky="ew", padx=2)
    ttk.Button(directory_frame, image=reload_icon, command=refresh_folder).grid(row=0, column=3)
    ttk.Button(directory_frame, text="Browse...", command=browse_folder).grid(row=0, column=4)

    list_frame = ttk.Frame(directory_frame)
    list_frame.grid(row=1, column=0, columnspan=5, sticky='nsew', pady=10)
    num_of_files_label = ttk.Label(list_frame, text="Number of files: 0")
    num_of_files_label.pack(anchor='w')

    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    file_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("Arial", 10))
    file_listbox.insert(tk.END, "No folder selected")
    file_listbox.bind('<Double-Button-1>', on_select_file)
    file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=file_listbox.yview)

    # COMPONENT 2: RIGHT FRAME (STATUS AND USER CONTROLS)
    control_frame = ttk.Frame(root, relief="ridge", padding=10)
    control_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
    control_frame.grid_propagate(False)
    control_frame.rowconfigure(0, weight=0)
    control_frame.rowconfigure(1, weight=0)
    control_frame.columnconfigure(0, weight=1)

    btn_panel = ttk.Frame(control_frame)
    btn_panel.grid(row=0, column=0, sticky="ew", pady=5)
    sort_btn = tk.Button(btn_panel, text="Sort", command=semantic_file_sort, width=8, font=("Arial", 11))
    sort_btn.pack(side=tk.LEFT)
    cancel_btn = tk.Button(btn_panel, text="Cancel", command=cancel_file_sort, width=8, font=("Arial", 11), state=tk.DISABLED)
    cancel_btn.pack(side=tk.RIGHT)

    progress_label = ttk.Label(control_frame, text="Sorting Progress Indicator:")
    progress_label.grid(row=2, column=0, sticky="w", pady=(10, 2))

    progress_bar = ttk.Progressbar(control_frame, orient="horizontal", mode="determinate")
    progress_bar.grid(row=3, column=0, sticky="ew", pady=5)

    status_text = ttk.Label(control_frame, text="", font=("Consolas", 9))
    status_text.grid(row=4, column=0, sticky="ew",pady=5)
    status_text.bind("<Configure>", auto_wrap)

    root.mainloop()