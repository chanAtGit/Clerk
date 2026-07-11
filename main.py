import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import os
import platform
import subprocess

from Semantics import init_huggingface_and_models, SemanticClustering, AutoLabelClusters, MoveFiles

selected_path = None

## File and Directory Functions
def display_files_in_dir(selected_dir: str):
        # Clear any existing entries in the Listbox
        num_of_files_label.config(text=f"Number of files: {len(os.listdir(selected_dir))}")
        file_listbox.delete(0, tk.END)

        try:
            # 3. Read all items inside the chosen directory
            # 3.1 Insert directories first
            for dir in os.listdir(selected_dir):
                # Verify that it is a file (and not a folder)
                full_path = os.path.join(selected_dir, dir)
                if os.path.isdir(full_path):
                    # Append the filename to the Tkinter Listbox
                    entry_text = f"📁{dir}"
                    file_listbox.insert(tk.END, entry_text)
            
            # 3.1 Insert files later
            for file in os.listdir(selected_dir):
                # Verify that it is a file (and not a folder)
                full_path = os.path.join(selected_dir, file)
                if os.path.isfile(full_path):
                    # Append the filename to the Tkinter Listbox
                    file_listbox.insert(tk.END, file)
        except Exception as e:
            file_listbox.insert(tk.END, f"Error: {e}")

def browse_folder():
    # Open the folder directory selector
    selected_dir = filedialog.askdirectory(title="Select a Directory")

    # 2. Check if the user selected a directory or cancelled
    if selected_dir:
        global selected_path
        selected_path = selected_dir
        path_entry.delete(0, tk.END)
        path_entry.insert(0, selected_dir)
        # Clear any existing entries in the Listbox
        display_files_in_dir(selected_dir)

def prev_folder():
    # Go to the parent directory of the selected path
    if not path_entry.get():
        return
    previous_dir = os.path.dirname(path_entry.get()) # get parent directory
    
    if previous_dir:
        path_entry.delete(0, tk.END)
        path_entry.insert(0, previous_dir)
        # Clear any existing entries in the Listbox
        display_files_in_dir(previous_dir)

def next_folder():
    # Go down the directory tree to the next folder (if applicable based on the selected path)
    global selected_path

    if not selected_path:
        return

    # Get the currently viewed directory from the entry box
    current_dir = path_entry.get()

    # Normalize paths to ensure accurate comparison and splitting
    current_dir = os.path.normpath(current_dir)
    target_dir = os.path.normpath(selected_path)

    # If already at the target or current_dir isn't part of the target path, do nothing
    if current_dir == target_dir or not target_dir.startswith(current_dir):
        return

    # Find the remaining relative path from current_dir to target_dir
    # e.g., if current is "C:/Users" and target is "C:/Users/Name/Documents", 
    # rel_path becomes "Name/Documents"
    rel_path = os.path.relpath(target_dir, current_dir)

    # Split the relative path into its individual directory parts
    # e.g., ["Name", "Documents"]
    path_parts = rel_path.split(os.sep)

    if path_parts and path_parts[0]:
        # Take just the immediate next directory
        next_step = os.path.join(current_dir, path_parts[0])

        # Update the path entry box
        path_entry.delete(0, tk.END)
        path_entry.insert(0, next_step)

        # Update the UI listbox with the files inside this next folder
        display_files_in_dir(next_step)

def on_select_file(event):
    # Get the selected file from the Listbox
    selected_indices = file_listbox.curselection()
    if not selected_indices:
        return  # No selection made

    selected_index = selected_indices[0]
    selected_file = file_listbox.get(selected_index)
    current_dir = path_entry.get()

    # Check if the selected item is a directory (starts with 📁)
    if selected_file.startswith("📁"):
        # Remove the folder icon to get the actual folder name
        folder_name = selected_file[1:]
        new_path = os.path.join(current_dir, folder_name)

        # Update the path entry box and display files in the new directory
        path_entry.delete(0, tk.END)
        path_entry.insert(0, new_path)
        global selected_path
        selected_path = new_path # update selected_path (for next_folder function to work properly)
        display_files_in_dir(new_path)
    else: # selected item is not a directory, and is a file instead
        # Open the file directly
        filename = os.path.join(current_dir, selected_file)
        if platform.system() == "Windows":
            os.startfile(filename)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", filename])
        else:  # Linux
            subprocess.Popen(["xdg-open", filename])

## User Control Functions

def semantic_file_sort():
    # Get the currently viewed directory from the entry box
    current_dir = path_entry.get()
    clusters = SemanticClustering(current_dir)
    clusters = AutoLabelClusters(current_dir, clusters)
    MoveFiles(current_dir, clusters)
    display_files_in_dir(current_dir) # update current display to reflect the changes made by the semantic sort

## GUI ##
# Create main application window
if __name__ == "__main__":
    init_huggingface_and_models()
    root = tk.Tk()
    root.title("Clerk")
    root.geometry("900x600")  # Width x Height in pixels

    # Configure main window grid weights
    root.rowconfigure(0, weight=0)  # Label row
    root.rowconfigure(1, weight=1)  # Directory frame and user control frame row
    root.rowconfigure(2, weight=0)  # Quit button row
    root.columnconfigure(0, weight=1) # Directory frame
    root.columnconfigure(1, weight=1) # user control frame

    ## Load Image Assets ##
    prev_icon = Image.open("assets/prev_icon.png")
    next_icon = Image.open("assets/next_icon.png")
    prev_icon = prev_icon.resize((15, 15))
    next_icon = next_icon.resize((15, 15))
    prev_icon = ImageTk.PhotoImage(prev_icon) #make image into a Tkinter-compatible object
    next_icon = ImageTk.PhotoImage(next_icon) 

    # Application Title
    label = tk.Label(root, text="Clerk - AI Powered File System", font=("Arial", 16))
    label.grid(row=0, column=0, columnspan=2) # center label over the two columns

    # COMPONENT 1: DISPLAY PATH DIRECTORY AND ITS FILES
    # Create a frame
    directory_frame = ttk.Frame(root, width=450, relief="ridge", padding=10)
    directory_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    # Configure the frame's grid to allow internal widgets to expand
    directory_frame.rowconfigure(0, weight=0)
    directory_frame.rowconfigure(1, weight=1)
    directory_frame.columnconfigure(0, weight=0)
    directory_frame.columnconfigure(1, weight=0)
    directory_frame.columnconfigure(2, weight=1)
    directory_frame.columnconfigure(3, weight=0)

    # COMPONENT 1.1: DIRECTORY SEARCH BAR
    # Previous and Next Directory Buttons
    prev_btn = ttk.Button(
        directory_frame, image=prev_icon, command=prev_folder
    )
    prev_btn.grid(row=0, column=0)
    next_btn = ttk.Button(
        directory_frame, image=next_icon, command=next_folder
    )
    next_btn.grid(row=0, column=1)

    # Path Input Display
    path_entry = ttk.Entry(directory_frame, width=50)
    path_entry.grid(row=0, column=2, sticky="ew")

    # Browse Button
    browse_btn = ttk.Button(
        directory_frame, text="Browse...", command=browse_folder
    )
    browse_btn.grid(row=0, column=3)

    # COMPONENT 1.2: FILE LIST
    list_frame = ttk.Frame(directory_frame, width=450)
    list_frame.grid(row=1, column=0, columnspan=4, sticky='nsew', pady=10)

    num_of_files_label = ttk.Label(list_frame, text="")
    num_of_files_label.pack(anchor='w')

    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Listbox to display filenames
    file_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("Arial",10))
    file_listbox.insert(tk.END, "No files here") # initialise with a default message
    file_listbox.bind('<Double-Button-1>', on_select_file) # Bind the selection event 
    file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Link scrollbar to listbox
    scrollbar.config(command=file_listbox.yview)


    # COMPONENT 2: USER CONTROLS
    control_frame = ttk.Frame(root, width=450, relief="ridge", padding=10)
    control_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

    # Configure the frame's grid to allow internal widgets to expand
    # directory_frame.rowconfigure(0, weight=0)
    # directory_frame.columnconfigure(0, weight=1)

    button = tk.Button(
        control_frame, text="Sort", command=semantic_file_sort, width=5, height=1, font=("Arial", 12)
    )
    button.pack()

    button = tk.Button(
        control_frame, text="Quit", command=root.destroy, width=5, height=1, font=("Arial", 12)
    )
    button.pack()

    # Start the infinite event loop
    root.mainloop()
