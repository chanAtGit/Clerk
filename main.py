import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os
from pathlib import Path

def browse_folder():
    # Open the folder directory selector
    selected_dir = filedialog.askdirectory(title="Select a Directory")

    # 2. Check if the user selected a directory or cancelled
    if selected_dir:
        path_entry.delete(0, tk.END)
        path_entry.insert(0, selected_dir)
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


# Create main application window
root = tk.Tk()
root.title("Clerk")
root.geometry("900x600")  # Width x Height in pixels

# Configure main window grid weights
root.rowconfigure(0, weight=0)  # Label row
root.rowconfigure(1, weight=1)  # Directory frame and user control frame row
root.rowconfigure(2, weight=0)  # Quit button row
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)

# Application Title
label = tk.Label(root, text="🤓Clerk - AI Powered File System🤓", font=("Arial", 16))
label.grid(row=0, column=0, columnspan=2) # center label over the two columns

# COMPONENT 1: DISPLAY PATH DIRECTORY AND ITS FILES
# Create a frame
directory_frame = ttk.Frame(root, width=450, relief="ridge", padding=10)
directory_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

# Configure the frame's grid to allow internal widgets to expand
directory_frame.rowconfigure(0, weight=0)
directory_frame.rowconfigure(1, weight=1)
directory_frame.columnconfigure(0, weight=1)

# Path Input Display
path_entry = ttk.Entry(directory_frame, width=40)
path_entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")

# Browse Button
browse_btn = ttk.Button(
    directory_frame, text="Browse...", command=browse_folder
)
browse_btn.grid(row=0, column=1)

# COMPONENT 1.2: FILE LIST
list_frame = ttk.Frame(directory_frame, width=450)
list_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=10)

num_of_files_label = ttk.Label(list_frame, text="")
num_of_files_label.pack(anchor='w')

scrollbar = tk.Scrollbar(list_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Listbox to display filenames
file_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("Arial",10))
file_listbox.insert(tk.END, "No files here")
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
    control_frame, text="Quit", command=root.destroy, width=5, height=1, font=("Arial", 12)
)
button.pack()

# Start the infinite event loop
root.mainloop()
