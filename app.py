import os
import platform
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, ttk
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageTk
from huggingface_hub import login

from file_embeddings import embeddings_init
from database import chat_db, ChatDB
from file_sorting_page import FileSortingPage
from clerkbot_page import ClerkBotPage
from settings_page import SettingsPage


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Clerk")
        self.root.geometry("950x650")

        # Initialize models and embeddings
        embeddings_init()

        # State Variables
        self.selected_path: str = None
        self.is_cancelled_all: bool = False
        self.current_chat_id: str = None

        # Thread Executor and Helpers
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.sorting_folders: list = []

        # Database
        self.database: ChatDB = chat_db 
        self.chat_session_list: list = self.database.get_all_chatsessions() 
        self.chat_inprogress_list: list = []

        os.makedirs("models", exist_ok=True)
        self.huggingface_token: str = self.database.get_huggingface_token()
        try:
            login(token=self.huggingface_token)
        except Exception as e:
            print(f"Error encountered when logging into huggingface: {e}. Using offline mode.")

        # Settings / Options
        self.use_online_llm = tk.BooleanVar(value=False)
        self.singular_sorting = tk.BooleanVar(value=False)
        self.generate_folder = tk.BooleanVar(value=True)

        # Main Grid Configuration
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)

        # Asset References
        self.prev_icon = ImageTk.PhotoImage(
            Image.open(resource_path("assets/prev_icon.png")).resize((15, 15))
        )
        self.next_icon = ImageTk.PhotoImage(
            Image.open(resource_path("assets/next_icon.png")).resize((15, 15))
        )
        self.reload_icon = ImageTk.PhotoImage(
            Image.open(resource_path("assets/reload_icon.png")).resize((15, 15))
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
        directory_frame = ttk.Frame(self.root, relief="ridge", padding=10)
        directory_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        directory_frame.rowconfigure(1, weight=1)
        directory_frame.columnconfigure(2, weight=1)

        ttk.Button(
            directory_frame, image=self.prev_icon, command=self.prev_folder, cursor="hand2"
        ).grid(row=0, column=0)
        ttk.Button(
            directory_frame, image=self.next_icon, command=self.next_folder, cursor="hand2"
        ).grid(row=0, column=1)

        self.path_entry = ttk.Entry(directory_frame, width=40)
        self.path_entry.grid(row=0, column=2, sticky="ew", padx=2)

        ttk.Button(
            directory_frame, image=self.reload_icon, command=self.refresh_folder, cursor="hand2"
        ).grid(row=0, column=3)
        ttk.Button(
            directory_frame, text="Browse...", command=self.browse_folder, cursor="hand2"
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
        right_container = ttk.Frame(self.root, width=350)
        right_container.grid_propagate(False)

        right_container.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        right_container.rowconfigure(1, weight=1)
        right_container.columnconfigure(0, weight=1)

        nav_frame = ttk.Frame(right_container)
        nav_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        btn_page1 = ttk.Button(
            nav_frame, text="File Sort", command=lambda: self.page1.tkraise(), cursor="hand2"
        )
        btn_page1.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        btn_page2 = ttk.Button(
            nav_frame, text="ClerkBot", command=lambda: self.page2.tkraise(), cursor="hand2"
        )
        btn_page2.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        btn_page3 = ttk.Button(
            nav_frame, text="Settings", command=lambda: self.page3.tkraise(), cursor="hand2"
        )
        btn_page3.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=0)

        pages_container = ttk.Frame(right_container)
        pages_container.grid(row=1, column=0, sticky="nsew")
        pages_container.rowconfigure(0, weight=1)
        pages_container.columnconfigure(0, weight=1)

        # Page instantiation
        self.page1 = FileSortingPage(pages_container, self)
        self.page1.grid(row=0, column=0, sticky="nsew")

        self.page2 = ClerkBotPage(pages_container, self)
        self.page2.grid(row=0, column=0, sticky="nsew")

        self.page3 = SettingsPage(pages_container, self)
        self.page3.grid(row=0, column=0, sticky="nsew")

        self.page1.tkraise()

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

    def _goto_folder(self, dir_path: str):
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, dir_path)
        if hasattr(self, 'page2'):
            self.page2.update_tracking_dir(dir_path)
        self.display_files_in_dir(dir_path)

    def browse_folder(self):
        selected_dir = filedialog.askdirectory(title="Select a Directory")
        if selected_dir:
            self.selected_path = selected_dir
            self._goto_folder(selected_dir)

    def prev_folder(self):
        if not self.path_entry.get():
            return
        previous_dir = os.path.dirname(self.path_entry.get())
        if previous_dir:
            self._goto_folder(previous_dir)

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
            self._goto_folder(next_step)

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
            self.selected_path = new_path
            self._goto_folder(new_path)
        else:
            filename = os.path.join(current_dir, selected_file)
            if platform.system() == "Windows":
                os.startfile(filename)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", filename])
            else:
                subprocess.Popen(["xdg-open", filename])

    def run(self):
        self.root.mainloop()