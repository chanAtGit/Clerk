import os
import platform
import subprocess
import time
import re
import tkinter as tk
import threading
from tkinter import scrolledtext, filedialog, ttk
from PIL import Image, ImageTk
from concurrent.futures import ThreadPoolExecutor

from file_sort import AutoLabelClusters, MoveFiles, SemanticClustering, SortIntoFolders, print_groups, LLM_NAME
from file_embeddings import embeddings_init, get_file_id, get_file_mean_embeddings
from model_commons import models_init, unload_embedding_model
from widgets import SortingJobWidget, ScrollableFrame, ChatWidget, TextBubble
from chat_functions import get_chat_response, get_new_chat_title
from database import chat_db, ChatDB, Inquiry

class App:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Clerk")
        self.root.geometry("950x650")

        # Initalize models and embeddings
        models_init()
        embeddings_init()

        # State Variables
        self.selected_path: str = None
        self.is_cancelled_all: bool = False # Flag to determine if all sorting jobs are to be cancelled
        self.current_chat_id: str = None

        # Thread Executor and Helpers
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.sorting_folders = []

        # Database
        self.database = chat_db 
        # They share the same memory address, so we can use the same instance of ChatDB across the app
        self.chat_session_list = self.database.get_all_chatsessions() 
        # chat session list contains tuples in the form of (chat_id, chat_name)
        self.chat_inprogress_list = []
        # list of chat session id with its corresponding chat awaiting bot response 

        # Settings / Options (Using Tkinter BooleanVars for clean binding)
        self.use_online_llm = tk.BooleanVar(value=False)
        self.singular_sorting = tk.BooleanVar(value=False)
        self.generate_folder = tk.BooleanVar(value=True)

        # Main Grid Configuration
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)

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

    def _build_sorting_page(self, parent_container):
        '''Build the sorting page'''
        self.page1 = ttk.Frame(parent_container, relief="groove", padding=10)
        self.page1.grid(row=0, column=0, sticky="nsew")
        self.page1.grid_propagate(False)
        self.page1.rowconfigure(0, weight=0)
        self.page1.rowconfigure(1, weight=0)
        self.page1.rowconfigure(2, weight=0)
        self.page1.rowconfigure(3, weight=1)  # <--- Allow row 3 to expand vertically
        self.page1.columnconfigure(0, weight=1)

        # OPTIONS PANEL
        options_panel = ttk.Frame(self.page1)
        options_panel.grid(row=0, column=0, sticky="w", pady=5)

        mode_label = tk.Label(options_panel, text="Choose sorting mode:", font=("Helvetica", 12, "bold"))
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

        # BUTTONS PANEL
        btn_panel = ttk.Frame(self.page1)
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

        sorting_jobs_label = tk.Label(self.page1, text="Current sorting jobs", font=("Helvetica", 10, "underline"))
        sorting_jobs_label.grid(row=2, column=0, sticky="w", pady=5)

        # Instantiating the standalone scrollable container to store current sorting jobs
        self.sorting_jobs_list = ScrollableFrame(self.page1)
        # Sticky "nsew" fills both vertical and horizontal space
        self.sorting_jobs_list.grid(row=3, column=0, sticky="nsew", pady=5)

    def _reload_chat_list(self, parent_container):
        """reload the recent chat list in the sidebar"""
        # Clear all ChatWidgets (if any)
        for widget in parent_container.winfo_children():
            widget.destroy()

        # add the chat widgets
        for chat_id, chat_name in self.chat_session_list:
            ChatWidget(parent_container, chat_name, chat_id, 
                       go_to_func=self._load_chatsession_chats,
                       delete_func=self._delete_chatsession)
        print("Reloaded chat_list.")
        
    def _build_sidebar(self, parent_container):
        self.sidebar_frame = ttk.Frame(
            parent_container, width=250, relief="solid", padding=10
        )
        
        tk.Button(self.sidebar_frame, 
                  text="💬 New Chat", 
                  font=("Arial", 12, "bold"), 
                  command=self._open_new_chat,
                  cursor="hand2").pack(fill=tk.X, pady=2)

        sidebar_title = ttk.Label(
            self.sidebar_frame, text="Recent Chats", font=("Arial", 10, "bold")
        )
        sidebar_title.pack(anchor="w", pady=(20,0))

        self.recent_chat_list = ScrollableFrame(self.sidebar_frame)
        self.recent_chat_list.pack(fill=tk.BOTH, expand=True)

        # ChatWidget(self.recent_chat_list.scrollable_frame, "test", '123')
        self._reload_chat_list(self.recent_chat_list.scrollable_frame)

    def chat_thread_worker(self, user_message: str, chat_id: str):
        try:
            temp_db = ChatDB()
            creating_new_chat: bool = (chat_id == None)
            # Clear the input field
            self.root.after(0, self.chat_input.delete("1.0", tk.END))

            # Display user's message in the chat window
            if self.current_chat_id == chat_id:
                TextBubble(
                    parent=self.chat_content.scrollable_frame,
                    text=user_message,
                    from_user=True
                )

                # Display loading message
                tk.Label(self.chat_content.scrollable_frame, 
                        text="Waiting for Clerkbot response...", 
                        font=("Arial", 10)).pack(anchor='w',pady=5)

                self.chat_content.scroll_to_bottom()

            bot_response: str = None

            # Create new chat session record in database if chat_id is none (meaning this is a new chat)
            if creating_new_chat:
                chat_id = temp_db.create_chatsession("Pending chat title...")
                new_chat_session = (chat_id, "Pending chat title...")
                self.chat_session_list.insert(0, new_chat_session) # insert at the beginning of the list
                self._reload_chat_list(self.recent_chat_list.scrollable_frame)

                if self.current_chat_id == None: # The user is staying on the newly created chat
                    self.current_chat_id = chat_id

            self.chat_inprogress_list.append(chat_id)
            if self.current_chat_id in self.chat_inprogress_list:
                self.chat_input.config(state=tk.DISABLED)
                self.input_btn.config(state=tk.DISABLED)

            # Create new inquiry record (with no response)
            new_inquiry = Inquiry(user_message, bot_response, chat_id)
            new_inquiry_id = temp_db.create_inquiry(new_inquiry)

            # Update file embeddings in the target directory (for RAG)
            get_file_mean_embeddings(self.selected_path)

            # Retrieve file chunks from ChromaDB based on the user's message and the tracked files
            file_id_list = [get_file_id(os.path.join(self.selected_path, f)) 
                            for f in os.listdir(self.selected_path) 
                            if os.path.isfile(os.path.join(self.selected_path, f))]
            retrieved_context = temp_db.retrieve_file_chunk(user_message, file_id_list)

            unload_embedding_model() # Safely unload embedding model

            new_title = []
            create_title_thread = threading.Thread(
                target=get_new_chat_title, 
                args=(user_message, new_title, 
                      creating_new_chat, 
                      self.use_online_llm.get())
            )
            create_title_thread.start()
            # Get a response from ClerkBot
            bot_response = get_chat_response(
                user_message, 
                retrieved_context, 
                self.selected_path, 
                online=self.use_online_llm.get()
                )

            create_title_thread.join() # wait for create title thread to finish before continuing

            # Update chat session's name if user input in a newly created chat
            if creating_new_chat:
                temp_db.update_chatsession_name_by_id(chat_id, new_title[0])
                # update chat_session_list
                for i, (chat_id_iter, _) in enumerate(self.chat_session_list):
                    if chat_id_iter == chat_id:
                        self.chat_session_list[i] = (chat_id, new_title[0])
                        break
                self._reload_chat_list(self.recent_chat_list.scrollable_frame)

            # Handle LLM response
            if bot_response:
                temp_db.update_inquiry_response_by_id(new_inquiry_id, bot_response)
                if self.current_chat_id == chat_id:
                    # Remove loading message
                    self.chat_content.scrollable_frame.winfo_children()[-1].destroy()
                    # Append and display LLM Response in chat window
                    TextBubble(
                        parent=self.chat_content.scrollable_frame,
                        text=bot_response,
                        from_user=False
                    )
                    self.chat_content.scroll_to_bottom()

        except Exception as e:
            if self.current_chat_id == chat_id:
                self.chat_content.scrollable_frame.winfo_children()[-1].destroy()
                TextBubble(
                    parent=self.chat_content.scrollable_frame,
                    text=f"An error occurred: {e}",
                    from_user=False
                )
                self.chat_content.scroll_to_bottom()
        finally:
            # lift restriction on chat_input and input_btn
            if self.current_chat_id in self.chat_inprogress_list:
                self.chat_input.config(state=tk.NORMAL)
                self.input_btn.config(state=tk.NORMAL)
                self.chat_inprogress_list.remove(chat_id)
            temp_db.close()
        
    def _build_chat_window(self, parent_container):
        """Build the chat window area"""
        self.chat_frame = ttk.Frame(parent_container)
        self.chat_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Pack bottom input frame FIRST so it anchors to the bottom space
        user_input_frame = ttk.Frame(self.chat_frame)
        user_input_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        def send_message():
            user_message = self.chat_input.get("1.0", tk.END).strip()
            if not (user_message and self.selected_path):
                return
            self.executor.submit(self.chat_thread_worker, user_message, self.current_chat_id)

        self.input_btn = tk.Button(
            user_input_frame, 
            text="⌯⌲", 
            command=send_message, 
            font=("Arial", 14),
            cursor="hand2"
            )
        self.input_btn.pack(side=tk.RIGHT)

        self.chat_input = tk.Text(
                    user_input_frame, 
                    height=3, 
                    padx=10,
                    pady=10,
                    font=("Arial", 11)
                    )
        self.chat_input.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10)
        )

        self.chat_content = ScrollableFrame(self.chat_frame)
        self.chat_content.pack(fill=tk.BOTH, expand=True)

        # Starting message
        TextBubble(
            parent=self.chat_content.scrollable_frame,
            text="Ask me anything about what's in the folder!",
            from_user = False
        )

    def _open_new_chat(self):
        # set state variables
        self.current_chat_id = None
        # clear chat window
        self.chat_content.clear_content()
        # Add starting message
        TextBubble(
            parent=self.chat_content.scrollable_frame,
            text="Ask me anything about what's in the folder!",
            from_user = False
        )
        # config chat_input and input_btn
        self.chat_input.config(state=tk.NORMAL)
        self.input_btn.config(state=tk.NORMAL)

    def _load_chatsession_chats(self, chatsession_id: str):
        '''Load all previous conversations from chatsession to chat window'''
        if self.current_chat_id == chatsession_id:
            return
        
        self.current_chat_id = chatsession_id
        # clear chat window
        self.chat_content.clear_content()

        prev_convs = self.database.get_inquiries_from_session(chatsession_id)
        # Add starting message
        TextBubble(
            parent=self.chat_content.scrollable_frame,
            text="Ask me anything about what's in the folder!",
            from_user = False
        )
        for user_message, bot_message in prev_convs:
            TextBubble(
                parent=self.chat_content.scrollable_frame,
                text=user_message,
                from_user = True
            )
            if bot_message:
                TextBubble(
                    parent=self.chat_content.scrollable_frame,
                    text=bot_message,
                    from_user = False
                )  
            else:
                tk.Label(self.chat_content.scrollable_frame, 
                                        text="Waiting for Clerkbot response...", 
                                        font=("Arial", 10)).pack(anchor='w',pady=5)
        self.chat_content.scroll_to_bottom()

        if chatsession_id in self.chat_inprogress_list:
            self.chat_input.config(state=tk.DISABLED)
            self.input_btn.config(state=tk.DISABLED)
        else:
            self.chat_input.config(state=tk.NORMAL)
            self.input_btn.config(state=tk.NORMAL)

    def _delete_chatsession(self, chatsession_id:str):
        if self.current_chat_id == chatsession_id:
            # user is currently accessing the chat session that is about to be deleted
            self._open_new_chat()
        self.database.delete_chatsession_by_id(chatsession_id)

        # Delete the chat session (id, name) from self.chat_session_list
        for i, (chat_id, _) in enumerate(self.chat_session_list):
            if chat_id == chatsession_id:
                del self.chat_session_list[i]
                break

        # reload recent chat list in sidebar
        self._reload_chat_list(self.recent_chat_list.scrollable_frame)


    def _build_clerkbot_page(self, parent_container):
            """Build ClerkBot (Chatbot) page"""
            self.page2 = ttk.Frame(parent_container, relief="groove", padding=10)
            self.page2.grid(row=0, column=0, sticky="nsew")

            self.sidebar_visible = False

            # --- Top Header Bar ---
            top_bar = ttk.Frame(self.page2)
            top_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

            # Menu Toggle Button
            self.sidebar_btn = ttk.Button(
                top_bar, text="☰", width=8, command=self._toggle_sidebar, cursor="hand2"
            )
            self.sidebar_btn.pack(side=tk.LEFT, padx=(0, 10))

            # Header Info Container
            header_info_frame = ttk.Frame(top_bar)
            header_info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Title Row containing "Ask ClerkBot!" and the toggle button
            title_row = ttk.Frame(header_info_frame)
            title_row.pack(side=tk.TOP, anchor='w')

            clerkbot_label = tk.Label(
                title_row, text="Ask ClerkBot!", font=("Arial", 16, "bold")
            )
            clerkbot_label.pack(side=tk.LEFT)

            # LLM Toggle Button (placed to the right of clerkbot_label)
            def toggle_llm():
                self.use_online_llm.set(not self.use_online_llm.get())

            self.llm_toggle_btn = tk.Button(
                title_row,
                text= "ᯤOnline" if self.use_online_llm.get() else "💻Local",
                borderwidth=0,          # Removes the standard button border
                relief="flat",          # Ensures the button style is flat
                highlightthickness=0,   # No extra highlight border from appearing when the button is clicked 
                command=toggle_llm,
                width=7,
                font=("Arial", 11),
                cursor="hand2",
                bg="blue" if self.use_online_llm.get() else "green"
            )
            self.llm_toggle_btn.pack(side=tk.LEFT, padx=(10, 0))

            # Update button text automatically whenever self.use_online_llm changes
            def update_llm_btn_text(*args):
                self.llm_toggle_btn.config(
                    text= "ᯤOnline" if self.use_online_llm.get() else "💻Local",
                    background="blue" if self.use_online_llm.get() else "green"
                )

            self.use_online_llm.trace_add("write", update_llm_btn_text)

            self.tracking_dir_label = tk.Label(
                header_info_frame, text="Folder: None", font=("Arial", 9)
            )
            self.tracking_dir_label.pack(anchor='w')

            # --- Body Area Container ---
            body_container = ttk.Frame(self.page2)
            body_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # --- Main Content Area - Chat Zone ---
            self._build_chat_window(body_container)

            # --- Overlay Sidebar Frame ---
            self._build_sidebar(body_container)

    def _toggle_sidebar(self):
        """Show or hide the sidebar overlay using place()"""
        if self.sidebar_visible:
            self.sidebar_frame.place_forget()
            self.sidebar_visible = False
        else:
            self.sidebar_frame.place(x=-10, y=0, relheight=1.0, width=250)
            self.sidebar_frame.lift()
            self.sidebar_visible = True
            
    def _build_right_frame(self):
        # COMPONENT 2: RIGHT FRAME (STATUS, USER CONTROLS, FILE SORT AND CLERKBOT)
        right_container = ttk.Frame(self.root, width=350)
        right_container.grid_propagate(False)

        right_container.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        right_container.rowconfigure(1, weight=1)
        right_container.columnconfigure(0, weight=1)

        # --- Top Navigation Frame ---
        nav_frame = ttk.Frame(right_container)
        nav_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        btn_page1 = ttk.Button(
            nav_frame, text="File Sort", command=lambda: self.page1.tkraise(), cursor="hand2"
        )
        btn_page1.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        btn_page2 = ttk.Button(
            nav_frame, text="ClerkBot", command=lambda: self.page2.tkraise(), cursor="hand2"
        )
        btn_page2.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # --- Pages Container ---
        pages_container = ttk.Frame(right_container)
        pages_container.grid(row=1, column=0, sticky="nsew")
        pages_container.rowconfigure(0, weight=1)
        pages_container.columnconfigure(0, weight=1)

        # --- Page 1: File Sort ---
        self._build_sorting_page(pages_container)

        # --- Page 2: ClerkBot Barebones ---
        self._build_clerkbot_page(pages_container)

        self.page1.tkraise()

    # --- Directory and File Helper Methods ---
    def _display_files_in_dir(self, selected_dir: str):
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
        self.tracking_dir_label.config(text=f"Folder: {re.split(r'[/\\]', dir_path)[-1]}")
        self._display_files_in_dir(dir_path)

    def browse_folder(self):
        selected_dir = filedialog.askdirectory(title="Select a Directory")
        if selected_dir:
            self.selected_path = selected_dir
            self._goto_folder(selected_dir)

    def prev_folder(self):
        if not self.path_entry.get():
            return
        previous_dir = os.path.dirname(self.path_entry.get())
        if previous_dir: self._goto_folder(previous_dir)

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
            self._display_files_in_dir(current_dir)

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

    def open_confirm_popup(self, directory: str, groups: dict) -> bool:
        """Display where the files will be sorted and allow user to confirm or reject."""
        popup = tk.Toplevel(self.root)
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

        confirm_btn = tk.Button(btn_frame, text="Confirm", width=12, command=confirm_sorting, cursor="hand2")
        confirm_btn.pack(side=tk.LEFT, padx=40)

        reject_btn = tk.Button(btn_frame, text="Reject", width=12, command=reject_sorting, cursor="hand2")
        reject_btn.pack(side=tk.RIGHT, padx=40)

        sorting_info = scrolledtext.ScrolledText(popup, height=20, width=65, font=("Arial", 10), wrap=tk.WORD)
        sorting_info.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)

        def print_line(msg: str):
            sorting_info.insert(tk.END, f"{msg}\n")

        print_groups(
            groups,
            sort_into_existing=not self.generate_folder.get(),
            print_to_widget=print_line
        )

        sorting_info.config(state="disabled") 

        popup.grab_set()
        popup.wait_window(popup)

        return result

    # --- Thread-Safe Updates & Worker Logic ---
    # def update_status_console(self, text):
    #     self.root.after(0, lambda: self.sorting_job_widget.set_status(text))

    # def update_progress_bar(self, val):
    #     self.root.after(0, lambda: self.sorting_job_widget.set_progress(val))

    def check_cancel_all_status(self):
        return self.is_cancelled_all

    def sorting_thread_worker(self, sorting_widget: SortingJobWidget):
        start_time = time.perf_counter()
        try:
            # Step 0: Basic Thread Setup.
            current_dir = sorting_widget.target_dir
            recursive_config: bool = sorting_widget.recursive
            online_config: bool = sorting_widget.online
            sort_into_existing_config: bool = sorting_widget.sort_into_existing

            # --- Helper methods to safely update Tkinter from background threads ---
            def safe_set_status(text):
                self.root.after(0, lambda: sorting_widget.set_status(text))

            def safe_set_progress(val):
                self.root.after(0, lambda: sorting_widget.set_progress(val))

            # Step 0.1: Append Sorting Job Widget to Sorting Job List ScollableFrame
            safe_set_progress(0)
            safe_set_status("Initializing semantic indexing system...")

            # Step 0.2: Setup cancellation function
            def check_cancel_status() -> bool:
                cancel_status: bool = self.is_cancelled_all or sorting_widget.is_cancel
                if cancel_status:
                    safe_set_status("Cancelling sorting...")
                    safe_set_progress(0)
                return cancel_status

            # Step 1: Semantic Clustering
            if self.generate_folder.get():
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

            # Step 2: Dynamic Auto-Labelling
            if groups:
                if self.generate_folder.get():
                    groups = AutoLabelClusters(
                        current_dir,
                        groups,
                        online=online_config,
                        status_callback=safe_set_status,
                        progress_callback=safe_set_progress,
                        check_cancel=check_cancel_status,
                    )

                # Step 3: Organise Files
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

                # Calculate sorting time
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
            safe_set_progress(0 if self.is_cancelled_all or sorting_widget.is_cancel else 100)
            if self.path_entry.get() == current_dir: # the user is in browsing the current sorted directory
                self._display_files_in_dir(current_dir) # Refresh the current directory

            # Delay 5 seconds and destroy the widget
            time.sleep(5)
            sorting_widget.destroy()
            self.sorting_folders.remove(current_dir)
            if len(self.sorting_folders) == 0: # no folders are currently in sorting progress
                self.cancel_all_btn.config(state=tk.DISABLED)


    def semantic_file_sort(self):
        current_dir = self.path_entry.get()
        if not current_dir or not os.path.exists(current_dir):
            print("Please select a valid directory first.")
            return

        if current_dir in self.sorting_folders:
            return

        self.is_cancelled_all = False
        self.cancel_all_btn.config(state=tk.NORMAL)
        self.sorting_folders.append(current_dir)
        
        # --- Create and pack the widget on the MAIN thread before starting the worker ---
        recursive_config: bool = not self.singular_sorting.get()
        online_config: bool = self.use_online_llm.get()
        sort_into_existing_config: bool = not self.generate_folder.get()
        
        sorting_widget = SortingJobWidget(
            self.sorting_jobs_list.scrollable_frame,
            target_dir=current_dir,
            recursive=recursive_config,
            online=online_config,
            sort_into_existing=sort_into_existing_config
        )
        sorting_widget.pack(pady=5, fill=tk.X)
    
        # Pass the pre-created sorting_widget into the thread worker
        self.executor.submit(self.sorting_thread_worker, sorting_widget)

    def cancel_file_sort(self):
        self.is_cancelled_all = True

    def run(self):
        self.root.mainloop()