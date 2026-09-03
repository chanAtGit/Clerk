import os
import threading
import tkinter as tk
from tkinter import ttk

from widgets import ChatWidget, TextBubble, ScrollableFrame
from chat_functions import get_chat_response, get_new_chat_title
from file_embeddings import get_file_id, get_file_mean_embeddings
from model_commons import unload_embedding_model
from database import ChatDB, Inquiry


class ClerkBotPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, relief="groove", padding=10)
        self.app = app
        self.sidebar_visible = False
        self._build_ui()

    def _build_ui(self):
        top_bar = ttk.Frame(self)
        top_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.sidebar_btn = ttk.Button(
            top_bar, text="☰", width=8, command=self._toggle_sidebar, cursor="hand2"
        )
        self.sidebar_btn.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        header_info_frame = ttk.Frame(top_bar)
        header_info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        title_row = ttk.Frame(header_info_frame)
        title_row.pack(side=tk.TOP, fill=tk.X)

        clerkbot_label = tk.Label(
            title_row, text="ClerkBot", font=("Arial", 16, "bold")
        )
        clerkbot_label.pack(side=tk.LEFT)

        def toggle_llm():
            self.app.use_online_llm.set(not self.app.use_online_llm.get())

        self.llm_toggle_btn = tk.Button(
            title_row,
            text="ᯤOnline" if self.app.use_online_llm.get() else "💻Local",
            borderwidth=0,
            relief="flat",
            highlightthickness=0,
            command=toggle_llm,
            width=7,
            font=("Arial", 11),
            cursor="hand2",
            bg="blue" if self.app.use_online_llm.get() else "green",
            fg="white"
        )
        self.llm_toggle_btn.pack(side=tk.RIGHT, padx=(0, 10))

        def update_llm_btn_text(*args):
            self.llm_toggle_btn.config(
                text="ᯤOnline" if self.app.use_online_llm.get() else "💻Local",
                background="blue" if self.app.use_online_llm.get() else "green",
            )

        self.app.use_online_llm.trace_add("write", update_llm_btn_text)

        self.tracking_dir_label = tk.Label(
            header_info_frame, text="Folder: None", font=("Arial", 9)
        )
        self.tracking_dir_label.pack(anchor='w')

        body_container = ttk.Frame(self)
        body_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._build_chat_window(body_container)
        self._build_sidebar(body_container)

    def update_tracking_dir(self, dir_path: str):
        self.tracking_dir_label.config(text=f"Folder: {os.path.basename(dir_path)}")

    def _build_chat_window(self, parent_container):
        self.chat_frame = ttk.Frame(parent_container)
        self.chat_frame.pack(fill=tk.BOTH, expand=True)

        user_input_frame = ttk.Frame(self.chat_frame)
        user_input_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        def send_message():
            user_message = self.chat_input.get("1.0", tk.END).strip()
            if not (user_message and self.app.selected_path):
                return
            self.app.executor.submit(
                self.chat_thread_worker, user_message, self.app.current_chat_id, self.app.selected_path
            )

        self.input_btn = tk.Button(
            user_input_frame, 
            text="⌯⌲", 
            command=send_message,
            bg="gray",
            fg="white",
            font=("Arial", 15),
            cursor="hand2"
        )
        self.input_btn.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_input = tk.Text(
            user_input_frame, 
            height=3, 
            padx=10,
            pady=10,
            wrap=tk.WORD,
            font=("Arial", 10)
        )
        self.chat_input.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10)
        )

        self.chat_content = ScrollableFrame(self.chat_frame)
        self.chat_content.pack(fill=tk.BOTH, expand=True)

        TextBubble(
            parent=self.chat_content.scrollable_frame,
            text="Ask me anything about what's in the folder!",
            from_user=False
        )

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
        sidebar_title.pack(anchor="w", pady=(20, 0))

        self.recent_chat_list = ScrollableFrame(self.sidebar_frame)
        self.recent_chat_list.pack(fill=tk.BOTH, expand=True)

        self._reload_chat_list(self.recent_chat_list.scrollable_frame)

    def _toggle_sidebar(self):
        if self.sidebar_visible:
            self.sidebar_frame.place_forget()
            self.sidebar_visible = False
        else:
            self.sidebar_frame.place(x=-10, y=0, relheight=1.0, width=250)
            self.sidebar_frame.lift()
            self.sidebar_visible = True

    def _reload_chat_list(self, parent_container):
        for widget in parent_container.winfo_children():
            widget.destroy()

        for chat_id, chat_name in self.app.chat_session_list:
            ChatWidget(parent_container, chat_name, chat_id, 
                       go_to_func=self._load_chatsession_chats,
                       delete_func=self._delete_chatsession)

    def chat_thread_worker(self, user_message: str, chat_id: str, tar_dir: str):
        try:
            temp_db = ChatDB()
            creating_new_chat: bool = (chat_id is None)
            self.app.root.after(0, self.chat_input.delete("1.0", tk.END))

            if self.app.current_chat_id == chat_id:
                latest_dir: str = temp_db.get_latest_inq_dir_from_session(chat_id) 
                if latest_dir is None or latest_dir != os.path.basename(tar_dir):
                    tk.Label(self.chat_content.scrollable_frame, 
                        text=f"== Tracking: {os.path.basename(tar_dir)} ==", 
                        font=("Arial", 10), 
                        fg="green").pack(anchor='center', pady=5)

                TextBubble(
                    parent=self.chat_content.scrollable_frame,
                    text=user_message,
                    from_user=True
                )

                tk.Label(self.chat_content.scrollable_frame, 
                        text="Waiting for ClerkBot response...", 
                        font=("Arial", 10),
                        fg="orange").pack(anchor='w', pady=5, padx=5)

                self.chat_content.scroll_to_bottom()

            bot_response: str = None

            if creating_new_chat:
                chat_id = temp_db.create_chatsession("Pending chat title...")
                new_chat_session = (chat_id, "Pending chat title...")
                self.app.chat_session_list.insert(0, new_chat_session)
                self._reload_chat_list(self.recent_chat_list.scrollable_frame)

                if self.app.current_chat_id is None:
                    self.app.current_chat_id = chat_id

            self.app.chat_inprogress_list.append(chat_id)
            if self.app.current_chat_id in self.app.chat_inprogress_list:
                self.chat_input.config(state=tk.DISABLED)
                self.input_btn.config(state=tk.DISABLED)

            history: list = temp_db.get_inquiries_for_llm_history(chat_id, tar_dir)

            new_inquiry = Inquiry(user_message, bot_response, chat_id, tar_dir)
            new_inquiry_id: str = temp_db.create_inquiry(new_inquiry)

            get_file_mean_embeddings(self.app.selected_path)

            file_id_list = [get_file_id(os.path.join(self.app.selected_path, f)) 
                            for f in os.listdir(self.app.selected_path) 
                            if os.path.isfile(os.path.join(self.app.selected_path, f))]
            retrieved_context = temp_db.retrieve_file_chunk(user_message, file_id_list)

            unload_embedding_model()

            new_title: list = []
            create_title_thread = threading.Thread(
                target=get_new_chat_title, 
                args=(user_message, new_title, 
                      creating_new_chat, 
                      self.app.use_online_llm.get())
            )
            create_title_thread.start()

            bot_response = get_chat_response(
                user_message, 
                retrieved_context,
                history=history, 
                dir_path=self.app.selected_path, 
                online=self.app.use_online_llm.get()
            )

            create_title_thread.join()

            for i, (chat_id_iter, _) in enumerate(self.app.chat_session_list):
                if chat_id_iter == chat_id:
                    if creating_new_chat:
                        temp_db.update_chatsession_name_by_id(chat_id, new_title[0])
                        self.app.chat_session_list[i] = (chat_id, new_title[0])
                    self.app.chat_session_list.insert(0, self.app.chat_session_list.pop(i))
                    break

            self._reload_chat_list(self.recent_chat_list.scrollable_frame)

            if bot_response:
                temp_db.update_inquiry_response_by_id(new_inquiry_id, bot_response)

                if self.app.current_chat_id == chat_id:
                    self.chat_content.scrollable_frame.winfo_children()[-1].destroy()
                    TextBubble(
                        parent=self.chat_content.scrollable_frame,
                        text=bot_response,
                        from_user=False
                    )
                    self.chat_content.scroll_to_bottom()

        except Exception as e:
            print(f"An error occurred while chatting: {e}")

        finally:
            if self.app.current_chat_id in self.app.chat_inprogress_list:
                self.chat_input.config(state=tk.NORMAL)
                self.input_btn.config(state=tk.NORMAL)
            self.app.chat_inprogress_list.remove(chat_id)
            temp_db.close()
            del temp_db

    def _open_new_chat(self):
        self.app.current_chat_id = None
        self.chat_content.clear_content()
        TextBubble(
            parent=self.chat_content.scrollable_frame,
            text="Ask me anything about what's in the folder!",
            from_user=False
        )
        self.chat_input.config(state=tk.NORMAL)
        self.input_btn.config(state=tk.NORMAL)

    def _load_chatsession_chats(self, chatsession_id: str):
        if self.app.current_chat_id == chatsession_id:
            return
        
        self.app.current_chat_id = chatsession_id
        self.chat_content.clear_content()

        prev_convs: list = self.app.database.get_inquiries_from_session(chatsession_id)
        TextBubble(
            parent=self.chat_content.scrollable_frame,
            text="Ask me anything about what's in the folder!",
            from_user=False
        )

        prev_dir_name: str = None
        for id, user_message, bot_message, dir_name in prev_convs:
            if bot_message is None and chatsession_id not in self.app.chat_inprogress_list:
                self.app.database.delete_inquiry_by_id(id)
                continue

            if prev_dir_name != dir_name:
                tk.Label(self.chat_content.scrollable_frame, 
                    text=f"== Tracking: {dir_name} ==", 
                    font=("Arial", 10), 
                    fg="green").pack(anchor='center', pady=5)
                prev_dir_name = dir_name

            TextBubble(
                parent=self.chat_content.scrollable_frame,
                text=user_message,
                from_user=True
            )
            if bot_message:
                TextBubble(
                    parent=self.chat_content.scrollable_frame,
                    text=bot_message,
                    from_user=False
                )  
            else:
                tk.Label(
                    self.chat_content.scrollable_frame, 
                    text="Waiting for ClerkBot response...", 
                    font=("Arial", 10),
                    fg="orange").pack(anchor='w', pady=5, padx=5)

        self.chat_content.scroll_to_bottom()

        if chatsession_id in self.app.chat_inprogress_list:
            self.chat_input.config(state=tk.DISABLED)
            self.input_btn.config(state=tk.DISABLED)
        else:
            self.chat_input.config(state=tk.NORMAL)
            self.input_btn.config(state=tk.NORMAL)

    def _delete_chatsession(self, chatsession_id: str):
        if self.app.current_chat_id == chatsession_id:
            self._open_new_chat()
        self.app.database.delete_chatsession_by_id(chatsession_id)

        for i, (chat_id, _) in enumerate(self.app.chat_session_list):
            if chat_id == chatsession_id:
                del self.app.chat_session_list[i]
                break

        self._reload_chat_list(self.recent_chat_list.scrollable_frame)