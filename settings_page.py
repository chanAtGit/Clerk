import tkinter as tk
from tkinter import ttk
from huggingface_hub import login, logout


class SettingsPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, relief="groove", padding=10)
        self.app = app
        self._build_ui()

    def _build_ui(self):
        settings_label = tk.Label(
            self, text="Settings", font=("Arial", 16, "bold")
        )
        settings_label.pack(anchor="center", pady=(0, 10))

        tk.Label(
            self, text="Hugging Face Token", font=("Arial", 11)
        ).pack(anchor="w", pady=(0, 5))

        token_entry = tk.Entry(self, show="*", font=("Arial", 12))
        token_entry.pack(anchor="w", fill=tk.X)
        if self.app.huggingface_token:
            token_entry.insert(0, self.app.huggingface_token)

        def save_settings():
            logout()
            try:
                login(token=token_entry.get())
                self.app.database.update_huggingface_token(token_entry.get())
                self.app.huggingface_token = token_entry.get()
            except Exception as e:
                print(f"Error encountered while saving settings: {e}")

        settings_save_btn = tk.Button(
            self, 
            text="Save", 
            font=("Arial", 11), 
            command=save_settings,
            cursor="hand2"
        )
        settings_save_btn.pack(anchor="center", pady=10)