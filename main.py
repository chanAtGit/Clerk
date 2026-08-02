import tkinter as tk
from model_commons import models_init
from file_embeddings import embeddings_init
from gui import GUI

if __name__ == "__main__":
    models_init()
    embeddings_init()
    root = tk.Tk()
    app = GUI(root)
    app.run()