import tkinter as tk
from file_sort import file_sort_init
from gui import GUI

if __name__ == "__main__":
    file_sort_init()
    root = tk.Tk()
    app = GUI(root)
    app.run()