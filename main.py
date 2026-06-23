import tkinter as tk

# 1. Create the main application window
root = tk.Tk()
root.title("Overseer")
root.geometry("600x450")  # Width x Height in pixels

# 2. Add a simple label
label = tk.Label(root, text="Hello, Tkinter!", font=("Arial", 14))
label.pack(pady=20)  # Add padding and place it in the window

# 3. Add an exit button
button = tk.Button(root, text="Quit", command=root.destroy)
button.pack()

# 4. Start the infinite event loop
root.mainloop()
