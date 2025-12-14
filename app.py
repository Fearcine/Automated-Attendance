import tkinter as tk
from recognize import run_recognition
from show_attendance import show_today

def start_recognition():
    run_recognition()

def show_att():
    show_today()

root = tk.Tk()
root.title("Attendance App")
root.geometry("1000x1000")

tk.Button(root, text="Start Attendance", command=start_recognition, height=2).pack(pady=10)
tk.Button(root, text="Show Today's Attendance", command=show_att, height=2).pack(pady=10)
tk.Button(root, text="Exit", command=root.destroy, height=2).pack(pady=10)

root.mainloop()
