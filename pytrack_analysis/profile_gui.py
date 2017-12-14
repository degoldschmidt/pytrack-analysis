import tkinter as tk
from tkinter import messagebox, filedialog

"""
profile_gui.py
AUTHOR: degoldschmidt
DATE: 14/12/2017

Contains GUI class for setting up analysis profile.
"""

class GUI(tk.Frame):
    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master, **kwargs)

        self.var = tk.StringVar()
        entry = tk.Entry(self, textvariable=self.var)
        entry.pack()

        btn = tk.Button(self, text='read', command=self.read_entry)
        btn.pack()

        btn = tk.Button(self, text='write', command=self.write_entry)
        btn.pack()

    def read_entry(self):
        print(self.var.get())

    def write_entry(self):
        self.var.set("hello world")

def main():
    root = tk.Tk()
    root.geometry('200x200')
    win = GUI(root)
    win.pack()
    root.mainloop()

if __name__ == '__main__':
    main()
