"""
Module: mini.gui
Description: GUI-related modules and configurations.
"""

import tkinter as tk
from tkinter import filedialog

from sentencepiece import SentencePieceProcessor


class MiniGUI:
    def __init__(self, title: str):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)
        self.processor = None
        self.processor_path = None
        self.create_file_menu()
        self.create_text_area()
        self.create_tokenizer_menu()
        self.create_status_bar()

    def create_text_area(self):
        self.text_area = tk.Text(self.root, wrap=tk.WORD, font=("Noto Sans Mono", 10))
        self.text_area.pack(expand=tk.YES, fill=tk.BOTH)

    def create_status_bar(self):
        self.status_bar = tk.Label(
            self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar.config(text="Ready")

    def create_file_menu(self):
        file_menu = tk.Menu(self.menu, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        self.menu.add_cascade(label="File", menu=file_menu)

    def select_model(self):
        self.processor_path = filedialog.askopenfilename()
        if self.processor_path:
            self.processor = SentencePieceProcessor(model_file=self.processor_path)
            self.status_bar.config(text=f"Model loaded: {self.processor_path}")
        else:
            self.status_bar.config(text="No model selected")
            self.processor = None
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, "No model selected. Please load a model.")

    def encode_text(self):
        if not self.processor:
            self.status_bar.config(text="No model selected. Please load a model.")
            return
        text = self.text_area.get(1.0, tk.END).strip()
        if not text:
            self.status_bar.config(text="No text to encode.")
            return
        text = self.text_area.get(1.0, tk.END).strip()
        tokens = self.processor.encode(text)
        # Convert list of token ids to a string of tokens separated by commas
        token_str = ", ".join([str(token) for token in tokens])
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, token_str)
        self.status_bar.config(text=f"Encoded {len(tokens)} tokens")

    def decode_text(self):
        if not self.processor:
            self.status_bar.config(text="No model selected. Please load a model.")
            return
        text = self.text_area.get(1.0, tk.END).strip()
        if not text:
            self.status_bar.config(text="No text to decode.")
            return
        # Convert string of tokens separated by commas to a list of token ids
        text = [int(token) for token in text.split(",")]
        decoded_text = self.processor.decode(text)
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, decoded_text + "\n")
        self.status_bar.config(text=f"Decoded {len(text)} tokens")

    def create_tokenizer_menu(self):
        tokenizer_menu = tk.Menu(self.menu, tearoff=0)
        tokenizer_menu.add_command(label="Open Model", command=self.select_model)
        tokenizer_menu.add_separator()
        tokenizer_menu.add_command(label="Encode Text", command=self.encode_text)
        tokenizer_menu.add_command(label="Decode Text", command=self.decode_text)
        self.menu.add_cascade(label="Tokenizer", menu=tokenizer_menu)

    def create_train_menu(self):
        pass

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            print(f"Selected file: {file_path}")
            self.text_area.delete(1.0, tk.END)
            with open(file_path, "r") as file:
                content = file.read()
                self.text_area.insert(tk.END, content)
            self.file_path = file_path
            self.status_bar.config(text=f"File: {file_path}")
        else:
            print("No file selected.")
        self.status_bar.config(text=f"File: {file_path}")

    def save_file(self):
        file_path = filedialog.asksaveasfilename()
        if file_path:
            print(f"Selected save location: {file_path}")

    def exit_app(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = MiniGUI("Mini GUI App")
    app.run()
