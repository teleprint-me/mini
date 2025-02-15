"""
Module: mini.gui
Description: GUI-related modules and configurations.
"""

import os

import dearpygui.dearpygui as dpg
from sentencepiece import SentencePieceProcessor


class MiniGUI:
    def __init__(self):
        self.file_path = None
        self.processor = None
        self.processor_path = None
        self.setup_ui()

    def setup_ui(self):
        """Creates the UI elements for the MiniGUI application."""
        dpg.create_context()

        with dpg.font_registry():
            with dpg.font(
                "/usr/share/fonts/noto/NotoSansMono-Regular.ttf", 14
            ) as default_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                dpg.bind_font(default_font)

        with dpg.window(label="Mini GUI", width=800, height=600):
            self.create_menu_bar()
            self.create_text_area()
            self.create_status_bar()

        dpg.create_viewport(title="Mini GUI App", width=800, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def create_menu_bar(self):
        """Creates the menu bar with File and Tokenizer options."""
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="New", callback=self.new_file)
                dpg.add_menu_item(label="Open", callback=self.open_file)
                dpg.add_menu_item(label="Save", callback=self.save_file)
                dpg.add_menu_item(label="Save As", callback=self.save_as_file)
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())

            with dpg.menu(label="Tokenizer"):
                dpg.add_menu_item(label="Load Model", callback=self.select_model)
                dpg.add_menu_item(label="Encode Text", callback=self.encode_text)
                dpg.add_menu_item(label="Decode Text", callback=self.decode_text)

    def create_text_area(self):
        """Creates the text area for input/output."""
        with dpg.group(horizontal=False):
            dpg.add_text("Text Editor:")
            dpg.add_input_text(tag="text_area", multiline=True, height=400, width=-1)

    def create_status_bar(self):
        """Creates a status bar at the bottom of the window."""
        dpg.add_text("Status: Ready", tag="status_bar")

    def set_status(self, message):
        """Updates the status bar with a message."""
        dpg.set_value("status_bar", f"Status: {message}")

    def new_file(self):
        """Clears the text area."""
        dpg.set_value("text_area", "")
        self.file_path = None
        self.set_status("New file created.")

    def open_file(self):
        """Opens a file and loads its contents into the text area."""
        file_path = dpg.open_file_dialog()
        if file_path:
            with open(file_path[0], "r") as file:
                dpg.set_value("text_area", file.read())
            self.file_path = file_path[0]
            self.set_status(f"Opened file: {self.file_path}")

    def save_file(self):
        """Saves the current file."""
        if self.file_path:
            with open(self.file_path, "w") as file:
                file.write(dpg.get_value("text_area"))
            self.set_status(f"Saved: {self.file_path}")
        else:
            self.save_as_file()

    def save_as_file(self):
        """Saves the file with a new name."""
        file_path = dpg.save_file_dialog()
        if file_path:
            with open(file_path[0], "w") as file:
                file.write(dpg.get_value("text_area"))
            self.file_path = file_path[0]
            self.set_status(f"Saved as: {self.file_path}")

    def select_model(self):
        """Loads a SentencePiece model."""
        file_path = dpg.open_file_dialog()
        if file_path:
            self.processor_path = file_path[0]
            self.processor = SentencePieceProcessor(model_file=self.processor_path)
            self.set_status(f"Loaded model: {self.processor_path}")

    def encode_text(self):
        """Encodes the text using the loaded tokenizer."""
        if not self.processor:
            self.set_status("No model selected.")
            return

        text = dpg.get_value("text_area").strip()
        if not text:
            self.set_status("No text to encode.")
            return

        tokens = self.processor.encode(text)
        token_str = ", ".join(map(str, tokens))
        dpg.set_value("text_area", token_str)
        self.set_status(f"Encoded {len(tokens)} tokens.")

    def decode_text(self):
        """Decodes tokenized text."""
        if not self.processor:
            self.set_status("No model selected.")
            return

        text = dpg.get_value("text_area").strip()
        if not text:
            self.set_status("No text to decode.")
            return

        try:
            token_list = list(map(int, text.split(",")))
            decoded_text = self.processor.decode(token_list)
            dpg.set_value("text_area", decoded_text)
            self.set_status(f"Decoded {len(token_list)} tokens.")
        except ValueError:
            self.set_status("Invalid token format.")


if __name__ == "__main__":
    MiniGUI()
