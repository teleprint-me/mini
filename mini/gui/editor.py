"""
Module: mini.gui
Description: GUI-related modules and configurations.
"""

import dearpygui.dearpygui as dpg
from sentencepiece import SentencePieceProcessor

from mini.common.logger import get_logger
from mini.gui.fonts.fuzzer import FontFuzzer
from mini.gui.window_manager import WindowManager


class MiniGUI:
    def __init__(self):
        self.file_path = None
        self.font_path = FontFuzzer.locate_font("Noto Sans Mono")
        self.processor = None
        self.processor_path = None
        self.current_font = None
        self.logger = get_logger(__name__)
        self.logger.info("Initializing MiniGUI")
        self.setup_ui()

    def setup_ui(self):
        """Creates the UI elements for the MiniGUI application."""
        dpg.create_context()

        # Load and apply font
        with dpg.font_registry():
            self.logger.info(f"Loading font from {self.font_path}")
            self.load_font(self.font_path)
            self.logger.info(f"Font loaded: {self.current_font}")

        with dpg.window(label="Main Menu", width=400, height=80):
            with dpg.tab_bar():
                with dpg.tab(label="Editor"):
                    WindowManager.toggle_window("editor")
                with dpg.tab(label="Tokenizer"):
                    WindowManager.toggle_window("tokenizer")
                with dpg.tab(label="Trainer"):
                    WindowManager.force_toggle("trainer", "generator")
                with dpg.tab(label="Inference"):
                    WindowManager.force_toggle("generator", "trainer")

        with dpg.window(label="Text Editor", width=800, height=600):
            self.create_main_menu()
            self.create_menu_bar()
            self.create_text_area()
            self.create_status_bar()
            self.create_file_dialogs()

        dpg.create_viewport(title="Mini GUI App", width=800, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def load_font(self, font_path):
        """Loads and applies a font dynamically."""
        if font_path:
            with dpg.font_registry():
                new_font = dpg.add_font(font_path, 16)
                dpg.bind_font(new_font)
                self.current_font = new_font

    def create_main_menu(self):
        """Creates the main menu for navigation."""
        with dpg.menu_bar():
            with dpg.menu(label="Navigation"):
                dpg.add_menu_item(
                    label="Text Editor", callback=self.switch_to_text_editor
                )
                dpg.add_menu_item(label="Tokenizer", callback=None)
                dpg.add_menu_item(label="Trainer", callback=None)
                dpg.add_menu_item(label="Inference", callback=None)

    def switch_to_text_editor(self):
        dpg.show_item("text_editor_window")
        dpg.hide_item("tokenizer_window")
        dpg.hide_item("trainer_window")
        dpg.hide_item("inference_window")

    def create_menu_bar(self):
        """Creates the menu bar with File and Tokenizer options."""
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="New", callback=self.new_file)
                dpg.add_menu_item(
                    label="Open", callback=lambda: dpg.show_item("open_file_dialog")
                )
                dpg.add_menu_item(label="Save", callback=self.save_file)
                dpg.add_menu_item(
                    label="Save As", callback=lambda: dpg.show_item("save_file_dialog")
                )
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())

            with dpg.menu(label="Tokenizer"):
                dpg.add_menu_item(
                    label="Load Model",
                    callback=lambda: dpg.show_item("open_model_dialog"),
                )
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

    def create_file_dialogs(self):
        """Creates file dialogs for opening and saving files."""

        # Open File Dialog
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.open_file_callback,
            tag="open_file_dialog",
            width=480,
            height=320,
        ):
            dpg.add_file_extension(".*")

        # Save File Dialog
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.save_as_file_callback,
            tag="save_file_dialog",
            width=480,
            height=320,
        ):
            dpg.add_file_extension(".*")

        # Open Model Dialog (for SentencePiece model)
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.select_model_callback,
            tag="open_model_dialog",
            width=480,
            height=320,
        ):
            dpg.add_file_extension(".model")

    def set_status(self, message):
        """Updates the status bar with a message."""
        dpg.set_value("status_bar", f"Status: {message}")

    def new_file(self):
        """Clears the text area."""
        dpg.set_value("text_area", "")
        self.file_path = None
        self.set_status("New file created.")

    def open_file_callback(self, sender, app_data):
        """Callback when a file is selected in the open dialog."""
        if app_data["selections"]:
            file_path = app_data["selections"][next(iter(app_data["selections"]))]
            if file_path:
                with open(file_path, "r") as file:
                    dpg.set_value("text_area", file.read())
                self.file_path = file_path
                self.set_status(f"Opened file: {self.file_path}")

    def save_file(self):
        """Saves the current file."""
        if self.file_path:
            with open(self.file_path, "w") as file:
                file.write(dpg.get_value("text_area"))
            self.set_status(f"Saved: {self.file_path}")
        else:
            dpg.show_item("save_file_dialog")

    def save_as_file_callback(self, sender, app_data):
        """Callback when a file path is selected in the save dialog."""
        if app_data["selections"]:
            file_path = app_data["selections"][next(iter(app_data["selections"]))]
            if file_path:
                with open(file_path, "w") as file:
                    file.write(dpg.get_value("text_area"))
                self.file_path = file_path
                self.set_status(f"Saved as: {self.file_path}")

    def select_model_callback(self, sender, app_data):
        """Callback when a SentencePiece model is selected."""
        file_path = app_data["file_path_name"]
        if file_path:
            self.processor_path = file_path
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
