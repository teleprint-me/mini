"""
Copyright © 2023 Austin Berrio
Module: mini.gui.__main__
Description: Main entry point for the Mini GUI application.
"""

import dearpygui.dearpygui as dpg

from mini.gui.fonts.fuzzer import FontFuzzer
from mini.gui.main_menu import MainMenu
from mini.gui.settings_window import SettingsWindow  # Import Settings


class MiniGUI:
    def __init__(self):
        self.last_width = None
        self.registered_windows = {}
        self.render_callbacks = []
        self.current_font = FontFuzzer.locate_font("Noto Sans Mono")  # Default Font
        self.setup_ui()

    def setup_ui(self):
        """Sets up the GUI layout and initializes windows."""
        dpg.create_context()
        dpg.create_viewport(title="Mini GUI App", width=640, height=480)

        # Initialize Components
        self.main_menu = MainMenu(self)
        self.settings_window = SettingsWindow(self)

        # Register Windows
        for tag in [
            "editor",
            "tokenizer",
            "trainer",
            "generator",
            "graph",
            "evaluator",
            "model",
        ]:
            self.register_window(tag)

        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Manual Render Loop
        while dpg.is_dearpygui_running():
            for callback in self.render_callbacks:
                callback()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def set_font(self, font_path):
        """Applies a new font dynamically."""
        with dpg.font_registry():
            new_font = dpg.add_font(font_path, 16)
            dpg.bind_font(new_font)
            self.current_font = font_path  # Update state

    def register_window(self, tag, existing_window=None):
        """Creates and registers a new window, unless it already exists."""
        if tag in self.registered_windows:
            print(f"Warning: Window '{tag}' is already registered. Skipping duplicate.")
            return

        if existing_window:
            self.registered_windows[tag] = existing_window  # Use the provided window
            return

        with dpg.window(label=tag.capitalize(), tag=tag, show=False, pos=(260, 10)):
            dpg.add_text(f"{tag.capitalize()} UI")
        self.registered_windows[tag] = False  # Store visibility state

    def toggle_window(self, tag):
        """Toggles visibility of a window."""
        if tag in self.registered_windows:
            is_visible = dpg.get_item_configuration(tag)["show"]
            dpg.configure_item(tag, show=not is_visible)

    def add_render_callback(self, callback):
        """Adds a function to be executed in the render loop."""
        self.render_callbacks.append(callback)


if __name__ == "__main__":
    MiniGUI()
