"""
Copyright © 2023 Austin Berrio
Module: mini.gui.__main__
Description: Main entry point for the Mini GUI application.
"""

from pathlib import Path

import dearpygui.dearpygui as dpg

from mini.gui.fonts.fuzzer import FontFuzzer
from mini.gui.main_menu import MainMenu
from mini.gui.settings_window import SettingsWindow  # Import Settings


class MiniGUI:
    def __init__(self):
        self.last_width = None
        self.registered_windows = {}
        self.render_callbacks = []
        self.font_path = FontFuzzer.locate_font("Noto Sans Mono")
        self.font_size = 16
        self.setup_ui()

    def setup_ui(self):
        """Sets up the GUI layout and initializes windows."""
        dpg.create_context()
        dpg.create_viewport(title="Mini GUI App", width=720, height=480)

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

    def set_font(self, font_path=None, font_size=None):
        """Applies a new font dynamically and prevents blurriness."""
        # Default to current values if not provided
        font_path = font_path or self.font_path
        font_size = font_size or self.font_size

        # Avoid redundant updates
        if font_path == self.font_path and font_size == self.font_size:
            return  # No need to reapply the same font

        # Store updated values
        self.font_path = font_path
        self.font_size = font_size

        # 🔥 Fix: Preserve DearPyGui's default scaling method
        scaled_size = font_size * 2  # Double the font size for sharper rendering
        scale_factor = 0.5  # Scale down to avoid blurriness

        # Remove existing fonts
        if dpg.does_item_exist("font_registry"):
            dpg.delete_item("font_registry")

        # Apply new font
        with dpg.font_registry(tag="font_registry"):
            dpg.add_font(self.font_path, scaled_size, tag="active_font")
            dpg.bind_font("active_font")
            dpg.set_global_font_scale(scale_factor)  # Avoid scaling artifacts

        # Log the change
        print(f"Font changed to: {Path(self.font_path).stem}, Size: {self.font_size}")

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
