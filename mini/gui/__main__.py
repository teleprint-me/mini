"""
Module: mini.gui.__main__
Description: Main entry point for the Mini GUI application.
"""

import dearpygui.dearpygui as dpg

from mini.gui.main_menu import MainMenu


class MiniGUI:
    def __init__(self):
        self.last_width = None
        self.registered_windows = {}  # Store registered windows
        self.render_callbacks = []  # Store functions to call every frame

        self.setup_ui()

    def setup_ui(self):
        """Sets up the GUI layout and initializes windows."""
        dpg.create_context()
        dpg.create_viewport(title="Mini GUI App", width=640, height=480)

        # Initialize Main Menu
        self.main_menu = MainMenu(self)

        # Create all windows dynamically
        for tag in [
            "editor",
            "tokenizer",
            "trainer",
            "generator",
            "graph",
            "evaluator",
            "model",
            "settings",
        ]:
            self.register_window(tag)

        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Manual render loop (polling for updates)
        while dpg.is_dearpygui_running():
            for callback in self.render_callbacks:
                callback()  # Run registered UI updates
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def register_window(self, tag):
        """Creates and registers a window with a given tag."""
        with dpg.window(label=tag.capitalize(), tag=tag, show=False, pos=(260, 10)):
            dpg.add_text(f"{tag.capitalize()} UI")
        self.registered_windows[tag] = False  # Store visibility state

    def toggle_window(self, tag):
        """Toggles visibility of a window."""
        is_visible = dpg.get_item_configuration(tag)["show"]
        dpg.configure_item(tag, show=not is_visible)

    def add_render_callback(self, callback):
        """Adds a function to be executed in the render loop."""
        self.render_callbacks.append(callback)


if __name__ == "__main__":
    MiniGUI()
