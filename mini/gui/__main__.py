"""
Module: mini.gui.__main__
Description: Main entry point for the Mini GUI application.
"""

import dearpygui.dearpygui as dpg


class MiniGUI:
    def __init__(self):
        self.setup_ui()

    def setup_ui(self):
        """Sets up the GUI layout and initializes windows."""

        dpg.create_context()
        dpg.create_viewport(title="Custom Title", width=600, height=300)

        # State with a simple main menu
        with dpg.window(label="Main Menu", width=200, height=500, pos=(10, 10)):
            dpg.add_text("Navigation:")
            dpg.add_separator()
            dpg.add_button(
                label="Editor", callback=lambda: self.toggle_window("editor")
            )
            dpg.add_button(
                label="Tokenizer", callback=lambda: self.toggle_window("tokenizer")
            )
            dpg.add_button(
                label="Trainer", callback=lambda: self.toggle_window("trainer")
            )
            dpg.add_button(
                label="Generator", callback=lambda: self.toggle_window("generator")
            )
            dpg.add_button(label="Graph", callback=lambda: self.toggle_window("graph"))
            dpg.add_button(
                label="Evaluator", callback=lambda: self.toggle_window("evaluator")
            )
            dpg.add_button(label="Model", callback=lambda: self.toggle_window("model"))
            dpg.add_button(
                label="Settings", callback=lambda: self.toggle_window("settings")
            )

        # Example UI Windows
        with dpg.window(label="Editor", tag="editor", show=False, pos=(220, 10)):
            dpg.add_text("Text Editor UI")
        with dpg.window(label="Tokenizer", tag="tokenizer", show=False, pos=(220, 10)):
            dpg.add_text("Tokenizer UI")
        with dpg.window(label="Trainer", tag="trainer", show=False, pos=(220, 10)):
            dpg.add_text("Trainer UI")
        with dpg.window(label="Generator", tag="generator", show=False, pos=(220, 10)):
            dpg.add_text("Generator UI")
        with dpg.window(label="Graph", tag="graph", show=False, pos=(220, 10)):
            dpg.add_text("Graph UI")
        with dpg.window(label="Evaluator", tag="evaluator", show=False, pos=(220, 10)):
            dpg.add_text("Evaluator UI")
        with dpg.window(label="Model", tag="model", show=False, pos=(220, 10)):
            dpg.add_text("Model UI")
        with dpg.window(label="Settings", tag="settings", show=False, pos=(220, 10)):
            dpg.add_text("Settings UI")

        # Main loop
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def toggle_window(self, tag):
        """Toggles the visibility of a window."""
        is_visible = dpg.get_item_configuration(tag)["show"]
        dpg.configure_item(tag, show=not is_visible)


if __name__ == "__main__":
    MiniGUI()
