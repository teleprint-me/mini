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
        dpg.create_viewport(title="Mini GUI App", width=640, height=480)

        # State with a simple main menu
        with dpg.window(
            label="Main Menu",
            width=100,
            height=320,
            pos=(10, 10),
        ) as main_menu:
            button_width = dpg.get_item_width(main_menu) - 20
            button_height = 20
            menu_items = [
                "editor",
                "tokenizer",
                "trainer",
                "generator",
                "graph",
                "evaluator",
                "model",
                "settings",
            ]
            dpg.add_text("Navigation:")
            dpg.add_separator()
            for tag in menu_items:
                dpg.add_button(
                    label=tag.capitalize(),
                    width=button_width,
                    height=button_height,
                    callback=self.toggle_window,
                    user_data=tag,
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

    def toggle_window(self, sender, app_data, user_data):
        """Toggles the visibility of a window."""
        print(f"Toggling window: {user_data}")

        is_visible = dpg.get_item_configuration(user_data)["show"]
        dpg.configure_item(user_data, show=not is_visible)


if __name__ == "__main__":
    MiniGUI()
