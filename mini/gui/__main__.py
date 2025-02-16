"""
Module: mini.gui.__main__
Description: Main entry point for the Mini GUI application.
"""

import dearpygui.dearpygui as dpg


class MiniGUI:
    def __init__(self):
        self.last_width = None  # Track last window width
        self.menu_buttons_group = None  # Store button group ID
        self.setup_ui()

    def setup_ui(self):
        """Sets up the GUI layout and initializes windows."""
        dpg.create_context()
        dpg.create_viewport(title="Mini GUI App", width=640, height=480)

        # Main Menu Window
        with dpg.window(
            label="Main Menu",
            tag="main_menu",
            width=240,
            height=380,
            pos=(10, 10),
        ):
            dpg.add_text("Navigation:")
            dpg.add_separator()
            self.menu_buttons_group = dpg.add_group()  # Store button group ID
            self.rebuild_buttons()  # Initialize buttons

        # Example UI Windows
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
            with dpg.window(label=tag.capitalize(), tag=tag, show=False, pos=(260, 10)):
                dpg.add_text(f"{tag.capitalize()} UI")

        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Start Manual Render Loop
        while dpg.is_dearpygui_running():
            self.update_button_width()  # Check for window width changes
            dpg.render_dearpygui_frame()  # Render new frame

        dpg.destroy_context()

    def toggle_window(self, sender, app_data, user_data):
        """Toggles the visibility of a window."""
        is_visible = dpg.get_item_configuration(user_data)["show"]
        dpg.configure_item(user_data, show=not is_visible)

    def update_button_width(self):
        """Detects window width changes and rebuilds buttons dynamically."""
        window_width = dpg.get_item_width("main_menu")
        if window_width != self.last_width:  # Only update if width actually changes
            self.last_width = window_width  # Store new width
            self.rebuild_buttons()  # Rebuild buttons with new width

    def rebuild_buttons(self):
        """Rebuilds the menu buttons dynamically when resizing."""
        dpg.delete_item(
            self.menu_buttons_group, children_only=True
        )  # Clear old buttons
        button_width = max(80, dpg.get_item_width("main_menu") - 20)

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
            dpg.add_button(
                label=tag.capitalize(),
                width=button_width,
                height=30,
                callback=self.toggle_window,
                user_data=tag,
                parent=self.menu_buttons_group,  # Attach to button group
            )


if __name__ == "__main__":
    MiniGUI()
