"""
Module: mini.gui.__main__
Description: Main entry point for the Mini GUI application.
"""

import dearpygui.dearpygui as dpg


class MiniGUI:
    def __init__(self):
        self.last_width = None  # Track last window width
        self.setup_ui()

    def setup_ui(self):
        """Sets up the GUI layout and initializes windows."""
        dpg.create_context()
        dpg.create_viewport(title="Mini GUI App", width=640, height=480)

        with dpg.window(
            label="Main Menu",
            tag="main_menu",  # Assigning a tag for easy reference
            width=240,
            height=380,
            pos=(10, 10),
        ):
            dpg.add_text("Navigation:")
            dpg.add_separator()

            self.menu_buttons = []  # Store button tags for later updates
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
                button = dpg.add_button(
                    label=tag.capitalize(),
                    width=220,  # Default button width
                    height=30,
                    callback=self.toggle_window,
                    user_data=tag,
                )
                self.menu_buttons.append(button)

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

        # Register a frame callback for dynamic width adjustment
        dpg.set_frame_callback(1, self.update_button_width)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def toggle_window(self, sender, app_data, user_data):
        """Toggles the visibility of a window."""
        is_visible = dpg.get_item_configuration(user_data)["show"]
        dpg.configure_item(user_data, show=not is_visible)

    def update_button_width(self, sender):
        """Polls for window width changes and updates button sizes."""
        window_width = dpg.get_item_width("main_menu")
        if window_width != self.last_width:  # Only update if width actually changes
            self.last_width = window_width  # Store new width
            button_width = max(80, window_width - 20)

            for button in self.menu_buttons:
                dpg.configure_item(button, width=button_width)

        # Keep polling every frame
        dpg.set_frame_callback(1, self.update_button_width)


if __name__ == "__main__":
    MiniGUI()
