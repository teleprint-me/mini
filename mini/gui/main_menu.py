"""
Copyright © 2023 Austin Berrio
Module: mini.gui.windows.main_menu
Description: Contains the main menu creation logic for the application.
"""

import dearpygui.dearpygui as dpg


class MainMenu:
    def __init__(self, gui):
        """Creates the main menu window and registers buttons."""
        self.gui = gui  # Reference to MiniGUI for window toggling
        self.last_width = None
        self.menu_buttons_group = None

        self.setup_ui()

    def setup_ui(self):
        """Creates the main menu UI elements."""
        with dpg.window(
            label="Main Menu",
            tag="main_menu",
            width=240,
            height=380,
            pos=(10, 10),
        ):
            dpg.add_text("Navigation:")
            dpg.add_separator()
            self.menu_buttons_group = dpg.add_group()
            self.rebuild_buttons()

        # Ensure the buttons update dynamically
        self.gui.add_render_callback(self.update_button_width)

    def toggle_window(self, sender, app_data, user_data):
        """Toggles visibility of registered windows in MiniGUI."""
        self.gui.toggle_window(user_data)

    def update_button_width(self):
        """Checks for window width changes and updates buttons dynamically."""
        window_width = dpg.get_item_width("main_menu")
        if window_width != self.last_width:
            self.last_width = window_width
            self.rebuild_buttons()

    def rebuild_buttons(self):
        """Clears and re-creates buttons dynamically."""
        dpg.delete_item(self.menu_buttons_group, children_only=True)
        button_width = max(80, dpg.get_item_width("main_menu") - 20)

        for tag in self.gui.registered_windows:
            dpg.add_button(
                label=tag.capitalize(),
                width=button_width,
                height=30,
                callback=self.toggle_window,
                user_data=tag,
                parent=self.menu_buttons_group,
            )
