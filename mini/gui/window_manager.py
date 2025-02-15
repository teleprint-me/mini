"""
Copyright © 2023 Austin Berrio
Module: mini.gui.window_manager
Description: Manages the visibility of different windows in the GUI.
"""

import dearpygui.dearpygui as dpg


class WindowManager:
    """Handles window visibility and toggling between sections."""

    windows = {
        "editor": True,  # Always open
        "tokenizer": True,  # Optional toggling
        "evaluator": False,  # Toggle with tokenizer
        "trainer": False,  # Toggle with generator
        "generator": False,  # Toggle with trainer
        "inference": False,  # Generate predictions
        "visualizer": False,  # Visualize predictions
    }

    @classmethod
    def toggle_window(cls, window_name):
        """Toggle a window's visibility."""
        if window_name in cls.windows:
            cls.windows[window_name] = not cls.windows[window_name]
            dpg.configure_item(window_name, show=cls.windows[window_name])

    @classmethod
    def force_toggle(cls, enable_window, disable_window):
        """Ensures only one of two windows is active at a time."""
        cls.windows[enable_window] = True
        cls.windows[disable_window] = False
        dpg.configure_item(enable_window, show=True)
        dpg.configure_item(disable_window, show=False)
