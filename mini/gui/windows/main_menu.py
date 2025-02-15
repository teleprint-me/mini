"""
Copyright © 2023 Austin Berrio
Module: mini.gui.windows.main_menu
Description: Contains the main menu creation logic for the application.
"""

import dearpygui.dearpygui as dpg

from mini.gui.window_manager import WindowManager


def create_main_menu():
    """Creates the main navigation menu."""
    with dpg.menu_bar():
        with dpg.menu(label="Windows"):
            dpg.add_menu_item(
                label="Text Editor",
                callback=lambda: WindowManager.toggle_window("editor"),
            )
            dpg.add_menu_item(
                label="Tokenizer",
                callback=lambda: WindowManager.toggle_window("tokenizer"),
            )
            dpg.add_menu_item(
                label="Trainer",
                callback=lambda: WindowManager.force_toggle("trainer", "generator"),
            )
            dpg.add_menu_item(
                label="Generator",
                callback=lambda: WindowManager.force_toggle("generator", "trainer"),
            )
            dpg.add_menu_item(
                label="Inference",
                callback=lambda: WindowManager.toggle_window("inference"),
            )
            dpg.add_menu_item(
                label="Visualizer",
                callback=lambda: WindowManager.toggle_window("visualizer"),
            )
