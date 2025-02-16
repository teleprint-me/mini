import dearpygui.dearpygui as dpg

from mini.gui.fonts.fuzzer import FontFuzzer


class SettingsWindow:
    def __init__(self, gui):
        """Creates the settings window with multiple tabs."""
        self.gui = gui
        self.font_options = []  # Store available fonts
        self.selected_font = None

        self.setup_ui()

    def setup_ui(self):
        """Creates the settings window layout with tabbed categories."""
        with dpg.window(
            label="Settings",
            tag="settings_window",
            show=False,
            width=400,
            height=300,
            pos=(260, 10),
        ):
            with dpg.tab_bar():
                with dpg.tab(label="Mini Config"):
                    self.create_ui_settings()
                with dpg.tab(label="Editor Config"):
                    dpg.add_text("Editor Settings Placeholder")
                with dpg.tab(label="Tokenizer Config"):
                    dpg.add_text("Tokenizer Settings Placeholder")
                with dpg.tab(label="Trainer Config"):
                    dpg.add_text("Trainer Settings Placeholder")
                with dpg.tab(label="Generator Config"):
                    dpg.add_text("Generator Settings Placeholder")
                with dpg.tab(label="Graph Config"):
                    dpg.add_text("Graph Settings Placeholder")

    def create_ui_settings(self):
        """Creates the UI settings tab, including font selection."""
        dpg.add_text("Select Font:")
        self.font_options = FontFuzzer.list_fonts(filter_common=True)
        self.selected_font = dpg.add_combo(
            self.font_options,
            callback=self.set_font,
            default_value=self.gui.current_font,
        )

    def set_font(self, sender, app_data):
        """Updates the UI font."""
        selected_font = FontFuzzer.locate_font(app_data)
        if selected_font:
            self.gui.set_font(selected_font)
            print(f"Font changed to: {selected_font}")
