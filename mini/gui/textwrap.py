import textwrap

import dearpygui.dearpygui as dpg


class TextWrapperApp:
    def __init__(self):
        self.last_width = None
        self.last_content = ""
        self.font_size = 16  # Expose font size for adjustments

        self.setup_ui()

    def setup_ui(self):
        with dpg.window(
            label="Text Wrapper", tag="text_wrapper", width=480, height=320
        ):
            dpg.add_input_text(
                tag="wrapped_text",
                multiline=True,
                width=-1,
                height=-1,
                default_value="The quick brown fox jumped over the lazy dog.",
                callback=self.update_text_wrap,
            )

    def estimate_char_width(self, current_width):
        """Estimate how many characters fit in the window width."""
        estimated_char_width = max(6, int(self.font_size * 0.6))
        max_chars_per_line = max(10, current_width // estimated_char_width)
        return max_chars_per_line

    def update_text(self, sender, app_data):
        """Update text wrapping dynamically when input is modified."""
        if app_data and isinstance(app_data, str):
            current_width = dpg.get_item_width("text_wrapper")
            max_chars_per_line = self.estimate_char_width(current_width)

            wrapped_text = textwrap.fill(app_data, width=max_chars_per_line)
            dpg.set_value(sender, wrapped_text)

            print(f"Updated Text → Width: {current_width}, Chars: {max_chars_per_line}")
            print(wrapped_text)
            print("---")

    def update_text_wrap(self, sender=None, app_data=None):
        """Rewrap text when window resizes or text changes."""
        sender = sender if sender else "wrapped_text"
        app_data = app_data if app_data else dpg.get_value(sender)
        current_width = dpg.get_item_width("text_wrapper")
        if app_data != self.last_content or current_width != self.last_width:
            self.last_width = current_width
            self.last_content = app_data
            self.update_text(sender, app_data)


if __name__ == "__main__":
    dpg.create_context()
    dpg.create_viewport(title="Dynamic Text Wrapping", width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    app = TextWrapperApp()

    while dpg.is_dearpygui_running():
        app.update_text_wrap()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
