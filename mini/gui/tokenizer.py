"""
Module: mini.gui.tokenizer
Description: Mini GUI window for training, encoding, and decoding text using sentencepiece tokenizer.
"""

import io
import logging
from pathlib import Path

import dearpygui.dearpygui as dpg
import sentencepiece as spm

from mini.common.logger import get_logger
from mini.config.tokenizer import ConfigTokenizer


class TokenizerWindow:
    def __init__(self, gui):
        # MiniGUI object reference
        self.gui = gui

        # Load configuration or use defaults
        self.config = ConfigTokenizer()

        # tokenizer model and trainer objects
        self.sp = io.BytesIO()
        self.train = spm.SentencePieceTrainer.train

        # Training log
        self.training_log = []

        # Setup UI components
        self.setup_ui()
        self.gui.register_window("tokenizer", existing_window=True)
        self.logger = get_logger(self.__class__.__name__, level=logging.DEBUG)

    def set_input_path(self, sender, app_data):
        """Handles input file path selection."""
        self.input_path = app_data
        dpg.set_value("tokenizer_input_path", self.input_path)
        self.logger.info(f"Input path set to {self.input_path}")

    def set_output_path(self, sender, app_data):
        """Handles output file path selection."""
        self.output_path = app_data
        dpg.set_value("tokenizer_output_path", self.output_path)
        self.logger.info(f"Output path set to {self.output_path}")

    def set_vocab_size(self, sender, app_data):
        """Handles vocabulary size input."""
        self.vocab_size = max(256, min(app_data, 65536))
        dpg.set_value("tokenizer_vocab_size", self.vocab_size)
        self.logger.info(f"Vocabulary size set to {self.vocab_size}")

    def log_training_output(self, level: str, message: str):
        """Logs training output to the UI."""
        message = f"[{level.upper()}] {message}"
        self.training_log.append(message)
        dpg.set_value("tokenizer_training_log", "\n".join(self.training_log))
        self.logger.info(f"Training log updated: {message}")

    def train_tokenizer(self):
        """Triggers tokenizer training."""
        self.logger.info("Training tokenizer...")
        if not Path(self.input_path).exists():
            self.log_training_output("ERROR", "Input path does not exist.")
            return
        if not self.output_path:
            self.log_training_output("ERROR", "Output path not set.")
            return

        try:
            self.log_training_output("INFO", "Starting tokenizer training...")
            spm.SentencePieceTrainer.train(
                input=self.input_path,
                model_prefix=self.output_path,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
                character_coverage=1.0,
                user_defined_symbols=[],
                max_sentence_length=4096,
                shuffle_input_sentence=True,
                enable_sampling=False,
                num_threads=4,
                write_utf8=True,
            )
        except Exception as e:
            self.log_training_output("ERROR", f"Error during tokenizer training: {e}")
            return
        self.log_training_output("INFO", "Tokenizer training completed successfully.")

    def setup_ui(self):
        with dpg.window(
            label="Tokenizer",
            tag="tokenizer",
            show=False,
            pos=(260, 10),
            width=450,
            height=380,
        ):
            with dpg.tab_bar(label="Tokenizer"):
                with dpg.tab(label="Train"):
                    dpg.add_input_text(
                        label="Set Input Path",
                        default_value="",
                        callback=self.set_input_path,
                    )
                    dpg.add_combo(
                        items=self.list_model_types(),
                        label="tokenizer_model_type",
                        default_value=self.model_type,
                    )
                    dpg.add_input_text(
                        label="Output Path",
                        default_value="",
                        callback=self.set_output_path,
                    )
                    dpg.add_input_int(
                        label="Vocab Size",
                        default_value=8000,
                        callback=self.set_vocab_size,
                    )
                    dpg.add_button(label="Train", callback=self.train_tokenizer)
                # with dpg.tab(label="Encode"):
                #     dpg.add_input_text(
                #         label="Text", default_value="", callback=self.encode_text
                #     )
                #     dpg.add_button(label="Encode", callback=self.encode_text)
                # with dpg.tab(label="Decode"):
                #     dpg.add_input_text(
                #         label="Encoded Text",
                #         default_value="",
                #         callback=self.decode_text,
                #     )
                #     dpg.add_button(label="Decode", callback=self.decode_text)
