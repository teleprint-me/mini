"""
Module: mini.gui.tokenizer
Description: Mini GUI window for training, encoding, and decoding text using sentencepiece tokenizer.
"""

import logging
import os
from pathlib import Path
from textwrap import TextWrapper

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
        self.sp = spm.SentencePieceProcessor(model_file=self.config.model_path)
        self.train = spm.SentencePieceTrainer.Train  # reference static method
        self.textwrapper = TextWrapper(width=80)

        # Training log
        self.training_log = []

        # Setup UI components
        self.setup_ui()
        self.gui.register_window("tokenizer", existing_window=True)
        self.logger = get_logger(self.__class__.__name__, level=logging.DEBUG)

    def set_input(self, sender, app_data):
        """Handles input file path selection."""
        self.config.input = app_data
        dpg.set_value(sender, self.config.input)
        self.logger.info(f"Input path set to {self.config.input}")

    def set_model_prefix(self, sender, app_data):
        """Handles output file path selection."""
        self.config.model_prefix = app_data
        dpg.set_value(sender, self.config.model_prefix)
        self.logger.info(f"Output path set to {self.config.model_prefix}")

    def set_model_type(self, sender, app_data):
        """Handles model type selection."""
        self.config.model_type = app_data
        dpg.set_value(sender, self.config.model_type)
        self.logger.info(f"Model type set to {self.config.model_type}")

    def set_vocab_size(self, sender, app_data):
        """Handles vocabulary size input."""
        self.vocab_size = max(256, min(app_data, 65536))
        dpg.set_value(sender, self.vocab_size)
        self.logger.info(f"Vocabulary size set to {self.vocab_size}")

    def set_character_coverage(self, sender, app_data):
        """Handles character coverage input."""
        self.config.character_coverage = max(0.1, min(app_data, 1.0))
        dpg.set_value(sender, self.config.character_coverage)
        self.logger.info(f"Character coverage set to {self.config.character_coverage}")

    def set_max_sentence_length(self, sender, app_data):
        """Handles maximum sequence length input."""
        self.config.max_sentence_length = max(1, min(app_data, 1024))
        dpg.set_value(sender, self.config.max_sentence_length)
        self.logger.info(
            f"Maximum sequence length set to {self.config.max_sentence_length}"
        )

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
            self.train(**self.config.as_dict())
            self.log_training_output(
                "INFO", f"Tokenizer model saved to {self.config.model_path}"
            )
        except Exception as e:
            self.log_training_output("ERROR", f"Error during tokenizer training: {e}")
            return
        self.log_training_output("INFO", "Tokenizer training completed successfully.")

    def set_model_path(self, sender, app_data):
        """Handles model file path selection."""
        if isinstance(app_data, str) and not os.path.isfile(app_data):
            return  # Do not update partial paths
        # Assume a string
        self.config.model_path = app_data
        # Override if not a string
        if isinstance(app_data, dict) and "selections" in app_data:
            self.config.model_path = list(app_data["selections"].values())[0]
        dpg.set_value("tokenizer_model_path", self.config.model_path)  # Update the path
        self.sp.load(self.config.model_path)
        self.logger.info(f"Model path set to {self.config.model_path}")

    def encode_text(self, sender, app_data):
        """Encodes input text using the trained tokenizer."""
        text = dpg.get_value("tokenizer_text_area")
        if not text:
            return

        if not Path(self.config.model_path).exists():
            dpg.set_value("tokenizer_text_piece", "[ERROR] Model not found.")
            return

        width = dpg.get_item_width("tokenizer")
        print("width:", width)
        self.sp.load(self.config.model_path)
        ids = self.sp.encode(text, out_type=int)
        pieces = self.sp.encode(text, out_type=str)
        meta = "\u2581"
        pieced = []
        for p in pieces:
            if meta in p:
                p = p.replace(meta, " ")  # fix mojibake
            pieced.append(f"'{p}'")  # wrap piece in quotes
        encoded_text = ", ".join(map(str, ids))
        pieced_text = ", ".join(map(str, pieced))
        dpg.set_value("tokenizer_text_piece", pieced_text)
        dpg.set_value("tokenizer_text_encoding", encoded_text)

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
                with dpg.tab(label="Trainer"):
                    dpg.add_text("Dataset Input Path:")
                    dpg.add_input_text(
                        default_value=self.config.input,
                        callback=self.set_input,
                        tag="tokenizer_input",
                    )
                    dpg.add_text("Model Prefix:")
                    dpg.add_input_text(
                        default_value=self.config.model_prefix,
                        callback=self.set_model_prefix,
                        tag="tokenizer_model_prefix",
                    )
                    dpg.add_text("Model Type:")
                    dpg.add_combo(
                        items=self.config.list_model_types(),
                        default_value=self.config.model_type,
                        callback=self.set_model_type,
                        tag="tokenizer_model_type",
                    )
                    dpg.add_text("Vocab Size:")
                    dpg.add_input_int(
                        default_value=self.config.vocab_size,
                        callback=self.set_vocab_size,
                        tag="tokenizer_vocab_size",
                    )
                    dpg.add_text("Character Coverage:")
                    dpg.add_input_float(
                        default_value=self.config.character_coverage,
                        callback=self.set_character_coverage,
                        tag="tokenizer_character_coverage",
                    )
                    dpg.add_button(label="Train", callback=self.train_tokenizer)

                with dpg.tab(label="Model"):
                    with dpg.group(horizontal=True):
                        dpg.add_text("Model Path:")
                        with dpg.file_dialog(
                            show=False,
                            width=480,
                            height=320,
                            default_path=Path(self.config.model_path).parent.name,
                            default_filename=Path(self.config.model_path).name,
                            callback=self.set_model_path,
                            tag="tokenizer_model_path_dialog",
                        ):
                            dpg.add_file_extension(".model", color=(0, 255, 0))
                            dpg.add_file_extension(".vocab", color=(0, 255, 0))
                        dpg.add_button(
                            label="Load",
                            callback=lambda: dpg.show_item(
                                "tokenizer_model_path_dialog"
                            ),
                        )
                        dpg.add_input_text(
                            default_value=self.config.model_path,
                            callback=self.set_model_path,
                            width=-1,
                            hint="models/tokenizer.model",
                            tag="tokenizer_model_path",
                        )

                    dpg.add_input_text(
                        multiline=True,
                        height=80,
                        width=-1,
                        tracked=True,
                        tab_input=True,
                        callback=self.encode_text,
                        tag="tokenizer_text_area",
                    )
                    dpg.add_input_text(
                        readonly=True,
                        multiline=True,
                        height=80,
                        width=-1,
                        tracked=True,
                        tag="tokenizer_text_piece",
                    )
                    dpg.add_input_text(
                        readonly=True,
                        multiline=True,
                        height=80,
                        width=-1,
                        tracked=True,
                        tag="tokenizer_text_encoding",
                    )
