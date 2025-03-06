"""
Copyright © 2023 Austin Berrio
Script: mini.cli.loader
Description: This script provides utilities for inspecting plaintext data loading.
"""

import os

from sentencepiece import SentencePieceProcessor

from mini.args.loader import TextLoaderArgs
from mini.data.loader import TextDatasetLoader


def main():
    args = TextLoaderArgs("Mini Loader CLI").parse_args()

    assert os.path.exists(args.model), "Model file does not exist!"
    assert os.path.isfile(args.dataset), "Dataset file must be provided!"

    processor = SentencePieceProcessor(model_file=args.model)
    dataset = TextDatasetLoader(
        args.dataset,
        processor,
        args.max_seq_len,
        args.batch_size,
        args.add_bos,
        args.add_eos,
        args.supervise,
        args.verbose,
    )
    dataset.load_data()

    for i, (x, y) in enumerate(dataset):
        print(f"\033[35;1;1mbatch\033[0m={i + 1}", end=", ")
        print(f"\033[32;1;1minput_shape\033[0m={x.shape}", end=", ")
        print(f"\033[36;1;1mtarget_shape\033[0m={y.shape}")
        if args.verbose:
            print(f"\033[32;1;1minput\033[0m={x}")
            print(f"\033[36;1;1mtarget\033[0m={y}")
            print()
        assert x.shape == y.shape, "Input shape failed to match Target shape"


if __name__ == "__main__":
    main()
