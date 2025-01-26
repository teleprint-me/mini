# Mini

Mini is a Language Model for Natural Language Processing instruction based tasks implemented in PyTorch.

## About

Mini is a Decoder-only Language Model (LM) designed for instruction-based tasks. It is trained on a synthetic corpus of instruction-based data created by a Teacher. Mini is trained to predict the next word in a sequence given an instruction and can be fine-tuned for specific tasks.

## Features

- **Instruction-based tasks**: Mini is designed to handle instruction-based tasks, such as summarization, translation, and question answering.
- **Fine-tuning**: Mini can be fine-tuned for specific tasks using a small amount of data.
- **Efficiency**: Mini is optimized for efficiency, with a small number of parameters and fast training times.
- **Scalability**: Mini can be scaled to handle large amounts of data and can be used in distributed settings.

## Setup

1. Clone the repository:

```sh
git clone https://github.com/teleprint-me/mini.git
cd mini
```

2. Setup PyTorch:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

3. Install requirements:

```sh
pip install -r requirements.txt
```

## Usage

1. Create a directory for the dataset:

```sh
mkdir data
```

2. Download the dataset:

```sh
wget https://raw.githubusercontent.com/teknium1/GPTeacher/refs/heads/main/Instruct/gpt4-instruct-dedupe-only-dataset.json -O data/gpt4-instruct-dedupe-only-dataset.json
```

3. Pre-process the dataset:

```sh
python -m mini.data.filter -i data/gpt4-instruct-dedupe-only-dataset.json \
    -o data/gpt4-instruct-mini-dataset.json \
    -s data/gpt4-instruct-mini-schema.json \
    --seed 42 --shuffle --pairs 10
```

4. Train the model:

```sh
# TODO: Add instructions for running the model
```

5. Evaluate the model:

```sh
# TODO: Add instructions for evaluating the model
```

6. Inference the model:

```sh
# TODO: Add instructions for running inference
```

## License

This project is licensed under the AGPL License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes. Make sure to follow the [Code of Conduct](CODE_OF_CONDUCT.md). 
