# Mini

Mini is a toolkit for pre-training, fine-tuning, and evaluating transformer based language models using PyTorch.

## About

Mini is a Decoder-only Language Model (LM) designed for instruction-based tasks. It is trained on a synthetic corpus of instruction-based data created by a Teacher. Mini is trained to predict the next word in a sequence given an instruction and can be fine-tuned for specific tasks.

## Features

- [x] **Pre-training**: Mini can be pre-trained using plaintext or structured data.
- [ ] **Inference**: Mini can be used for inference to generate text given a prompt or instruction.
- [ ] **Fine-tuning**: Mini can be fine-tuned for specific tasks using a small amount of data.
- [ ] **Evaluation**: Mini can be evaluated on various metrics such as perplexity and BLEU score.

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
wget https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt -O data/tinyshakespeare.txt
```

3. Train the model:

```sh
python -m mini.transformer.train --processor models/tokenizer.model \
    --dataset data/tinyshakespeare.txt \
    --model models/mini_transformer.pt \
    --embed-dim 256 \
    --num-heads 4 \
    --head-dim 64 \
    --num-layers 6 \
    --ff-dim 512 \
    --max-seq-len 128 \
    --batch-size 4 \
    --batch-stride 16 \
    --num-epochs 3 \
    --save-every 1 \
    --lr 5e-4 \
    --eps 1e-8 \
    --weight-decay 0.01 \
    --step-size 1 \
    --gamma 0.9 \
    --grad-accum-steps 2 \
    -v
```

4. Inference the model:

```sh
python -m mini.transformer.infer --processor models/tokenizer.model \
    --model models/mini_transformer.pt \
    --embed-dim 256 \
    --num-heads 4 \
    --head-dim 64 \
    --num-layers 6 \
    --ff-dim 512 \
    --max-seq-len 128 \
    --max-tokens 64 \
    --prompt 'You common cry of curs! whose'
```

5. Evaluate the model:

```sh
# TODO: Add instructions for evaluating the model
```

## License

This project is licensed under the AGPL License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes. Make sure to follow the [Code of Conduct](CODE_OF_CONDUCT.md). 
