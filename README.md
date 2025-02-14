# **Mini**

_A lightweight modular framework for language modeling in PyTorch._

Mini is a lightweight modular framework for pre-training, fine-tuning, and
evaluating transformer-based language models. It is designed to support
sequence-to-sequence learning tasks such as **next-word prediction**,
**instruction tuning**, and **text generation**.

## **Key Features**

- **Pre-training on structured & unstructured text**
- **Fine-tuning for instruction-based tasks** _(coming soon)_
- **Evaluation with perplexity and future metric support** _(BLEU, ROUGE, etc.)_
- **Easy CLI interface for training, inference, and evaluation**
- **Support for model checkpointing & resuming**
- **Optimized training with RMSNorm, SiLU activation, and RoPE Attention**

## **Architecture**

Mini follows both classical and state-of-the-art transformer-based architectures
with a simplified and efficient design:

### **Encoding**

- **PositionalEncoding** - Adds positional information to input tokens.
- **BertEncoding** - Uses residual learning to improve positional encoding.
- **LinearEncoding** - Uses linear transformations for positional encoding.
- **RotaryEncoding** - Uses rotary positional encoding for efficient
  computation.

### **Embedding**

- **PositionalEmbedding** - Embeds input tokens with positional information.
- **BertEmbedding** - Uses residual learning to improve embedding.
- **LinearEmbedding** - Uses linear transformations for embedding.

### **Attention**

- **SelfAttention** - Computes self-attention between tokens.
- **RotaryAttention** - Uses rotary positional encoding for efficient
  computation.

### **Normalization**

- **LayerNorm** - Normalizes across the last dimension.
- **RMSNorm** - Normalizes activations using the root mean square.

### **Feed-Forward**

- **PositionWiseFeedForward** - Applies feed-forward transformations to each
  token.
- **GatedFeedForward** - Applies feed-forward transformations with gating.

### **Transformer**

- **PositionWiseBlock** - Combines self-attention and feed-forward layers.
- **GatedBlock** - Combines rotary self-attention and feed-forward layers with gating.

Current implementations focus on **position-wise** and **gated architectures**.
The goal is to provide a flexible and efficient framework for building
transformer-based models. Mini includes a variety of components and modules that
allow for easy experimentation and customization.

## **Installation & Setup**

### **1. Clone the repository**

```sh
git clone https://github.com/teleprint-me/mini.git
cd mini
```

### **2. Setup a virtual environment**

```sh
python3.12 -m venv .venv
source .venv/bin/activate
```

### **3. Install dependencies**

#### **Install PyTorch**

- **CPU**

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- **CUDA**

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

- **ROCm**

```sh
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2.4
```

#### **Install Requirements**

```sh
pip install -r requirements.txt
```

### **4. Validate tokenizer and dataset**

```sh
sha256sum -c sha256sum.txt
```

Expected output:

```sh
models/tokenizer.model: OK
data/mini-fairy.md: OK
data/mini-owl.md: OK
data/mini-owl-fairy.md: OK
```

Check the dataset character count.

```sh
wc -c data/mini-owl.md
```

Expected output:

```sh
1078 data/mini-owl.md
```

## **Usage**

### **Pre-training**

Train a model from scratch on a dataset:

```sh
python -m mini.cli.train \
    --processor models/tokenizer.model \
    --model models/misty-owl.pth \
    --dataset data/mini-owl.md \
    --architecture misty \
    --num-epochs 10 \
    --batch-size 2 \
    --batch-stride 8 \
    --lr 1e-4 \
    --optimizer adamw \
    --scheduler none \
    --criterion cross_entropy \
    --verbose
```

- **Parameters:**
  - `--processor models/tokenizer.model`: Path to the tokenizer model.
  - `--model models/misty-owl.pth`: Path to the model state.
  - `--dataset data/mini-owl.md`: Path to the dataset.
  - `--architecture misty`: Model architecture.
  - `--num-epochs 10`: Number of training epochs.
  - `--batch-size 2`: Batch size.
  - `--batch-stride 8`: Sequence length per sub-batch.
  - `--lr 1e-4`: Learning rate.
  - `--optimizer adamw`: Optimizer algorithm.
  - `--scheduler none`: Learning rate scheduler.
  - `--criterion cross_entropy`: Loss function.
  - `--verbose`: Enable verbosity for debugging.

**NOTE:** Any plaintext file will work. `mini-owl.md` is used for isolated and
controlled experimentation. See [training.md](docs/training.md) for more
information.

### **Inference**

Run inference on a trained model:

```sh
python -m mini.cli.infer \
    --processor models/tokenizer.model \
    --model models/misty-owl.pth \
    --temperature 0.5 \
    --prompt "The young bird listened"
```

- **Parameters:**
  - `--processor models/tokenizer.model`: Path to the tokenizer model.
  - `--model models/misty-owl.pth`: Path to the model state.
  - `--temperature 0.5`: Temperature for sampling. Lower values make the output
    more deterministic, and higher values make it more diverse.
  - `--prompt "The young bird listened"`: Input sequence for inference.

### **Fine-tuning** _(coming soon)_

Fine-tune on an instruction-based dataset.

```sh
# Placeholder for fine-tuning command
```

### **Evaluation** _(coming soon)_

Evaluate model performance with perplexity, BLEU, and other metrics.

```sh
# Placeholder for evaluation script
```

## **Development Roadmap**

- [x] **Pre-training on custom datasets**
- [x] **Inference support for text generation**
- [ ] **Fine-tuning for instruction-based tasks** _(up next! 🚀)_
- [ ] **Evaluation with additional NLP metrics**
- [ ] **Distributed training & performance optimizations**

## **License**

This project is licensed under **AGPL-3.0**. See the [LICENSE](LICENSE) file for
details.

## **Contributing**

Contributions are welcome! If you have ideas or improvements, feel free to open
an issue or submit a pull request. Be sure to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
