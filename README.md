# **Mini**

_A lightweight toolkit for training language models in PyTorch._

Mini is a modular framework for pre-training, fine-tuning, and evaluating
transformer-based language models. It is designed to support
sequence-to-sequence learning tasks such as **next-word prediction**,
**instruction tuning**, and **text generation**.

## **Key Features**

- **Pre-training on structured & unstructured text**
- **Fine-tuning for instruction-based tasks** _(coming soon)_
- **Evaluation with perplexity and future metric support** _(BLEU, ROUGE, etc.)_
- **Easy CLI interface for training, inference, and evaluation**
- **Support for model checkpointing & resuming**
- **Optimized training with RMSNorm, SiLU activation, and learnable embeddings**

## **Architecture**

Mini follows a transformer-based architecture with a simplified and efficient
design:

- **`MiniEmbedding`** – Learns both token and positional embeddings.
- **`MiniAttention`** – Implements scaled dot-product self-attention.
- **`MiniBlock`** – Stacks self-attention and feedforward layers.
- **`MiniTransformer`** – A lightweight sequence-to-sequence model.
- **`RMSNorm`** – Stabilizes training by normalizing activations.
- **`FeedForward`** – Non-linear transformations with SiLU activation.

The current implementation focuses on **pre-training** with a learnable
embedding block. Future iterations may include **RoPE (Rotary Positional
Encoding)** and **adaptive attention mechanisms**.

## **Installation & Setup**

### **1. Clone the repository**

```sh
git clone https://github.com/teleprint-me/mini.git
cd mini
```

### **2. Install dependencies**

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### **3. Prepare a dataset**

```sh
mkdir data
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/tinyshakespeare.txt
```

## **Usage**

### **Pre-training**

Train a model from scratch on a dataset:

```sh
python -m mini.transformer.train --processor models/tokenizer.model \
    --dataset data/tinyshakespeare.txt \
    --model models/mini.pt \
    --embed-dim 256 --num-heads 4 --head-dim 64 \
    --num-layers 6 --ff-dim 512 --max-seq-len 128 \
    --batch-size 4 --batch-stride 16 --num-epochs 3 \
    --save-every 1 --lr 5e-4 --eps 1e-8 --weight-decay 0.01 \
    --step-size 1 --gamma 0.9 --grad-accum-steps 2 -v
```

### **Inference** _(coming soon)_

Run inference on a trained model:

```sh
python -m mini.transformer.infer --processor models/tokenizer.model \
    --model models/mini.pt \
    --embed-dim 256 --num-heads 4 --head-dim 64 \
    --num-layers 6 --ff-dim 512 --max-seq-len 128 \
    --max-tokens 64 --prompt 'You common cry of curs! whose'
```

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
- [ ] **Inference support for text generation** _(up next! 🚀)_
- [ ] **Fine-tuning for instruction-based tasks**
- [ ] **Evaluation with additional NLP metrics**
- [ ] **Distributed training & performance optimizations**

## **License**

This project is licensed under **AGPL-3.0**. See the [LICENSE](LICENSE) file for
details.

## **Contributing**

Contributions are welcome! If you have ideas or improvements, feel free to open
an issue or submit a pull request. Be sure to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
