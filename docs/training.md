# Training with Mini

Mini is a language modeling framework built with PyTorch, designed to support
modern transformer architectures. This guide walks through the process of
training a model from scratch using a simple plaintext dataset.

## **1. Setting Up**

Before training a model, ensure you have the required dependencies installed:

```sh
pip install -r requirements.txt
```

Mini provides a command-line tool (`train.py`) to manage training runs. The tool
is highly configurable, supporting various optimizers, loss functions, and
scheduling options.

## **2. Choosing a Dataset**

For this tutorial, we use the `data/mini-owl.md` dataset—a **tiny, synthetic
short story** distilled from a larger parent model.

- **Size**: ~1053 characters
- **Purpose**: Demonstrates model convergence on small datasets
- **Impact**: Model parameters must align with dataset size to avoid
  underfitting or overfitting.

A small dataset like this **limits generalization** but allows quick iterations
to observe learning patterns.

## **3. Tokenization Strategy**

Mini uses **SentencePiece** tokenization. The framework includes a pre-trained
`tokenizer.model` from **Mistral**, allowing immediate use without training a
custom tokenizer.

- **Why SentencePiece?**
  - Subword-based for flexibility
  - No need for custom tokenization pretraining
  - Works efficiently with small and large datasets

```sh
# Tokenizing a sample input
python -m mini.data.tokenizer --model-file models/tokenizer.model --input-file data/mini-owl.md
```

This command will generate a tokenized version of the dataset, which is required
for training.

The tokenizer impacts **sequence length, vocabulary size, and overall model
efficiency**, making it a crucial step in training.

## **4. Understanding Batching and Stride**

Mini's training pipeline processes data in batched input-target pairs. Several
key parameters control how text is structured before training:

- **`--max-seq-len`**: Defines the length of each encoded sequence. Shorter
  sequences are padded.
- **`--batch-size`**: Determines the number of sequences grouped into a single
  training batch.
- **`--batch-stride`**: Controls the length of the batch sequence per step. This
  defines how input-target pairs are grouped and processed.

### **Batch Dimensions**

The shape of the input tensors follows `[B, T]`, where:

- `B` is the **batch size** (number of grouped sequences)
- `T` is the **sequence length** (determined by `--batch-stride`)

Example: With `--batch-size 2`, `--batch-stride 8`, and `--max-seq-len 128`, we
may process **3 or 4 batches per epoch** depending on dataset size.

Since batch settings **depend on the dataset**, there are default values, but no
universally "sane" defaults. These must be tuned per dataset and model size.

## **5. Configuring Training Parameters**

Model hyperparameters directly affect how well the model converges. The
**dataset size, tokenizer, and model architecture** all influence:

- **Learning Rate (`--lr`)**: Too high → exploding loss, too low → slow
  convergence.
- **Batch Size (`--batch-size`)**: Limited by memory constraints.
- **Optimizer (`--optimizer`)**: Supports **Adam, AdamW, SGD**.
- **Loss Function (`--criterion`)**: Choose between **MSE, MAE, CE**.

Example configuration:

```sh
python -m mini.cli.train \
    --dataset data/mini-owl.md \
    --processor models/tokenizer.model \
    --model models/mini.pt \
    --optimizer 'adamw' \
    --scheduler 'none' \
    --criterion 'cross_entropy' \
    --batch-size 2 --batch-stride 8 \
    --lr 1e-4 --epochs 100
```

During training, monitor logs for:

✅ **Steady loss decrease** → Healthy training  
⚠️ **Exploding loss** → Learning rate too high  
⚠️ **Plateauing loss** → Potential underfitting

## **6. Running Training**

Run the CLI tool to start training:

```sh
python -m mini.cli.train \
    --dataset data/mini-owl.md \
    --processor models/tokenizer.model \
    --model models/mini.pt \
    --batch-size 2 \
    --batch-stride 8 \
    --num-epochs 100 \
    --save-every 10 \
    --num-layers 4 \
    --num-heads 8 \
    --head-dim 16 \
    --embed-dim 256 \
    --lr 1e-4 \
    --scheduler none \
    --bias \
    --verbose
```

Here, we set a small number of layers and a decent number of heads to allow the
model to generalize appropriately to the Mini Owl dataset. Head and embedding
dimensions are conservative and set according to the expected model fit.

**Example Output:**

```sh
[Epoch: 100/100] [Avg Loss: 1.8174] [Perplexity: 6.155758]
Training complete! Checkpoint saved: models/mini.pt
```

Training speed depends on **GPU vs CPU**. Expect significantly longer runtimes
on CPU. The device is automatically determined at runtime.

## **7. Debugging Training Issues**

- **Vanishing gradients?** Try increasing the learning rate.
- **Exploding loss?** Reduce learning rate or check batch size.
- **Loss plateaus?** Consider adjusting optimizer or dataset size.
- **Poor convergence?** Ensure dataset and model size are balanced.

## **Next Steps**

After training, the next step is **evaluation and inference**, covered in
separate guides. This tutorial focuses on training fundamentals to get a working
model.

## **Conclusion**

This guide provides a **high-level** overview of training with Mini. The CLI
tool simplifies pretraining, and the choice of tokenizer, dataset, and
hyperparameters all impact model convergence.

For more advanced topics like evaluation, fine-tuning, and inference, see the
dedicated documentation.

### **References**

- [SentencePiece Documentation](https://github.com/google/sentencepiece)
- [Mini Framework Repository](https://github.com/teleprint-me/mini)
