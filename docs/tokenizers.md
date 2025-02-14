# **Tokenization with SentencePiece**

## **1. Introduction**

Tokenization is a crucial step in training and using language models. It breaks
down raw text into a format that models can process efficiently. Traditional
tokenization methods rely on splitting text by words or characters, but these
approaches struggle with large vocabularies, rare words, and multilingual
support.

Mini uses **SentencePiece**, a subword-based tokenization method designed to
handle diverse datasets effectively. SentencePiece allows **vocabulary control,
efficient compression, and seamless handling of rare words** while avoiding
issues with word-based tokenization.

This document provides an overview of how SentencePiece is used in Mini, how to
train a tokenizer, and how tokenization impacts training.

## **2. What is SentencePiece?**

SentencePiece is a data-driven, unsupervised tokenization algorithm developed by
Google. Unlike traditional tokenizers, it treats input text as a **raw byte
stream** and does not rely on language-specific tokenization rules. Instead, it
learns subword units from the dataset itself.

### **Why SentencePiece?**

- **Subword-based**: Efficiently handles out-of-vocabulary (OOV) words.
- **No need for custom preprocessing**: Works on raw text without requiring
  prior tokenization.
- **Supports multiple languages**: Handles spaces and punctuation without
  predefined rules.
- **Compact vocabulary**: Reduces memory footprint while maintaining efficiency.

SentencePiece supports different tokenization models, including:

- **Byte Pair Encoding (BPE)**
- **Unigram Language Model**
- **Character-based Tokenization**

Mini’s default tokenizer uses **BPE**, a widely used method for subword
tokenization.

## **3. Training a SentencePiece Model**

If you need a custom tokenizer, you can train one using SentencePiece. The
following command trains a tokenizer with a vocabulary size of 32,000:

```sh
spm_train --input=data.txt --model_prefix=tokenizer --vocab_size=32000 --character_coverage=1.0 --model_type=bpe
```

### **Key Parameters:**

- `--input`: The dataset file used for training.
- `--model_prefix`: Name for the output model files (`tokenizer.model`,
  `tokenizer.vocab`).
- `--vocab_size`: The number of tokens in the final vocabulary.
- `--character_coverage`: Determines how much of the dataset's characters should
  be included.
- `--model_type`: Tokenization algorithm (options: `bpe`, `unigram`, `char`,
  `word`).

The resulting model can then be loaded into Mini for tokenization.

## **4. Using a Pretrained Tokenizer in Mini**

Mini uses a **pretrained SentencePiece tokenizer** by default, allowing
immediate use without retraining. The pretrained model is compatible with
Mistral-based architectures and is automatically loaded during training and
inference.

To manually tokenize a file using Mini’s tokenizer CLI tool:

```sh
python -m mini.cli.tokenizer --model models/tokenizer.model --input sample.txt
```

This outputs the tokenized representation of the text.

## **5. Tokenizer Special Tokens & IDs**

SentencePiece assigns **unique token IDs** to different types of tokens,
including special ones:

| Token Type        | Symbol  | ID  |
| ----------------- | ------- | --- |
| Padding           | `<pad>` | 0   |
| Unknown           | `<unk>` | 1   |
| Start of Sequence | `<s>`   | 2   |
| End of Sequence   | `</s>`  | 3   |

### **Handling Special Tokens in Training**

- `<pad>` ensures sequences have uniform length during batching.
- `<unk>` replaces words that do not exist in the vocabulary.
- `<s>` and `</s>` mark the start and end of sequences, aiding structured
  inputs.

When training on large datasets, proper handling of these tokens **prevents
unintended model behavior**.

## **6. Tokenization in Training**

Once text is tokenized, it is formatted for training in Mini. The framework
relies on **three key parameters** for structured batch processing:

- **`--max-seq-len`**: Maximum length of a sequence. Shorter sequences are
  padded.
- **`--batch-size`**: Number of sequences processed at once.
- **`--batch-stride`**: Defines the step size for batch sequencing.

Mini applies a **sliding window approach**, shifting sequences by the batch
stride. This ensures continuous training samples while maintaining efficient
memory usage.

## **7. Debugging Tokenization**

Mini provides a **tokenizer utility** to inspect tokenized outputs, check
vocabulary sizes, and analyze sequence lengths.

#### **Check vocabulary size:**

```sh
python -m mini.cli.tokenizer --model models/tokenizer.model --vocab-size
```

#### **Get tokenized sequence length:**

```sh
python -m mini.cli.tokenizer --model models/tokenizer.model --input sample.txt --seq-length
```

#### **Print tokenized output:**

```sh
python -m mini.cli.tokenizer --model models/tokenizer.model --input sample.txt --clip 10
```

These commands help verify whether text is tokenized correctly before training.

## **8. Experimenting with the Dataset Loader**

Mini includes a **dataset loader utility** for inspecting how text is batched
and tokenized before training. The loader processes input text according to
specified **batch size, stride, and sequence length**, allowing fine-tuned
control over training data.

To inspect dataset processing behavior:

```sh
python -m mini.cli.tokenizer --model models/tokenizer.model --input sample.txt --loader --verbose
```

This prints how Mini structures input sequences:

```sh
Batch 1:
  Input  | seq_len=128 | [1, 422, 415, 14100, 2920, ...]
  Target | seq_len=128 | [422, 415, 14100, 2920, 302, ...]
```

### **What Does This Show?**

- **Sliding Window Processing**: Each new batch overlaps with the previous by
  the batch stride value.
- **Batch Size Impact**: Increasing batch size increases the number of sequences
  processed simultaneously.
- **Effect of `batch-stride`**: Adjusting this value shifts sequences more or
  less aggressively.

### **Example: Adjusting Stride for Different Overlaps**

```sh
python -m mini.cli.tokenizer --model models/tokenizer.model --input sample.txt --loader --verbose --batch-stride 8
```

This will process the dataset using a **smaller stride**, increasing **token
overlap** between sequences.

```sh
python -m mini.cli.tokenizer --model models/tokenizer.model --input sample.txt --loader --verbose --batch-stride 64
```

This will process the dataset using a **larger stride**, reducing **overlap**
without skipping any tokens.

By experimenting with these parameters, you can fine-tune the **sequence
structure for optimal training efficiency**.

## **Conclusion**

Tokenization is a **critical** step in NLP workflows, influencing model
efficiency and performance. Mini's use of **SentencePiece** allows for flexible,
efficient tokenization that works across different dataset sizes and languages.

When training models, **carefully selecting tokenizer parameters, handling
special tokens, and debugging tokenized sequences** ensures better training
stability and accuracy.

For more details, refer to:

- [SentencePiece GitHub](https://github.com/google/sentencepiece)
- [Mini Framework Repository](https://github.com/teleprint-me/mini)
