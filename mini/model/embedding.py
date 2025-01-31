"""
Copyright © 2023 Austin Berrio
Module: mini.model.embedding
Description: A simple embedding model to handle semantic similarities and word representations.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor

from mini.common.json import JsonUtils
from mini.data.processor import JsonDataset, MiniDataProcessor, TensorDataset


# Define the Embedding class
class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        dropout_rate: float = 0.1,
        n_layers: int = 1,
        device: torch.device = None,
    ):
        """
        Initializes the embedding model.
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Number of neurons in the hidden layers.
            dropout_rate (float): Dropout rate for regularization.
            n_layers (int): Number of layers in the model.
        """
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Learnable embedding layer over the Llama embeddings
        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            device=device,
        )

        # Dynamically create hidden layers
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        self.embedding_dim if i == 0 else self.hidden_dim,
                        self.hidden_dim,
                        device=device,
                    ),
                    nn.LayerNorm(self.hidden_dim, device=device),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                )
                for i in range(n_layers)
            ]
        )

        # Final projection layer to map to the desired output embedding size
        self.projection = nn.Linear(hidden_dim, self.embedding_dim, device=device)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights of the model using Xavier uniform initialization.
        """
        nn.init.xavier_uniform_(self.embeddings.weight)
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer[0].weight)
            nn.init.uniform_(layer[0].bias, a=-0.1, b=0.1)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.uniform_(self.projection.bias, a=-0.1, b=0.1)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass to generate embeddings.

        Args:
            x (torch.Tensor): Input tensor of token IDs (batch_size, seq_len).
            padding_mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len),
                where 1 indicates valid tokens and 0 indicates padding tokens. Default is None.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, embedding_dim).
        """
        # Embedding lookup
        embeddings = self.embeddings(x)  # Shape: (batch_size, seq_len, embedding_dim)

        if padding_mask is not None:
            embeddings = embeddings * padding_mask.unsqueeze(-1)

        # Pass through hidden layers
        hidden = self.hidden_layers[0](embeddings)
        for i, layer in enumerate(self.hidden_layers[1:]):
            hidden = layer(hidden)

        # Project to output embedding space
        output_embeddings = self.projection(hidden)

        # Aggregate along sequence dimension (e.g., mean pooling)
        output_embeddings = output_embeddings.mean(dim=1)

        # Normalize embeddings for downstream tasks
        return F.normalize(output_embeddings, p=2, dim=1)

    def compute_similarity(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between query and document embeddings.
        Args:
            query_embeddings (torch.Tensor): Tensor of shape (num_queries, embedding_dim).
            document_embeddings (torch.Tensor): Tensor of shape (num_documents, embedding_dim).
        Returns:
            torch.Tensor: Pairwise similarity scores of shape (num_queries, num_documents).
        """
        if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
            raise ValueError("Both queries and documents must be 2D tensors.")

        # Normalize embeddings for cosine similarity
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        document_norm = F.normalize(document_embeddings, p=2, dim=1)

        # Compute pairwise cosine similarity
        return torch.mm(query_norm, document_norm.mT)


# Define the function to train the embedding model
def train_embedding_model(
    model_path: str,
    embedding_model: nn.Module,
    batched_dataset: TensorDataset,
    save_every: int = 10,
    epochs: int = 10,
    learning_rate: float = 1e-6,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-6,
    weight_decay: float = 0,
    amsgrad: bool = False,
) -> None:
    """
    Train the embedding model using a batched dataset of instruction-response pairs.

    Args:
        model_path (str): Path to save the trained model.
        embedding_model (nn.Module): The embedding model to train.
        batched_dataset (list[dict[str, torch.Tensor]]): List of batched data.
        save_every (int): Interval for saving the model.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        betas (tuple[float, float]): Adam optimizer's betas.
        eps (float): Adam optimizer's epsilon.
        weight_decay (float): Regularization parameter for optimizer.
        amsgrad (bool): Whether to use AMSGrad for the optimizer.
    """
    # Load the model if a checkpoint exists
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        embedding_model.load_state_dict(torch.load(model_path, weights_only=True))

    optimizer = torch.optim.Adam(
        embedding_model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    loss_fn = nn.CosineEmbeddingLoss()

    embedding_model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(batched_dataset):
            optimizer.zero_grad()

            # Extract instruction and response tokens
            input_tensors = batch["input"]  # Shape: (batch_size, seq_len)
            target_tensors = batch["target"]  # Shape: (batch_size, seq_len)

            # Generate embeddings
            instruction_embeddings = embedding_model(input_tensors)
            response_embeddings = embedding_model(target_tensors)

            # Normalize embeddings
            instruction_embeddings = F.normalize(instruction_embeddings, p=2, dim=1)
            response_embeddings = F.normalize(response_embeddings, p=2, dim=1)
            # All pairs are positive matches
            labels = torch.ones(len(instruction_embeddings), dtype=torch.float32)

            # Compute loss (minimize distance between instruction and response embeddings)
            loss = loss_fn(instruction_embeddings, response_embeddings, labels)

            # Backpropagation
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(embedding_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

        # Save the model periodically
        if (epoch + 1) % save_every == 0:
            torch.save(embedding_model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    # Final save after all epochs
    torch.save(embedding_model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")


# Function to evaluate the embedding model
def eval_embedding_model(
    embedding_model: nn.Module,
    batched_dataset: TensorDataset,
    json_dataset: JsonDataset,
    top_k: int = 3,
    eval_path: str = None,
    verbose: bool = False,
) -> None:
    """
    Evaluates the embedding model on a given dataset and computes similarities.

    Args:
        embedding_model (nn.Module): The trained embedding model.
        batched_dataset (TensorDataset): The dataset in batched tensor format.
        json_dataset (JsonDataset): The dataset in JSON format.
        top_k (int): Number of top matches to display per query.
        eval_path (str, optional): Path to save evaluation results. If None, results are printed.
        verbose (bool, optional): Whether to print evaluation results. Defaults to False.
    """
    embedding_model.eval()
    all_results = []

    with torch.no_grad():
        for i, batch in enumerate(batched_dataset):
            # Extract instruction and response tensors
            input_tensors = batch["input"]  # Shape: (batch_size, seq_len)
            target_tensors = batch["target"]  # Shape: (batch_size, seq_len)

            # Generate embeddings
            instruction_embeddings = embedding_model(
                input_tensors
            )  # Shape: (batch_size, embedding_dim)
            response_embeddings = embedding_model(
                target_tensors
            )  # Shape: (batch_size, embedding_dim)

            # Compute similarity scores
            similarities = embedding_model.compute_similarity(
                instruction_embeddings, response_embeddings
            )

            # Process results for each query in the batch
            for j in range(instruction_embeddings.size(0)):
                # Get the current query and response similarities
                query_similarities = similarities[i]  # Shape: (batch_size,)

                # Rank responses by similarity
                sorted_indices = query_similarities.argsort(descending=True)

                # Collect top-k matches
                query_results = {
                    "instruction": json_dataset[i * len(batch["input"]) + j][
                        "instruction"
                    ],
                    "top_matches": [
                        {
                            "response": json_dataset[i * len(batch["target"]) + k][
                                "response"
                            ],
                            "similarity": query_similarities[k].item(),
                        }
                        for rank, k in enumerate(sorted_indices[:top_k])
                    ],
                }
                all_results.append(query_results)

    # Output results
    if eval_path:
        json_utils = JsonUtils(verbose=verbose)
        json_utils.save_json(file_path=eval_path, data=all_results)
    elif verbose:
        for result in all_results:
            print(f"Instruction: {result['instruction']}")
            for rank, match in enumerate(result["top_matches"]):
                print(
                    f"  Rank {rank + 1}: {match['response']} (Similarity: {match['similarity']:.4f})"
                )
    else:
        print(
            "Verbosity is disabled and no evaluation path was provided. Results will not be saved."
        )
        print("Use the --verbose flag to see the results in the console.")
        print("Use the --eval-path flag to save the results to a file.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the embedding model."
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to save or load the model."
    )
    parser.add_argument(
        "--processor-path", required=True, help="Path to SentencePiece processor."
    )
    parser.add_argument("--input", required=True, help="Path to input JSON dataset.")
    parser.add_argument("--schema-path", help="Path to JSON schema file.")
    parser.add_argument("--eval-output", help="Path to save evaluation results.")
    parser.add_argument(
        "--add-bos",
        action="store_false",
        help="Add BOS token to input (default: True).",
    )
    parser.add_argument(
        "--add-eos",
        action="store_false",
        help="Add EOS token to input (default: True).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for training (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--max-length", type=int, default=256, help="Maximum sequence length."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training."
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=256, help="Embedding dimension."
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden layer dimension."
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument(
        "--n-layers",
        type=int,
        default=1,
        help="Number of layers in the embedding model.",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save model every N epochs."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-6, help="Learning rate for optimizer."
    )
    parser.add_argument(
        "--betas",
        nargs=2,
        type=float,
        default=(0.9, 0.999),
        help="Betas for Adam optimizer.",
    )
    parser.add_argument(
        "--eps", type=float, default=1e-6, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0, help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--amsgrad", action="store_true", help="Use AMSGrad variant of Adam optimizer."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top matches to display during evaluation.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


# Example Usage:
if __name__ == "__main__":
    args = parse_args()

    # Load JSON schema if provided
    json_utils = JsonUtils(verbose=args.verbose)
    json_dataset = json_utils.load_json(args.input)

    # Load schema from file if provided
    schema = None
    if args.schema_path:
        schema = json_utils.load_json(args.schema_path)

    # Validate JSON data if provided schema
    json_utils.validate_json(json_dataset, schema=schema)

    # Set device type
    device = {
        "cpu": torch.device("cpu"),
        "cuda": (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    }.get(args.device, torch.device("cpu"))

    # Initialize SentencePieceProcessor
    processor = SentencePieceProcessor(args.processor_path)

    # Initialize MiniDataProcessor
    mini_data_processor = MiniDataProcessor(
        processor=processor,
        verbose=args.verbose,
    )

    # Tokenize the dataset
    encoded_dataset = mini_data_processor.tokenize(
        json_dataset=json_dataset,
        max_length=args.max_length,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )

    # Batch the dataset
    batched_dataset = mini_data_processor.batch(
        encoded_dataset=encoded_dataset,
        batch_size=args.batch_size,
        device=device,
    )
    for batch in batched_dataset:
        if (batch["input"] < 0).any() or (
            batch["input"] >= processor.vocab_size()
        ).any():
            print(f"Invalid indices in input: {batch['input']}")
        if (batch["target"] < 0).any() or (
            batch["target"] >= processor.vocab_size()
        ).any():
            print(f"Invalid indices in target: {batch['target']}")

    # Initialize MiniEmbedding model
    embedding_model = Embedding(
        vocab_size=processor.vocab_size(),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout,
        n_layers=args.n_layers,
        device=device,
    )

    try:
        # Train the model
        train_embedding_model(
            model_path=args.model_path,
            embedding_model=embedding_model,
            batched_dataset=batched_dataset,
            save_every=args.save_every,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad,
        )
    except KeyboardInterrupt:
        print("Model training interrupted by user.")

    try:
        eval_embedding_model(
            embedding_model=embedding_model,
            batched_dataset=batched_dataset,
            json_dataset=json_dataset,
            top_k=args.top_k,
            eval_path=args.eval_output,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("Model training interrupted by user.")
