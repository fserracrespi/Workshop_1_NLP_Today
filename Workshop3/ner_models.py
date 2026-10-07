"""NER model architectures used in the embedding comparison experiments.

The training, evaluation and plotting logic intentionally lives in the notebook.
This module only defines the neural architectures and a small model factory so
that the experimental notebook stays focused on the experiment itself.
"""

import torch
import torch.nn as nn


class LSTMNER(nn.Module):
    """Two-layer bidirectional LSTM for token-level NER classification."""

    def __init__(
        self,
        embedding_matrix,
        mode,
        padding_idx,
        num_labels,
        hidden_dim=128,
        dropout=0.3,
        num_layers=2,
    ):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.clone(),
            freeze=mode == "frozen",
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_matrix.size(1),
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, tokens):
        embeddings = self.embedding(tokens)
        features, _ = self.lstm(embeddings)
        return self.classifier(self.dropout(features))


class CNNNER(nn.Module):
    """One-dimensional convolutional token classifier for NER."""

    def __init__(
        self,
        embedding_matrix,
        mode,
        padding_idx,
        num_labels,
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.clone(),
            freeze=mode == "frozen",
            padding_idx=padding_idx,
        )

        self.conv = nn.Conv1d(
            in_channels=embedding_matrix.size(1),
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters, num_labels)

    def forward(self, tokens):
        embeddings = self.embedding(tokens)
        features = self.conv(embeddings.transpose(1, 2))
        features = torch.relu(features).transpose(1, 2)
        return self.classifier(self.dropout(features))


def build_model(
    architecture,
    embedding_matrix,
    mode,
    padding_idx,
    num_labels,
    hidden_dim=128,
    dropout=0.3,
    num_layers=2,
    num_filters=128,
    kernel_size=3,
):
    """Create the requested NER architecture with a shared configuration."""

    if architecture == "LSTM":
        return LSTMNER(
            embedding_matrix=embedding_matrix,
            mode=mode,
            padding_idx=padding_idx,
            num_labels=num_labels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
        )

    if architecture == "CNN":
        return CNNNER(
            embedding_matrix=embedding_matrix,
            mode=mode,
            padding_idx=padding_idx,
            num_labels=num_labels,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout=dropout,
        )

