import torch
import torch.nn as nn

class GRUStockModel(nn.Module):
    """
    GRU-based model for stock movement classification.

    This model takes as input a sequence of feature vectors (e.g., stock data, economic indicators,
    and optional document embeddings) over a fixed window size and outputs class probabilities
    corresponding to stock movement: up, down, or stable.

    Args:
        input_size (int): Number of features in the input sequence (e.g., 7 or 7 + doc embedding dim).
        hidden_size (int): Number of hidden units in the GRU layer.
        num_layers (int): Number of stacked GRU layers.
        output_size (int): Number of output classes (typically 3 for up/down/stable).
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUStockModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the GRU model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size),
                          representing class logits for each sequence in the batch.
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # Use last time step's output for classification
        return out
