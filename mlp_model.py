import torch.nn as nn

class MLPStockModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPStockModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),

            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),

            nn.Linear(hidden_size // 4, output_size)
        )

    def forward(self, x):
        return self.model(x)
