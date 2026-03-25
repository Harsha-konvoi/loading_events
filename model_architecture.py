import numpy as np
import torch
import torch.nn as nn


# Transformer model classes
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0), :])

class ImprovedTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, n_heads=8, num_layers=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout=dropout, batch_first=True, norm_first=True),
            num_layers=num_layers
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.pre_norm(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = self.act(self.fc1(x))
        return self.fc2(x)  # shape: (batch_size, seq_len, num_classes)

class EnhancedLabelingStrategies:
    def __init__(self, window_size=120, step_size=26):
        self.window_size = window_size
        self.step_size = step_size

    def strategy_4_multi_scale_voting(self, df, window_sizes=[200, 400, 800]):
        sequences = []
        labels = []
        indices = []

        for ws in window_sizes:
            for start in range(0, len(df) - ws + 1, self.step_size):
                end = start + ws
                window = df.iloc[start:end][['accel_x', 'accel_y', 'accel_z']].values
                if len(window) == ws:
                    sequences.append(window)
                    labels.append("unknown")  # Placeholder, not used
                    indices.append((start, end))

        return sequences, labels, indices