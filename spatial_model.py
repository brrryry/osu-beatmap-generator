import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer model."""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LSTMSpatialModel(nn.Module):
    """
    LSTM architecture for spatial hit object placement.
    Processes the sequence of hit note features to predict polar delta coordinates (sin, cos, distance).
    """
    def __init__(self, input_dim=425, hidden_dim=128, lstm_layers=2, output_dim=3, dropout=0.2):
        super(LSTMSpatialModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # Regression head for polar coordinates: sin(delta_theta), cos(delta_theta), distance
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        self.lstm.flatten_parameters()
        with torch.backends.cudnn.flags(enabled=False):
            lstm_out, _ = self.lstm(x)
        
        # Output: (batch_size, seq_len, output_dim)
        out = self.fc(lstm_out)
        return out

class CNNLSTMSpatialModel(nn.Module):
    """
    CNN-LSTM architecture for spatial hit object placement.
    Applies 1D convolutions over the frame-wise input features, followed by a BiLSTM and linear projection.
    """
    def __init__(self, input_dim=425, cnn_channels=128, lstm_hidden=128, lstm_layers=2, output_dim=3, dropout=0.2):
        super(CNNLSTMSpatialModel, self).__init__()
        
        # CNN Front-end
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, output_dim)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x_trans = x.transpose(1, 2).contiguous()
        cnn_out = self.cnn(x_trans)
        
        # Transpose back for LSTM: (batch_size, seq_len, cnn_channels)
        lstm_in = cnn_out.transpose(1, 2).contiguous()
        
        self.lstm.flatten_parameters()
        with torch.backends.cudnn.flags(enabled=False):
            lstm_out, _ = self.lstm(lstm_in)
            
        out = self.fc(lstm_out)
        return out

class TransformerSpatialModel(nn.Module):
    """
    Transformer Encoder architecture for spatial hit object placement.
    """
    def __init__(self, input_dim=425, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, output_dim=3, dropout=0.1):
        super(TransformerSpatialModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x_proj = self.input_projection(x)
        x_pos = self.pos_encoder(x_proj)
        
        trans_out = self.transformer_encoder(x_pos)
        out = self.fc(trans_out)
        return out
