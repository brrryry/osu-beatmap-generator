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
    Processes the sequence of hit note features to predict polar delta coordinates (sin, cos, distance)
    in an autoregressive manner using scheduled sampling and teacher forcing.
    """
    def __init__(self, input_dim=432, hidden_dim=128, lstm_layers=2, output_dim=3, dropout=0.2):
        super(LSTMSpatialModel, self).__init__()
        self.input_dim = input_dim
        
        # Unidirectional LSTM for autoregression
        self.lstm = nn.LSTM(
            input_size=input_dim + 3, # input features + 3 polar coordinates of previous step
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # Regression head for polar coordinates: sin(delta_theta), cos(delta_theta), distance
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, targets=None, teacher_forcing_ratio=0.0):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initial previous placement: [sin=0.0, cos=1.0, dist=0.0]
        prev_placement = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(batch_size, 1)
        
        preds = []
        hx = None
        
        import numpy as np
        
        for t in range(seq_len):
            step_feat = x[:, t, :] # (batch_size, input_dim)
            step_input = torch.cat([step_feat, prev_placement], dim=-1).unsqueeze(1) # (batch_size, 1, input_dim + 3)
            
            self.lstm.flatten_parameters()
            lstm_out, hx = self.lstm(step_input, hx) # lstm_out shape: (batch_size, 1, hidden_dim)
            
            pred_t = self.fc(lstm_out.squeeze(1)) # shape: (batch_size, 3)
            preds.append(pred_t.unsqueeze(1))
            
            # Scheduled sampling
            if self.training and targets is not None and np.random.rand() < teacher_forcing_ratio:
                prev_placement = targets[:, t, :]
            else:
                sin_p = pred_t[:, 0:1]
                cos_p = pred_t[:, 1:2]
                dist_p = torch.nn.functional.softplus(pred_t[:, 2:3])
                prev_placement = torch.cat([sin_p, cos_p, dist_p], dim=-1).detach()
                
        return torch.cat(preds, dim=1)

class CNNLSTMSpatialModel(nn.Module):
    """
    CNN-LSTM architecture for spatial hit object placement.
    Applies 1D convolutions over the frame-wise input features, followed by a unidirectional LSTM
    running autoregressively using scheduled sampling and teacher forcing.
    """
    def __init__(self, input_dim=432, cnn_channels=128, lstm_hidden=128, lstm_layers=2, output_dim=3, dropout=0.2):
        super(CNNLSTMSpatialModel, self).__init__()
        self.input_dim = input_dim
        
        # CNN Front-end (processes static sequence features frame-by-frame)
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
        
        # Unidirectional LSTM for autoregression, taking CNN features + previous placement
        self.lstm = nn.LSTM(
            input_size=cnn_channels + 3,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, output_dim)
        )

    def forward(self, x, targets=None, teacher_forcing_ratio=0.0):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x_trans = x.transpose(1, 2).contiguous()
        cnn_out = self.cnn(x_trans)
        
        # Transpose back for LSTM: (batch_size, seq_len, cnn_channels)
        lstm_in = cnn_out.transpose(1, 2).contiguous()
        
        # Initial previous placement: [sin=0.0, cos=1.0, dist=0.0]
        prev_placement = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(batch_size, 1)
        
        preds = []
        hx = None
        
        import numpy as np
        
        for t in range(seq_len):
            step_feat = lstm_in[:, t, :] # (batch_size, cnn_channels)
            step_input = torch.cat([step_feat, prev_placement], dim=-1).unsqueeze(1) # (batch_size, 1, cnn_channels + 3)
            
            self.lstm.flatten_parameters()
            lstm_out, hx = self.lstm(step_input, hx)
            
            pred_t = self.fc(lstm_out.squeeze(1))
            preds.append(pred_t.unsqueeze(1))
            
            # Scheduled sampling
            if self.training and targets is not None and np.random.rand() < teacher_forcing_ratio:
                prev_placement = targets[:, t, :]
            else:
                sin_p = pred_t[:, 0:1]
                cos_p = pred_t[:, 1:2]
                dist_p = torch.nn.functional.softplus(pred_t[:, 2:3])
                prev_placement = torch.cat([sin_p, cos_p, dist_p], dim=-1).detach()
                
        return torch.cat(preds, dim=1)

class TransformerSpatialModel(nn.Module):
    """
    Transformer Encoder architecture for spatial hit object placement.
    Non-autoregressive baseline.
    """
    def __init__(self, input_dim=432, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, output_dim=3, dropout=0.1):
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
