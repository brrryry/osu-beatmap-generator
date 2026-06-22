import math
import os
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
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class BandChannelAttention(nn.Module):
    """
    Channel Attention module that splits 84 Mel bins (and optional 84 Delta Mel bins)
    into N frequency bands and applies frame-wise attention.
    """
    def __init__(self, input_dim=169, num_bands=3):
        super(BandChannelAttention, self).__init__()
        self.input_dim = input_dim
        self.num_bands = num_bands
        self.enabled = True
        
        assert 84 % num_bands == 0, f"84 must be divisible by num_bands, got {num_bands}"
        self.band_dim = 84 // num_bands
        
        # Bottleneck size (minimum 2)
        bottleneck = max(2, num_bands // 2)
        
        # Attention for Mel
        self.fc_mel = nn.Sequential(
            nn.Linear(num_bands, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, num_bands),
            nn.Sigmoid()
        )
        
        # Attention for Delta Mel (if input has delta features)
        if input_dim >= 168:
            self.fc_delta = nn.Sequential(
                nn.Linear(num_bands, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, num_bands),
                nn.Sigmoid()
            )

    def forward(self, x):
        if not self.enabled:
            return x
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, dim = x.shape
        
        if dim < 84:
            # Fallback if input dimension is too small
            return x
            
        mel = x[:, :, :84]
        # Reshape to (batch_size, seq_len, num_bands, band_dim)
        mel_bands = mel.view(batch_size, seq_len, self.num_bands, self.band_dim)
        sq_mel = mel_bands.mean(dim=-1)
        att_mel = self.fc_mel(sq_mel)
        scaled_mel = (mel_bands * att_mel.unsqueeze(-1)).view(batch_size, seq_len, 84)
        
        if dim >= 168:
            delta = x[:, :, 84:168]
            timing = x[:, :, 168:]
            
            delta_bands = delta.view(batch_size, seq_len, self.num_bands, self.band_dim)
            sq_delta = delta_bands.mean(dim=-1)
            att_delta = self.fc_delta(sq_delta)
            scaled_delta = (delta_bands * att_delta.unsqueeze(-1)).view(batch_size, seq_len, 84)
            
            out = torch.cat([scaled_mel, scaled_delta, timing], dim=-1)
        else:
            timing = x[:, :, 84:]
            out = torch.cat([scaled_mel, timing], dim=-1)
            
        return out

class CNNLSTMRhythmModel(nn.Module):
    """
    CNN-LSTM architecture for rhythm transcription with multi-task heads.
    Determines if a note exists (fc_onset) and classifies the note type (fc_type).
    """
    def __init__(self, input_dim=85, cnn_channels=128, lstm_hidden=128, lstm_layers=2, num_classes=5, dropout=0.2, num_bands=3):
        super(CNNLSTMRhythmModel, self).__init__()
        
        self.attention = BandChannelAttention(input_dim=input_dim, num_bands=num_bands)
        
        # CNN Front-end (processes sequence frame-by-frame, keeping length same)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
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
        
        # Multi-task heads
        self.fc_onset = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 2) # Binary: None (0) or Note exists (1)
        )
        
        self.fc_type = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes - 1) # Note type classification (Circle, Slider, etc.)
        )

    def forward(self, x):
        # Apply Channel Attention
        x = self.attention(x)
        # Input shape: (batch_size, seq_len, input_dim)
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x_trans = x.transpose(1, 2).contiguous()
        
        cnn_out = self.cnn(x_trans)
        
        # Transpose back for LSTM: (batch_size, seq_len, cnn_channels)
        lstm_in = cnn_out.transpose(1, 2).contiguous()
        
        self.lstm.flatten_parameters()
        with torch.backends.cudnn.flags(enabled=False):
            lstm_out, _ = self.lstm(lstm_in)
        
        # Multi-task outputs
        logits_onset = self.fc_onset(lstm_out)
        logits_type = self.fc_type(lstm_out)
        return logits_onset, logits_type

class TransformerRhythmModel(nn.Module):
    """
    Transformer Encoder architecture for rhythm transcription with multi-task heads.
    """
    def __init__(self, input_dim=85, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, num_classes=5, dropout=0.1, num_bands=3, cnn_channels=0, lstm_hidden = 0):
        super(TransformerRhythmModel, self).__init__()
        
        self.attention = BandChannelAttention(input_dim=input_dim, num_bands=num_bands)
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
        
        self.fc_onset = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        self.fc_type = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes - 1)
        )

    def forward(self, x):
        # Apply Channel Attention
        x = self.attention(x)
        # Input shape: (batch_size, seq_len, input_dim)
        x_proj = self.input_projection(x)
        x_pos = self.pos_encoder(x_proj)
        
        trans_out = self.transformer_encoder(x_pos)
        
        logits_onset = self.fc_onset(trans_out)
        logits_type = self.fc_type(trans_out)
        return logits_onset, logits_type

class CNNTransformerRhythmModel(nn.Module):
    """
    CNN-Transformer architecture for rhythm transcription with multi-task heads.
    """
    def __init__(self, input_dim=85, cnn_channels=128, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, num_classes=5, dropout=0.1, num_bands=3):
        super(CNNTransformerRhythmModel, self).__init__()
        
        self.attention = BandChannelAttention(input_dim=input_dim, num_bands=num_bands)
        # CNN Front-end (same as CNN-LSTM front-end)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.input_projection = nn.Linear(cnn_channels, d_model) if cnn_channels != d_model else nn.Identity()
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
        
        self.fc_onset = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        self.fc_type = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes - 1)
        )

    def forward(self, x):
        # Apply Channel Attention
        x = self.attention(x)
        # Input shape: (batch_size, seq_len, input_dim)
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x_trans = x.transpose(1, 2).contiguous()
        
        cnn_out = self.cnn(x_trans)
        
        # Transpose back for Transformer: (batch_size, seq_len, cnn_channels)
        trans_in = cnn_out.transpose(1, 2).contiguous()
        
        x_proj = self.input_projection(trans_in)
        x_pos = self.pos_encoder(x_proj)
        
        trans_out = self.transformer_encoder(x_pos)
        
        logits_onset = self.fc_onset(trans_out)
        logits_type = self.fc_type(trans_out)
        return logits_onset, logits_type

def load_model_helper(model_class, checkpoint_path, device, num_classes, input_dim=169, default_num_bands=3):
    num_bands = default_num_bands
    cnn_channels = 128
    lstm_hidden = 128
    d_model = 128
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            if 'attention.fc_mel.0.weight' in ckpt:
                num_bands = ckpt['attention.fc_mel.0.weight'].shape[1]
                print(f"Auto-detected num_bands = {num_bands} from checkpoint.")
            if 'cnn.0.weight' in ckpt:
                cnn_channels = ckpt['cnn.0.weight'].shape[0]
                input_dim = ckpt['cnn.0.weight'].shape[1]
                print(f"Auto-detected cnn_channels = {cnn_channels}, input_dim = {input_dim} from checkpoint.")
            if 'lstm.weight_ih_l0' in ckpt:
                lstm_hidden = ckpt['lstm.weight_ih_l0'].shape[0] // 4
                print(f"Auto-detected lstm_hidden = {lstm_hidden} from checkpoint.")
            if 'input_projection.weight' in ckpt:
                d_model = ckpt['input_projection.weight'].shape[0]
                input_dim = ckpt['input_projection.weight'].shape[1]
                print(f"Auto-detected d_model = {d_model}, input_dim = {input_dim} from checkpoint.")
        except Exception as e:
            print(f"Warning: failed to auto-detect shapes from checkpoint: {e}")
            
    if model_class.__name__ == "CNNLSTMRhythmModel":
        model = model_class(
            input_dim=input_dim, 
            num_classes=num_classes, 
            num_bands=num_bands,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden
        ).to(device)
    elif model_class.__name__ == "CNNTransformerRhythmModel":
        model = model_class(
            input_dim=input_dim, 
            num_classes=num_classes, 
            num_bands=num_bands,
            cnn_channels=cnn_channels,
            d_model=d_model
        ).to(device)
    else:
        model = model_class(
            input_dim=input_dim, 
            num_classes=num_classes, 
            num_bands=num_bands
        ).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            missing_keys, unexpected_keys = model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
            if any(k.startswith("attention.") for k in missing_keys):
                print("Warning: Attention weights missing from checkpoint. Disabling Channel Attention layer.")
                model.attention.enabled = False
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            raise e
    return model
