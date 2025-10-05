"""
Deep Learning Models for Trading Signal Generation
Implements SOTA architectures: TCN, Transformer, LSTM
Optimized for RTX 4070 GPU (12GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# TEMPORAL CONVOLUTIONAL NETWORK (TCN) - SOTA for Time Series
# ============================================================================

class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated causal convolutions
    Paper: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks"
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, 
                 stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        
        # Two convolutional layers with dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for trading signal prediction
    Target accuracy: 55-57% (SOTA for time series)
    
    Architecture:
    - 6 temporal blocks with exponentially increasing dilation
    - Kernel size: 3
    - Receptive field: covers 50+ time steps
    - Output: 3-class (buy/hold/sell) with softmax
    """
    def __init__(self, input_dim: int, num_channels: list = None, 
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        
        if num_channels is None:
            # Default: 6 layers with 64 channels each
            num_channels = [64] * 6
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation: 1, 2, 4, 8, 16, 32
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size, padding=padding,
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        
        # Output layers
        self.fc1 = nn.Linear(num_channels[-1], 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 3)  # 3 classes: buy, hold, sell
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            logits: (batch_size, 3)
        """
        # Transpose for Conv1d: (batch, channels, length)
        x = x.transpose(1, 2)
        
        # TCN layers
        y = self.network(x)
        
        # Global average pooling over time dimension
        y = y.mean(dim=2)
        
        # Fully connected layers
        y = F.relu(self.fc1(y))
        y = self.dropout(y)
        logits = self.fc2(y)
        
        return logits


# ============================================================================
# TRANSFORMER MODEL - Multi-Head Attention
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    """
    Transformer encoder for trading signal prediction
    Target accuracy: 54-56%
    
    Architecture:
    - 4 transformer encoder layers
    - 8 attention heads
    - 256 hidden dimensions
    - Positional encoding
    - Output: 3-class (buy/hold/sell)
    """
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.2):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            logits: (batch_size, 3)
        """
        # Project input to d_model dimensions
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Use [CLS] token (first position) or mean pooling
        x = x.mean(dim=1)  # Mean pooling over sequence
        
        # Output layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


# ============================================================================
# LSTM MODEL - Baseline Comparison
# ============================================================================

class LSTMModel(nn.Module):
    """
    LSTM model for trading signals (baseline)
    Target accuracy: 52-54%
    
    Architecture:
    - 2 LSTM layers with 128 hidden units
    - Dropout for regularization
    - Output: 3-class prediction
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            logits: (batch_size, 3)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        x = h_n[-1]  # Last layer's hidden state
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


# ============================================================================
# ENSEMBLE MODEL - Combine All Models
# ============================================================================

class EnsembleModel(nn.Module):
    """
    Ensemble of TCN + Transformer + LSTM
    Target accuracy: 56-58% (SOTA)
    
    Combines predictions using:
    1. Simple averaging (baseline)
    2. Weighted averaging (learned weights)
    3. Stacking with meta-learner
    """
    def __init__(self, input_dim: int, ensemble_method: str = 'weighted'):
        super().__init__()
        
        self.ensemble_method = ensemble_method
        
        # Individual models
        self.tcn = TCNModel(input_dim)
        self.transformer = TransformerModel(input_dim)
        self.lstm = LSTMModel(input_dim)
        
        if ensemble_method == 'weighted':
            # Learnable weights (sum to 1)
            self.weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]))
        
        elif ensemble_method == 'stacking':
            # Meta-learner: takes 9 features (3 logits × 3 models)
            self.meta_learner = nn.Sequential(
                nn.Linear(9, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 3)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            logits: (batch_size, 3)
        """
        # Get predictions from all models
        tcn_logits = self.tcn(x)
        transformer_logits = self.transformer(x)
        lstm_logits = self.lstm(x)
        
        if self.ensemble_method == 'average':
            # Simple averaging
            logits = (tcn_logits + transformer_logits + lstm_logits) / 3
            
        elif self.ensemble_method == 'weighted':
            # Weighted averaging with learned weights
            weights = F.softmax(self.weights, dim=0)
            logits = (weights[0] * tcn_logits + 
                     weights[1] * transformer_logits + 
                     weights[2] * lstm_logits)
            
        elif self.ensemble_method == 'stacking':
            # Stacking: concatenate predictions and learn from meta-learner
            combined = torch.cat([tcn_logits, transformer_logits, lstm_logits], dim=1)
            logits = self.meta_learner(combined)
        
        return logits


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader, num_epochs: int = 50,
                learning_rate: float = 0.001, device: str = 'cuda',
                patience: int = 10) -> dict:
    """
    Train deep learning model with early stopping
    
    Returns:
        dict with training history and best metrics
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    logger.info(f"Training on {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Logging
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                       f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                       f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'results/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('results/best_model.pth'))
    
    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'final_epoch': epoch + 1
    }


def evaluate_model(model: nn.Module, test_loader: torch.utils.data.DataLoader,
                   device: str = 'cuda') -> dict:
    """Evaluate model on test set"""
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels)
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


if __name__ == "__main__":
    # Test model instantiation
    print("Testing model architectures...")
    
    input_dim = 15  # Number of features
    seq_len = 20    # Sequence length
    batch_size = 32
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test TCN
    print("\n1. TCN Model:")
    tcn = TCNModel(input_dim)
    tcn_out = tcn(x)
    print(f"   Input: {x.shape} → Output: {tcn_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in tcn.parameters()):,}")
    
    # Test Transformer
    print("\n2. Transformer Model:")
    transformer = TransformerModel(input_dim)
    trans_out = transformer(x)
    print(f"   Input: {x.shape} → Output: {trans_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Test LSTM
    print("\n3. LSTM Model:")
    lstm = LSTMModel(input_dim)
    lstm_out = lstm(x)
    print(f"   Input: {x.shape} → Output: {lstm_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in lstm.parameters()):,}")
    
    # Test Ensemble
    print("\n4. Ensemble Model:")
    ensemble = EnsembleModel(input_dim, ensemble_method='stacking')
    ensemble_out = ensemble(x)
    print(f"   Input: {x.shape} → Output: {ensemble_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in ensemble.parameters()):,}")
    
    print("\n✓ All models initialized successfully!")
