import torch
import torch.nn as nn


class VehicleMotionTransformer(nn.Module):
    """
    Transformer model for predicting vehicle motion trajectories.

    This model takes time sequences and context parameters as input to predict the vehicle's
    displacement, velocity, and acceleration over time. It uses a transformer architecture
    to capture temporal dependencies in the motion.

    Args:
        d_model (int): Dimension of the model's internal representations. Default: 16
        num_heads (int): Number of attention heads in transformer layers. Default: 2
        num_layers (int): Number of transformer encoder layers. Default: 2
    """

    def __init__(self, d_model=16, num_heads=2, num_layers=2):
        super().__init__()
        # Time embedding: maps scalar time values to d_model dimensions
        self.time_embedding = nn.Linear(1, d_model)

        # Context embedding: maps 5D context vector [x₀, v₀, A, ω, m] to d_model dimensions
        self.context_embedding = nn.Linear(5, d_model)

        # Projects concatenated embeddings back to d_model dimensions
        self.input_projection = nn.Linear(2 * d_model, d_model)

        # Transformer encoder for processing temporal sequences
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection to predict [x, v, a] at each timestep
        self.output_layer = nn.Linear(d_model, 3)

    def forward(self, time_seq, context):
        """
        Forward pass of the model.

        Args:
            time_seq (torch.Tensor): Time sequence tensor of shape (batch, seq_len, 1)
            context (torch.Tensor): Context parameters tensor of shape (batch, 5) containing
                                  initial conditions and physical parameters [x₀, v₀, A, ω, m]

        Returns:
            torch.Tensor: Predicted motion states of shape (batch, seq_len, 3) containing
                         [displacement, velocity, acceleration] for each timestep
        """
        # Embed time sequence to higher dimensions
        time_emb = self.time_embedding(
            time_seq
        )  # Shape: (batch, seq_len, d_model)

        # Embed context and broadcast along sequence dimension
        context_emb = self.context_embedding(
            context
        )  # Shape: (batch, d_model)
        context_emb_expanded = context_emb.unsqueeze(1).expand(
            -1, time_seq.size(1), -1
        )  # Shape: (batch, seq_len, d_model)

        # Combine time and context information
        combined = torch.cat([time_emb, context_emb_expanded], dim=-1)
        combined = self.input_projection(combined)

        # Process through transformer (requires seq_len first)
        combined = combined.permute(
            1, 0, 2
        )  # Shape: (seq_len, batch, d_model)
        encoded = self.transformer_encoder(combined)
        encoded = encoded.permute(1, 0, 2)  # Shape: (batch, seq_len, d_model)

        # Generate final predictions
        output = self.output_layer(encoded)  # Shape: (batch, seq_len, 3)
        return output
