import torch
import torch.nn as nn


class VehicleMotionTransformer(nn.Module):
    def __init__(self, d_model=16, num_heads=2, num_layers=2):
        super().__init__()
        # Embed a single time scalar into a d_model-dimensional vector.
        self.time_embedding = nn.Linear(1, d_model)
        # Embed the context vector (which includes [x0, v0, A, omega, m]) into d_model dimensions.
        self.context_embedding = nn.Linear(5, d_model)
        # After concatenating the time and context embeddings, project back to d_model.
        self.input_projection = nn.Linear(2 * d_model, d_model)

        # Transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Final layer to predict 3 outputs: displacement, velocity, acceleration.
        self.output_layer = nn.Linear(d_model, 3)

    def forward(self, time_seq, context):
        """
        Args:
            time_seq: Tensor of shape (batch, seq_len, 1)
            context: Tensor of shape (batch, 5)
        Returns:
            Tensor of shape (batch, seq_len, 3)
        """
        # Embed the time sequence.
        time_emb = self.time_embedding(time_seq)  # (batch, seq_len, d_model)

        # Embed the context and expand it along the sequence dimension.
        context_emb = self.context_embedding(context)  # (batch, d_model)
        context_emb_expanded = context_emb.unsqueeze(1).expand(
            -1, time_seq.size(1), -1
        )

        # Concatenate the time and context embeddings.
        combined = torch.cat([time_emb, context_emb_expanded], dim=-1)
        combined = self.input_projection(combined)

        # Transformer expects input of shape (seq_len, batch, d_model)
        combined = combined.permute(1, 0, 2)
        encoded = self.transformer_encoder(combined)
        encoded = encoded.permute(1, 0, 2)  # (batch, seq_len, d_model)

        # Project to the outputs (displacement, velocity, acceleration)
        output = self.output_layer(encoded)
        return output
