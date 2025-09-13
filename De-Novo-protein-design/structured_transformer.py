
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import math

class GraphAttentionLayer(MessagePassing):
    """Graph attention mechanism for spatial k-nearest neighbors"""

    def __init__(self, in_dim, out_dim, heads=8, dropout=0.1):
        super().__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout

        # Linear transformations for query, key, value
        self.W_q = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim * heads, bias=False)

        # Edge embedding for spatial relationships
        self.edge_mlp = nn.Sequential(
            nn.Linear(3, out_dim),  # 3D distance vector
            nn.ReLU(),
            nn.Linear(out_dim, out_dim * heads)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_dim] node features
        # edge_index: [2, E] edge connectivity
        # edge_attr: [E, 3] 3D distance vectors

        N = x.size(0)

        # Compute queries, keys, values
        q = self.W_q(x).view(N, self.heads, self.out_dim)
        k = self.W_k(x).view(N, self.heads, self.out_dim)
        v = self.W_v(x).view(N, self.heads, self.out_dim)

        # Process edge features
        edge_emb = self.edge_mlp(edge_attr).view(-1, self.heads, self.out_dim)

        return self.propagate(edge_index, q=q, k=k, v=v, edge_emb=edge_emb)

    def message(self, q_i, k_j, v_j, edge_emb, edge_index_i):
        # Compute attention scores with edge information
        attn = (q_i * (k_j + edge_emb)).sum(dim=-1) / math.sqrt(self.out_dim)
        attn = F.softmax(attn, dim=1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        return (attn.unsqueeze(-1) * v_j).sum(dim=1)

class StructuredTransformerEncoder(nn.Module):
    """Encoder for 3D protein structure representation"""

    def __init__(self, node_dim=128, edge_dim=64, num_layers=6, num_heads=8):
        super().__init__()
        self.node_dim = node_dim
        self.num_layers = num_layers

        # Initial node embedding
        self.node_embedding = nn.Linear(20, node_dim)  # 20 amino acid types

        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(node_dim, node_dim // num_heads, heads=num_heads)
            for _ in range(num_layers)
        ])

        # Layer normalization and feedforward
        self.layer_norms = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(num_layers)])
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim, node_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(node_dim * 4, node_dim)
            ) for _ in range(num_layers)
        ])

    def forward(self, node_features, edge_index, edge_attr):
        x = self.node_embedding(node_features)

        for i, (gat, norm, ff) in enumerate(zip(self.gat_layers, self.layer_norms, self.feed_forwards)):
            # Graph attention with residual connection
            x_new = gat(x, edge_index, edge_attr)
            x = norm(x + x_new)

            # Feed forward with residual connection
            x_new = ff(x)
            x = norm(x + x_new)

        return x

class AutoregressiveDecoder(nn.Module):
    """Autoregressive decoder for sequence generation"""

    def __init__(self, structure_dim=128, vocab_size=20, max_len=512):
        super().__init__()
        self.structure_dim = structure_dim
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Position embedding
        self.pos_embedding = nn.Embedding(max_len, structure_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=structure_dim, nhead=8, dim_feedforward=512, dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Output projection
        self.output_projection = nn.Linear(structure_dim, vocab_size)

    def forward(self, structure_repr, tgt_sequence=None):
        batch_size, seq_len = structure_repr.shape[:2]

        if tgt_sequence is not None:
            # Training mode - use target sequence
            pos_ids = torch.arange(seq_len, device=structure_repr.device)
            pos_emb = self.pos_embedding(pos_ids).unsqueeze(0).expand(batch_size, -1, -1)

            # Create causal mask for autoregressive generation
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            tgt_mask = tgt_mask.to(structure_repr.device)

            # Combine target sequence with positional encoding
            tgt_emb = tgt_sequence + pos_emb

            # Decode
            output = self.transformer_decoder(
                tgt_emb.transpose(0, 1),
                structure_repr.transpose(0, 1),
                tgt_mask=tgt_mask
            )

            return self.output_projection(output.transpose(0, 1))
        else:
            # Inference mode - autoregressive generation
            generated = []
            for i in range(seq_len):
                pos_id = torch.tensor([i], device=structure_repr.device)
                pos_emb = self.pos_embedding(pos_id)

                if i == 0:
                    tgt_emb = pos_emb.unsqueeze(0)
                else:
                    prev_tokens = torch.stack(generated, dim=1)
                    prev_pos = torch.arange(i, device=structure_repr.device)
                    prev_pos_emb = self.pos_embedding(prev_pos).unsqueeze(0)
                    tgt_emb = prev_tokens + prev_pos_emb

                output = self.transformer_decoder(
                    tgt_emb.transpose(0, 1),
                    structure_repr[:, :i+1].transpose(0, 1)
                )

                logits = self.output_projection(output[-1])
                next_token = F.softmax(logits, dim=-1)
                generated.append(next_token)

            return torch.stack(generated, dim=1)

class StructuredTransformer(nn.Module):
    """Complete Structured Transformer for inverse protein folding"""

    def __init__(self, node_dim=128, num_heads=8, num_layers=6, vocab_size=20):
        super().__init__()
        self.encoder = StructuredTransformerEncoder(node_dim, num_layers=num_layers, num_heads=num_heads)
        self.decoder = AutoregressiveDecoder(node_dim, vocab_size)

    def forward(self, node_features, edge_index, edge_attr, tgt_sequence=None):
        # Encode 3D structure
        structure_repr = self.encoder(node_features, edge_index, edge_attr)

        # Decode to sequence
        sequence_logits = self.decoder(structure_repr, tgt_sequence)

        return sequence_logits

# Diffusion model components
class NoiseScheduler:
    """Denoising diffusion probabilistic model scheduler"""

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t, noise=None):
        """Add noise to clean data"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])

        return sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise

    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps for training"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

class DiffusionModel(nn.Module):
    """Denoising diffusion model for protein sequence generation"""

    def __init__(self, structured_transformer):
        super().__init__()
        self.model = structured_transformer
        self.noise_scheduler = NoiseScheduler()

    def forward(self, node_features, edge_index, edge_attr, sequence, timesteps):
        # Add noise to sequence based on timesteps
        noise = torch.randn_like(sequence)
        noisy_sequence = self.noise_scheduler.add_noise(sequence, timesteps, noise)

        # Predict noise
        predicted_noise = self.model(node_features, edge_index, edge_attr, noisy_sequence)

        return predicted_noise, noise

    def sample(self, node_features, edge_index, edge_attr, num_steps=50):
        """Generate sequence using denoising process"""
        batch_size, seq_len = node_features.shape[0], node_features.shape[1]

        # Start from pure noise
        x = torch.randn(batch_size, seq_len, 20, device=node_features.device)

        for t in reversed(range(0, self.noise_scheduler.num_timesteps, 
                               self.noise_scheduler.num_timesteps // num_steps)):
            timestep = torch.full((batch_size,), t, device=node_features.device)

            # Predict noise
            with torch.no_grad():
                predicted_noise = self.model(node_features, edge_index, edge_attr, x)

            # Remove predicted noise
            alpha = self.noise_scheduler.alphas[t]
            alpha_cumprod = self.noise_scheduler.alphas_cumprod[t]
            beta = self.noise_scheduler.betas[t]

            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)

            # Add noise (except for the last step)
            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta) * noise

        return F.softmax(x, dim=-1)
