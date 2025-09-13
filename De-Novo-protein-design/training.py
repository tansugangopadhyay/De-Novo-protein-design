
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score

class ProteinDataset(Dataset):
    """Dataset for protein structure-sequence pairs"""

    def __init__(self, data_path, max_length=512):
        self.max_length = max_length
        self.data = self._load_data(data_path)

        # Amino acid vocabulary
        self.aa_vocab = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }
        self.vocab_size = len(self.aa_vocab)

    def _load_data(self, data_path):
        """Load preprocessed protein data"""
        # This would load from PDB files processed into graph format
        # For demo purposes, create synthetic data
        return self._create_synthetic_data()

    def _create_synthetic_data(self):
        """Create synthetic protein data for demonstration"""
        data = []
        for i in range(1000):  # 1000 synthetic proteins
            length = np.random.randint(50, 300)  # Variable length proteins

            # Synthetic 3D coordinates (Cα atoms)
            coords = np.random.randn(length, 3) * 10

            # Synthetic sequence
            sequence = np.random.randint(0, 20, length)

            # Create k-nearest neighbor graph (k=32)
            distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
            k_neighbors = np.argsort(distances, axis=1)[:, :32]  # 32 nearest neighbors

            edge_index = []
            edge_attr = []

            for i_node in range(length):
                for j_node in k_neighbors[i_node]:
                    if i_node != j_node:
                        edge_index.append([i_node, j_node])
                        edge_attr.append(coords[j_node] - coords[i_node])

            data.append({
                'coords': coords,
                'sequence': sequence,
                'edge_index': np.array(edge_index).T,
                'edge_attr': np.array(edge_attr),
                'length': length
            })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Convert to tensors
        coords = torch.FloatTensor(item['coords'])
        sequence = torch.LongTensor(item['sequence'])
        edge_index = torch.LongTensor(item['edge_index'])
        edge_attr = torch.FloatTensor(item['edge_attr'])

        # Create one-hot encoding for sequence
        sequence_onehot = torch.zeros(len(sequence), self.vocab_size)
        sequence_onehot.scatter_(1, sequence.unsqueeze(1), 1)

        return {
            'coords': coords,
            'sequence': sequence,
            'sequence_onehot': sequence_onehot,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'length': item['length']
        }

class DiffusionTrainer:
    """Trainer for the diffusion-based protein design model"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc='Training')

        for batch in progress_bar:
            # Move batch to device
            coords = batch['coords'].to(self.device)
            sequence_onehot = batch['sequence_onehot'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_attr = batch['edge_attr'].to(self.device)

            # Sample random timesteps
            batch_size = coords.shape[0]
            timesteps = self.model.noise_scheduler.sample_timesteps(batch_size, self.device)

            # Forward pass through diffusion model
            predicted_noise, target_noise = self.model(
                sequence_onehot, edge_index, edge_attr, sequence_onehot, timesteps
            )

            # Compute MSE loss
            loss = nn.MSELoss()(predicted_noise, target_noise)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': loss.item()})

        self.scheduler.step()
        return total_loss / num_batches

    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                coords = batch['coords'].to(self.device)
                sequence_onehot = batch['sequence_onehot'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                edge_attr = batch['edge_attr'].to(self.device)

                batch_size = coords.shape[0]
                timesteps = self.model.noise_scheduler.sample_timesteps(batch_size, self.device)

                predicted_noise, target_noise = self.model(
                    sequence_onehot, edge_index, edge_attr, sequence_onehot, timesteps
                )

                loss = nn.MSELoss()(predicted_noise, target_noise)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_dataloader, val_dataloader, num_epochs=100, save_path='model_checkpoint.pt'):
        """Full training loop"""
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_dataloader)

            # Validation
            val_loss = self.validate(val_dataloader)

            self.logger.info(f'Epoch {epoch+1}/{num_epochs}: '
                           f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, save_path)
                self.logger.info(f'Saved new best model with validation loss: {val_loss:.4f}')

    def generate_sequences(self, structures, num_samples=8):
        """Generate protein sequences for given structures"""
        self.model.eval()

        generated_sequences = []

        with torch.no_grad():
            for structure in structures:
                coords = structure['coords'].unsqueeze(0).to(self.device)
                edge_index = structure['edge_index'].unsqueeze(0).to(self.device)
                edge_attr = structure['edge_attr'].unsqueeze(0).to(self.device)

                # Generate multiple sequences per structure
                for _ in range(num_samples):
                    sequence_probs = self.model.sample(coords, edge_index, edge_attr)
                    sequence = torch.argmax(sequence_probs, dim=-1)
                    generated_sequences.append(sequence.cpu().numpy())

        return generated_sequences

# Validation utilities
class ProteinValidator:
    """Validation using AlphaFold2 and other metrics"""

    def __init__(self):
        self.aa_alphabet = 'ARNDCQEGHILKMFPSTWYV'

    def sequence_recovery(self, predicted_sequences, target_sequences):
        """Calculate sequence recovery percentage"""
        correct = 0
        total = 0

        for pred, target in zip(predicted_sequences, target_sequences):
            correct += (pred == target).sum()
            total += len(target)

        return (correct / total) * 100

    def validate_with_alphafold(self, sequences, structures):
        """Validate generated sequences using AlphaFold2 prediction"""
        # This would interface with AlphaFold2 or ESMFold
        # For demonstration, return synthetic validation metrics

        validation_results = []

        for seq, struct in zip(sequences, structures):
            # Synthetic validation metrics
            confidence_score = np.random.uniform(0.7, 0.95)  # pLDDT-like score
            rmsd = np.random.uniform(1.0, 3.0)  # RMSD to target structure
            tm_score = np.random.uniform(0.8, 0.98)  # TM-score

            validation_results.append({
                'confidence_score': confidence_score,
                'rmsd': rmsd,
                'tm_score': tm_score,
                'success': confidence_score > 0.8 and rmsd < 2.0 and tm_score > 0.85
            })

        return validation_results

    def assess_designability(self, sequences):
        """Assess protein designability metrics"""
        results = []

        for seq in sequences:
            # Calculate various designability metrics
            hydrophobic_ratio = sum(1 for aa in seq if aa in [0, 9, 10, 12, 13, 17, 18, 19]) / len(seq)
            charged_ratio = sum(1 for aa in seq if aa in [1, 3, 6, 11]) / len(seq)
            small_ratio = sum(1 for aa in seq if aa in [0, 2, 4, 7, 15, 16]) / len(seq)

            results.append({
                'length': len(seq),
                'hydrophobic_ratio': hydrophobic_ratio,
                'charged_ratio': charged_ratio,
                'small_ratio': small_ratio,
                'designable': 0.3 < hydrophobic_ratio < 0.6 and charged_ratio < 0.3
            })

        return results
