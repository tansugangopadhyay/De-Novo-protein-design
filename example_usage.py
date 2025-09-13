
import torch
from structured_transformer import StructuredTransformer, DiffusionModel
from training import ProteinDataset, DiffusionTrainer, ProteinValidator
from torch.utils.data import DataLoader

def main():
    """Example usage of the protein design system"""

    # Initialize model
    base_model = StructuredTransformer(
        node_dim=128,
        num_heads=8, 
        num_layers=6,
        vocab_size=20
    )

    diffusion_model = DiffusionModel(base_model)

    # Setup data
    dataset = ProteinDataset('path/to/pdb/data')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize trainer
    trainer = DiffusionTrainer(diffusion_model)

    # Train model
    trainer.train(train_loader, val_loader, num_epochs=100)

    # Generate sequences for new structures
    test_structures = [val_dataset[i] for i in range(10)]
    generated_sequences = trainer.generate_sequences(test_structures)

    # Validate results
    validator = ProteinValidator()
    validation_results = validator.validate_with_alphafold(
        generated_sequences, test_structures
    )

    # Print results
    success_rate = sum(1 for r in validation_results if r['success']) / len(validation_results)
    print(f"Design success rate: {success_rate:.2%}")

    avg_confidence = sum(r['confidence_score'] for r in validation_results) / len(validation_results)
    print(f"Average confidence score: {avg_confidence:.3f}")

if __name__ == "__main__":
    main()
