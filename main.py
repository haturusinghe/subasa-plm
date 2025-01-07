import argparse
from src.config.config import ModelConfig
from src.utils.logging_utils import setup_logging
import logging
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate XLM-RoBERTa model')
    parser.add_argument('--model_name', type=str, default="xlm-roberta-base",
                       help='Name or path of the model')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for model checkpoints')
    return parser.parse_args()

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting training pipeline")

    # Parse arguments
    args = parse_args()

    # Create configuration
    config = ModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        output_dir=args.output_dir
    )



    try:
        # Train the model
        logger.info("Starting training")


        # Evaluate the model
        logger.info("Starting evaluation")

        # Log results
        logger.info("Classification Report:")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
