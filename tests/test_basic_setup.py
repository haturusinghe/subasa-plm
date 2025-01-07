import unittest
import torch
import sys
import os
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.helpers import get_device
from main import parse_args
from src.config.config import ModelConfig

class TestBasicSetup(unittest.TestCase):
    def test_device_detection(self):
        device = get_device()
        self.assertIsInstance(device, torch.device)
        
    def test_argument_parsing(self):
        # Mock command line arguments
        sys.argv = ['main.py', 
                   '--pretrained_model', 'xlm-roberta-base',
                   '--batch_size', '16',
                   '--epochs', '5',
                   '--lr', '0.00005',
                   '--intermediate', 'mrp']
        
        args = parse_args()
        self.assertEqual(args.pretrained_model, 'xlm-roberta-base')
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.lr, 0.00005)
        self.assertEqual(args.intermediate, 'mrp')

    def test_model_config(self):
        config = ModelConfig(
            model_name='xlm-roberta-base',
            max_length=128,
            batch_size=16,
            learning_rate=0.00005,
            epochs=5,
            output_dir='output'
        )
        
        self.assertEqual(config.model_name, 'xlm-roberta-base')
        self.assertEqual(config.max_length, 128)
        self.assertEqual(config.batch_size, 16)

if __name__ == '__main__':
    unittest.main()
