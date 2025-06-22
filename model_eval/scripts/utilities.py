import argparse

def parse_generic_training_args():
    """Generic argument parser"""
    parser = argparse.ArgumentParser(description='Train deep learning model')
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, required=True,
                      help='Name of the dataset to use')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, required=True,
                      choices=['medmamba', 'medvit', 'medvitv2'],
                      help='Model to use')
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'],
                      help='Model size to use')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=48,
                      help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=6e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda:1',
                      help='Device to use for training')
    
    # Logging arguments
    parser.add_argument('--wandb_project', type=str, default='medical-image-classification',
                      help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default='royalty-hong-kong-university-of-science-and-technology',
                      help='Weights & Biases entity name')
    parser.add_argument('--wandb_name', type=str, default=None,
                      help='Weights & Biases run name')
    
    return parser.parse_args()