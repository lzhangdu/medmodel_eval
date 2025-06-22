import datasetProcesser
import datasetLoader
import metrics
import utilities
import wandb
import os
import sys
import torch
import numpy as np
import json
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torchvision import transforms

# Add MedViTV2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../MedViTV2'))

from MedViT import MedViT_small, MedViT_base, MedViT_large

def get_model(model_size, num_classes, device):
    """Create model and move to specified device"""
    if model_size == 'small':
        return MedViT_small(pretrained=True, num_classes=num_classes).to(device)
    elif model_size == 'base':
        return MedViT_base(pretrained=True, num_classes=num_classes).to(device)
    elif model_size == 'large':
        return MedViT_large(pretrained=True, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model size: {model_size}")

def train(args):
    dataset_processor = datasetProcesser.DatasetProcesser([args.dataset_name])
    if not dataset_processor.check_dataset_exists_locally(args.dataset_name):
        dataset_processor.copy_dataset_to_local(args.dataset_name)

    saving_base_dir = metrics.SAVING_BASE_DIR
    model_save_path = os.path.join(saving_base_dir, f"{args.model_name}_{args.model_size}_{args.dataset_name}")
    os.makedirs(model_save_path, exist_ok=True)

    # Initialize wandb
    if args.wandb_name:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                "model_name": args.model_name,
                "model_size": args.model_size,
                "dataset_name": args.dataset_name,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "device": args.device
            }
        )
    else:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "model_name": args.model_name,
                "model_size": args.model_size,
                "dataset_name": args.dataset_name,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "device": args.device
            }
        )
        
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Data transforms - using MedViT standard transforms
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    }

    # Load dataset
    dataset_loader = datasetLoader.DatasetLoader(
        dataset_path=os.path.join(dataset_processor.local_dataset_base_path, args.dataset_name),
        label_set=dataset_processor.dataset_meta['D' + args.dataset_name]['label_set'],
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    train_loader = dataset_loader.get_train_loader(data_transform['train'])
    test_loader = dataset_loader.get_test_loader(data_transform['test'])

    # Initialize model
    model = get_model(args.model_size, len(dataset_processor.dataset_meta['D' + args.dataset_name]['label_set']), device)
    model.to(device)

    # Initialze optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_epoch = 0
    best_predictions = None
    best_ground_truth = None

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for batch_idx, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            train_correct = (np.array(train_predictions) == np.array(train_targets)).sum()
            train_total = len(train_targets)
            train_bar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })

        # Evaluation phase on test set
        model.eval()
        test_loss = 0.0
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Test]')
            for batch_idx, (images, labels) in enumerate(test_bar):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                
                test_predictions.extend(predicted.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())
                
                # Update progress bar
                test_correct = (np.array(test_predictions) == np.array(test_targets)).sum()
                test_total = len(test_targets)
                test_bar.set_postfix({
                    'loss': f'{test_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*test_correct/test_total:.2f}%'
                })

        # Update learning rate
        scheduler.step()

        # Calculate metrics - handle different return formats based on task type
        task_type = dataset_processor.dataset_meta['D' + args.dataset_name]['tasktype']

        if task_type == 'binary':
            train_acc, train_f1, _, _ = metrics.calculate_metrics(train_predictions, train_targets, dataset_processor, args)
            test_acc, test_f1, _, _ = metrics.calculate_metrics(test_predictions, test_targets, dataset_processor, args)
            
            # Convert to Python scalars if they're tensors
            train_metrics = {'accuracy': float(train_acc), 'f1_score': float(train_f1)}
            test_metrics = {'accuracy': float(test_acc), 'f1_score': float(test_f1)}
            
            # Log metrics to wandb
            wandb_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'train_acc': float(train_acc),
                'train_f1': float(train_f1),  
                'test_loss': test_loss / len(test_loader),
                'test_acc': float(test_acc),  
                'test_f1': float(test_f1),    
                'learning_rate': scheduler.get_last_lr()[0]
            }
            
        else:  # multiclass or multilabel
            train_acc, train_macro_f1, train_micro_f1, _, _ = metrics.calculate_metrics(train_predictions, train_targets, dataset_processor, args)
            test_acc, test_macro_f1, test_micro_f1, _, _ = metrics.calculate_metrics(test_predictions, test_targets, dataset_processor, args)
            
            # Convert to Python scalars if they're tensors (same as binary case)
            train_metrics = {'accuracy': float(train_acc), 'macro_f1': float(train_macro_f1), 'micro_f1': float(train_micro_f1)}
            test_metrics = {'accuracy': float(test_acc), 'macro_f1': float(test_macro_f1), 'micro_f1': float(test_micro_f1)}
            
            # Log metrics to wandb
            wandb_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'train_acc': float(train_acc),           
                'train_macro_f1': float(train_macro_f1), 
                'train_micro_f1': float(train_micro_f1), 
                'test_loss': test_loss / len(test_loader),
                'test_acc': float(test_acc),             
                'test_macro_f1': float(test_macro_f1),   
                'test_micro_f1': float(test_micro_f1),   
                'learning_rate': scheduler.get_last_lr()[0]
            }

        wandb.log(wandb_metrics)

        # Save best model based on test accuracy
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            best_epoch = epoch + 1
            best_predictions = test_predictions.copy()
            best_ground_truth = test_targets.copy()

            # Save best model checkpoint
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_acc,
                'args': vars(args),
            }, os.path.join(model_save_path, 'best_model_checkpoint.pth'))

            # Save best results JSON
            best_results = {
                'epoch': best_epoch,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'class_names': dataset_processor.dataset_meta['D' + args.dataset_name]['label_set'],
                'class_to_idx': dataset_loader.get_class_to_idx(),
                'predictions': []
            }
            
            # Add individual predictions
            for i, (pred, gt) in enumerate(zip(best_predictions, best_ground_truth)):
                best_results['predictions'].append({
                    'sample_id': i,
                    'ground_truth': int(gt),
                    'ground_truth_class': dataset_processor.dataset_meta['D' + args.dataset_name]['label_set'][gt],
                    'prediction': int(pred),
                    'prediction_class': dataset_processor.dataset_meta['D' + args.dataset_name]['label_set'][pred],
                    'correct': bool(pred == gt)
                })
            
            with open(os.path.join(model_save_path, 'best_result.json'), 'w') as f:
                json.dump(best_results, f, indent=4)

if __name__ == "__main__":
    args = utilities.parse_generic_training_args()
    train(args)