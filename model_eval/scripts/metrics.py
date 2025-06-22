"""
This module contains:
    1. utility functions for model evaluation
    2. path for saving evaluation results
"""

SAVING_BASE_DIR = "/jhcnas5/Generalist/shebd/luoyi"
# /{Method}_{Scale}_{Dataset}/
# checkpoints: best.pth, last.pth
# json: best_results.json, last_results.json (prediction and ground_truth of all instances)


import torch
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, MultilabelAccuracy
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score, MultilabelF1Score

def format_output_binary(result_json, label_set):
    """Format output for binary classification"""
    preds = []
    targets = []
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(label_set)}
    
    for item in result_json:
        # Handle predictions
        pred = item['prediction']
        if isinstance(pred, str):
            pred_idx = label_to_idx[pred]
        else:
            pred_idx = int(pred)
        
        # Handle ground truth
        gt = item['ground_truth']
        if isinstance(gt, str):
            gt_idx = label_to_idx[gt]
        else:
            gt_idx = int(gt)
        
        preds.append(pred_idx)
        targets.append(gt_idx)
    
    return torch.tensor(preds), torch.tensor(targets)

def format_multiclass_output(result_json, label_set):
    """Format output for multiclass classification"""
    preds = []
    targets = []
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(label_set)}
    
    for item in result_json:
        # Handle predictions
        pred = item['prediction']
        if isinstance(pred, str):
            # If prediction is a string label, convert to index
            pred_idx = label_to_idx[pred]
        else:
            # If prediction is already an index, use it directly
            pred_idx = int(pred)
        
        # Handle ground truth
        gt = item['ground_truth']
        if isinstance(gt, str):
            # If ground truth is a string label, convert to index
            gt_idx = label_to_idx[gt]
        else:
            # If ground truth is already an index, use it directly
            gt_idx = int(gt)
        
        preds.append(pred_idx)
        targets.append(gt_idx)
    
    return torch.tensor(preds), torch.tensor(targets)

def format_multilabel_output(result_json, label_set):
    """Format output for multilabel classification"""
    preds = []
    targets = []
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(label_set)}
    
    for item in result_json:
        # For multilabel, predictions and ground_truth should be lists of labels
        pred_labels = item['prediction'] if isinstance(item['prediction'], list) else [item['prediction']]
        target_labels = item['ground_truth'] if isinstance(item['ground_truth'], list) else [item['ground_truth']]
        
        # Create binary vectors
        pred_vector = [0] * len(label_set)
        target_vector = [0] * len(label_set)
        
        # Set 1 for present labels
        for label in pred_labels:
            if label in label_to_idx:
                pred_vector[label_to_idx[label]] = 1
                
        for label in target_labels:
            if label in label_to_idx:
                target_vector[label_to_idx[label]] = 1
        
        preds.append(pred_vector)
        targets.append(target_vector)
    
    return torch.tensor(preds, dtype=torch.float), torch.tensor(targets, dtype=torch.float)

def eval_binary(result_json, label_set):
    
    preds, targets = format_output_binary(result_json, label_set)
    acc_metric = BinaryAccuracy()
    acc = acc_metric(preds, targets)
    f1_metric = BinaryF1Score()
    f1 = f1_metric(preds, targets)

    print(f"Accuracy: {acc}")
    print(f"F1: {f1}")

    return acc, f1, preds, targets

def eval_multiclass(result_json, label_set):

    preds, targets = format_multiclass_output(result_json, label_set)
    acc_metric = MulticlassAccuracy(num_classes=len(label_set))
    acc = acc_metric(preds, targets)
    macro_f1_metric = MulticlassF1Score(num_classes=len(label_set), average='macro')
    macro_f1 = macro_f1_metric(preds, targets)
    micro_f1_metric = MulticlassF1Score(num_classes=len(label_set), average='micro')
    micro_f1 = micro_f1_metric(preds, targets)

    print(f"Accuracy: {acc}")
    print(f"Macro F1: {macro_f1}")
    print(f"Micro F1: {micro_f1}")

    return acc, macro_f1, micro_f1, preds, targets

def eval_multilabel(result_json, label_set):

    preds, targets = format_multilabel_output(result_json, label_set)
    acc_metric = MultilabelAccuracy(num_labels=len(label_set))
    acc = acc_metric(preds, targets)
    macro_f1_metric = MultilabelF1Score(num_labels=len(label_set), average='macro')
    macro_f1 = macro_f1_metric(preds, targets)
    micro_f1_metric = MultilabelF1Score(num_labels=len(label_set), average='micro')
    micro_f1 = micro_f1_metric(preds, targets)

    print(f"Accuracy: {acc}")
    print(f"Macro F1: {macro_f1}")
    print(f"Micro F1: {micro_f1}")

    return acc, macro_f1, micro_f1, preds, targets

def calculate_metrics(predictions, targets, dataset_processor, args):
    """Helper function to calculate metrics"""
    if dataset_processor.dataset_meta['D' + args.dataset_name]['tasktype'] == 'binary':
        return eval_binary(
            result_json=[{'prediction': pred, 'ground_truth': target} for pred, target in zip(predictions, targets)],
            label_set=dataset_processor.dataset_meta['D' + args.dataset_name]['label_set']
        )
    elif dataset_processor.dataset_meta['D' + args.dataset_name]['tasktype'] == 'multiclass':
        return eval_multiclass(
            result_json=[{'prediction': pred, 'ground_truth': target} for pred, target in zip(predictions, targets)],
            label_set=dataset_processor.dataset_meta['D' + args.dataset_name]['label_set']
        )
    elif dataset_processor.dataset_meta['D' + args.dataset_name]['tasktype'] == 'multilabel':
        return eval_multilabel(
            result_json=[{'prediction': pred, 'ground_truth': target} for pred, target in zip(predictions, targets)],
            label_set=dataset_processor.dataset_meta['D' + args.dataset_name]['label_set']
        )
    else:
        raise ValueError("Unsupported task-type in dataset metadata")