import torch
from torchmetrics.functional import precision, recall, f1_score, auroc

# Accuracy
def accuracy(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    acc = pred.eq(target.view_as(pred)).sum().item()
    return acc

# Recall & Precision
def get_recall(output, target, type=0, n_classes=0):
    pred = output.argmax(dim=1, keepdim=True).squeeze()
    if type == 0: # micro (average of total sample)
        value = recall(pred, target, average='micro', num_classes=n_classes, task='multiclass')
    elif type == 1: # macro (average of average for each class)
        value = recall(pred, target, average='macro', num_classes=n_classes, task='multiclass')
    else:
        raise ValueError(f'Invalid recall precision type input')
    
    return value.item()

def get_precision(output, target, type=0, n_classes=0):
    pred = output.argmax(dim=1, keepdim=True).squeeze()
    if type == 0: # micro (average of total sample)
        value = precision(pred, target, average='micro', num_classes=n_classes, task='multiclass')
    elif type == 1: # macro (average of average for each class)
        value = precision(pred, target, average='macro', num_classes=n_classes, task='multiclass')
    else:
        raise ValueError(f'Invalid recall precision type input')
    
    return value.item()

def get_recall_k(output, target, type=0, n_classes=0, k=10):
    pred = output.argmax(dim=1, keepdim=True).squeeze()
    if type == 0: # micro (average of total sample)
        value = recall(pred, target, average='micro',  num_classes=n_classes, top_k=k, task='multiclass')
    elif type == 1: # macro (average of average for each class)
        value = recall(pred, target, average='macro', num_classes=n_classes, top_k=k, task='multiclass')
    else:
        raise ValueError(f'Invalid recall precision type input')
    
    return value.item()

def get_precision_k(output, target, type=0, n_classes=0, k=10):
    pred = output.argmax(dim=1, keepdim=True).squeeze()
    if type == 0: # micro (average of total sample)
        value = precision(pred, target, average='micro', num_classes=n_classes, top_k=k, task='multiclass')
    elif type == 1: # macro (average of average for each class)
        value = precision(pred, target, average='macro', num_classes=n_classes, top_k=k, task='multiclass')
    else:
        raise ValueError(f'Invalid recall precision type input')
    
    return value.item()

# F1_score
def get_f1_score(output, target, type=0, n_classes=0):
    pred = output.argmax(dim=1, keepdim=True).squeeze()
    if type == 0: # micro (calculate globally, across all samples and classes)
        value = f1_score(pred, target, average='micro', num_classes=n_classes, task='multiclass')
    elif type == 1: # macro (calculate for each class, and average the metrics across classes)
        value = f1_score(pred, target, average='macro', num_classes=n_classes, task='multiclass')
    else:
        raise ValueError(f'Invalid f1 score type input')
    
    return value.item()

# AUROC is area of the ROC curve (for FPR & TPR; 'micro' is impossible)
def get_auroc(output, target, n_classes=0):
    int_target = target.type(torch.long)
    value = auroc(output, int_target, average='macro', num_classes=n_classes, task='multiclass')
    return value.item()
