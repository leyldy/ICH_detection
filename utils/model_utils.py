import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    
    # Calculate metrics by each individual label
    precision, recall, fscore, num_true_occur = precision_recall_fscore_support(y_true=target, y_pred=pred, average=None, zero_division="warn")
    num_pred_occur = np.sum(pred, axis=0)
    accuracy = (target == pred).mean(axis=0)
    ind_metrics_dict = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'fscore': fscore, 
                        'num_true_occur': num_true_occur, 'num_pred_occur': num_pred_occur}
    
    # Calculate and average overall (micro method)
    precision, recall, fscore, num_true_occur = precision_recall_fscore_support(y_true=target, y_pred=pred, average='micro', zero_division="warn")
    accuracy = (target==pred).mean()
    avg_micro_metrics_dict = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'fscore': fscore, 
                              'num_true_occur': np.sum(num_true_occur), 'num_pred_occur': np.sum(num_pred_occur)}
    return ind_metrics_dict, avg_micro_metrics_dict

def test_baseline_model_shape():
    x = torch.zeros((32, 1, 12, 512, 512))  
    model = baseline_3DCNN(in_num_ch=1)
    scores = model(x)
    print(scores.size()) 