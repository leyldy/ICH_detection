import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


# Function to update predictions so that if patient is classified as not ICH, the rest of ICH type predictions are changed to 0
# Only so that accuracies actually make sense interpretability-wise
# long-term solution: need to change the actual loss function so elements are correlated
def update_predictions(pred):
    # NOTE: NEED TO INCORPORATE THIS IN LOSS FUNCTION?
    pred_mask = np.expand_dims(pred[:, 0], axis=1) # N X 1 vector
    pred_new = pred * pred_mask # broadcast multiply 
    
    return pred_new

# Calculate metrics by different class
def calculate_ind_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    num_pred_occur = np.sum(pred, axis=0)
    
    # Calculate precision, recall, fscore, etc (NEED TO CHANGE)
    precision, recall, fscore, num_true_occur = precision_recall_fscore_support(y_true=target, y_pred=pred, average=None, zero_division=0)
    num_pred_occur = np.sum(pred, axis=0)
    
    # Updating predictions (see above)
    pred = update_predictions(pred)
    
    # Calculate accuracy of ICH or not with new predictions
    acc_ich = np.mean(target[:, 0] == pred[:, 0])
    
    # Among actual patients who had ICH, what the accuracy was for each ICH type
    ich_index = (target[:, 0] == 1)
    acc_type = np.mean(pred[ich_index, 1:] == target[ich_index, 1:], axis=0)
    
    accuracy = [acc_ich] + acc_type.tolist()
    
    ind_metrics_dict = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'fscore': fscore, 
                        'num_true_occur': num_true_occur, 'num_pred_occur': num_pred_occur}
    
    return ind_metrics_dict

def calculate_avg_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    
#     # Updating predictions (see above)
#     pred = update_predictions(pred)
    
    # Calculating individual metrics first
    ind_metrics_dict = calculate_ind_metrics(pred, target)
    
    acc_ich = ind_metrics_dict['accuracy'][0]
    acc_type_total = np.mean(ind_metrics_dict['accuracy'][1:]) # Just macro average across all the different types
    
#     # Among actual patients who had ICH, accuracy in predicting type (calculate per patient, then avg across patient)
#     ich_index = (target[:, 0] == 1)
#     num_diff_types = target.shape[1] - 1
#     acc_type_by_patient = np.mean(pred[ich_index, 1:] == target[ich_index, 1:], axis=1)/num_diff_types
#     acc_type_total = np.mean(acc_type_by_patient, axis=0)
    
    # Precision, recall, etc (NEED TO CHANGE)
    precision, recall, fscore, num_true_occur = precision_recall_fscore_support(y_true=target, y_pred=pred, average='micro', zero_division=0)
    avg_metrics_dict = {'accuracy_ich': acc_ich, 'accuracy_ich_type': acc_type_total, 
                        'precision': precision, 'recall': recall, 'fscore': fscore, 
                        'num_true_occur': np.sum(num_true_occur), 'num_pred_occur': np.sum(pred)}
    return avg_metrics_dict
    

def log_metrics(scores_np, targets_np, loss_value, iteration, writer, curr_mode='train'):
    avg_dict = calculate_avg_metrics(scores_np, targets_np)
    ind_dict = calculate_ind_metrics(scores_np, targets_np)
    
    # Save average metrics to tensorboard
    writer.add_scalar("Loss/"+curr_mode, loss_value, iteration)
    writer.add_scalar("ICH detection Accuracy/"+curr_mode, avg_dict['accuracy_ich'], iteration)
    writer.add_scalar("Overall ICH type classification Accuracy/"+curr_mode, avg_dict['accuracy_ich_type'], iteration)

    # Save individual element accuracies to tensorboard (including overall=ICH_type_0)
    ind_acc_dict = {('ICH_type_' + str(i)):elem for i, elem in enumerate(ind_dict['accuracy'])}
    writer.add_scalars("Individual ICH type accuracies/"+curr_mode, ind_acc_dict, iteration)

    writer.add_scalars("Micro Precision-Recall-F1/"+curr_mode, 
                       {'Precision': avg_dict['precision'], 'Recall': avg_dict['recall'], 'F1': avg_dict['fscore']}, 
                       iteration)

def test_baseline_model_shape():
    x = torch.zeros((32, 1, 12, 512, 512))  
    model = baseline_3DCNN(in_num_ch=1)
    scores = model(x)
    print(scores.size()) 