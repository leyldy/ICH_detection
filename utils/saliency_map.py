import torch
import random
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

# from .image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
# from scipy.ndimage.filters import gaussian_filter1d

def compute_saliency_maps(X, y, model, device):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, D, H, W)
    - y: Labels for X; LongTensor of shape (N, 6) # 6 ICH Types
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model = model.float().to(device)
    model.eval()

    # Make input tensor same dtype as model
    X = X.float().to(device)

    # Saliency: (ICH_types, N, D, H, W)
    sal_shape = (y.shape[1],) + X.shape 
    saliency_all = torch.zeros(sal_shape).to(device)


    # Compute saliency map for each target vector dimension:
    for dim in range(y.shape[1]):
        curr_y = y[:, dim] # Select current ICH type
        rel_ind = (curr_y == 1) # Index to only select images which have presence in this ICH type
        curr_y = curr_y[rel_ind] 

        if curr_y.nelement() != 0: # Only if batch contains at least one example
            # Select only relevant X values
            curr_x = X[rel_ind, :, :, :]
            # Require grad
            curr_x.requires_grad_()
            curr_x.retain_grad()
            # Forward pass:
            curr_scores = model(curr_x) # N X 6 (6-dim output vector)

            # Loss and saliency
            loss = curr_scores[:, dim].sum() # Only sum loss in curr ICH type
            loss.backward()
            saliency_all[dim, rel_ind, :, :, :] += torch.abs(curr_x.grad.detach())
            
            del curr_x
            del curr_scores
            del loss
    
    del X
    del model
        
    return saliency_all # (ICH_types, N, D, H, W)


def rank_saliency_slices(saliency_df):
    """ 
        Return index of highest amount of gradient exposure from saliency
        for each patient

        Input: 
          saliency_df - (ICH_types, N, D, H, W)
    """

    sal_sum = saliency_df.sum(axis=(3,4)) # (ICH_types, N, D)
    sal_slices = sal_sum.argsort(axis=2, descending=True) # (ICH_types, N, D)

    return sal_slices



def plot_saliency_maps(X, saliency_df, ich_num, patient_id, d_range, num_rows, num_cols):
    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency_np = saliency_df[ich_num, patient_id, d_range, :, :].cpu().numpy() # (D,H,W)
    D = saliency_np.shape[0]
    curr_slice = 0
    assert num_rows * num_cols == 2*D, print("WRONG DIMS FOR PLOTTING")
    fig = plt.figure(figsize=(14,9))
    for d in d_range: 
        ax = fig.add_subplot(num_rows, num_cols, (curr_slice//num_cols) * num_cols + curr_slice + 1)
        ax.imshow(X[patient_id, d, :, :])
        ax.axis('off')
        ax.title.set_text("Slice: "+str(d))
        del ax
        
        ax = fig.add_subplot(num_rows, num_cols, (curr_slice//num_cols) * num_cols + curr_slice + 1 + num_cols)
        ax.imshow(saliency_np[curr_slice, :, :], cmap=plt.cm.hot)
        ax.axis('off')
        # plt.gcf().set_size_inches(12, 5)
        del ax

        curr_slice += 1
    return fig