
import sys
import os
sys.path.append('..')

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from torchvision import transforms
import webdataset as wds

from utils.model_utils import *
from time import sleep

dtype = torch.float

def train(model, optimizer, criterion, loader_train, loader_val, log_dir, device, epochs=5, val_every=1):
    """
    Train a model.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    total_iter = 0
    val_loss_dict = {}
    train_loss_dict = {}
    
    # Writer for tensorboard
    writer = SummaryWriter(log_dir)

    for e in range(1, epochs+1):
        
        with tqdm(loader_train, unit="batch") as tepoch:
            ep_train_losses = []
            ep_val_losses = []

#             print("************EPOCH: {:2d} ***************".format(e))
            for t, (x, y, z) in enumerate(tepoch):
                model.train()  # put model to training mode
                
                tepoch.set_description("Epoch %d" % e)
                
                t += 1 # To start from iteration 1 (1-indexing)
                
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=dtype)

                scores = model(x)
                loss = criterion(scores, y)
                curr_batchloss = loss.item()
                ep_train_losses.append(curr_batchloss)

                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                optimizer.zero_grad()

                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                optimizer.step()

                # Log metrics
                scores_np = scores.detach().cpu().numpy()
                targets_np = y.detach().cpu().numpy()
                log_metrics(scores_np, targets_np, curr_batchloss, total_iter+t, writer, curr_mode='train')

                # Run validation step every val_every iteration
                if t % val_every == 0:
#                     print('Current epoch %d, epoch iteration %d, train loss = %.4f' % (e, t, curr_batchloss))
                    val_loss = validate_model(loader_val, model, criterion, total_iter+t, writer, device)
                    ep_val_losses.append(val_loss)

                    # Also save checkpoint of model 
                    ckpt_path = os.path.join(log_dir, 'Checkpoints', 'ep_%d_iter_%d_ckpt.pt' % (e, t))
                    torch.save({
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, ckpt_path)

                    print()
                    writer.flush()
                    
                tepoch.set_postfix(loss=curr_batchloss)
                sleep(0.1)

        train_loss_dict['epoch_'+str(e)] = ep_train_losses
        val_loss_dict['epoch_'+str(e)] = ep_val_losses
        total_iter += t+1

    # Close the summarywriter for tensorboard
    writer.close()
    
    # Return train and validation loss
    return (train_loss_dict, val_loss_dict)

def validate_model(loader, model, criterion, iteration, writer, device):
#     print('Checking accuracy on validation set')

    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        targets_np, scores_np, val_losses, batch_sizes = [], [], [], []
        
        # Run validation on validation batches
        
        for x, y, z in loader:
            print(x.size())
            x = transforms.Resize(size=(256, 256))(x)

            # Add code to unsqueeze because we only have 1 channel (axis=1) of this 3d image (N, C, H, W)
            # NOTE: DON'T NEED TO ADD EXTRA DIMENSION HERE BECAUSE LOADED IN WITH C DIM ALREADY
            # x = x.unsqueeze(axis=1)

            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            scores = model(x)
            val_loss = criterion(scores, y)
            val_losses.append(val_loss.item())
            batch_sizes.append(x.shape[0])

            scores_np.extend(scores.cpu().numpy())
            targets_np.extend(y.cpu().numpy())


    # Calculate metrics after running full validation set
    scores_np, targets_np = np.array(scores_np), np.array(targets_np)

    # Log Metrics
    val_loss = np.average(val_losses, weights=batch_sizes)
    log_metrics(scores_np, targets_np, val_loss, iteration, writer, curr_mode="validation")

    # Print results
    print('Total iteration %d, validation loss = %.4f' % (iteration, val_loss))
#         print("Loss: {:.4f}, Micro accuracy: {:.3f}, Micro precision: {:.3f}, Micro recall: {:.3f}, Micro F1: {:.3f}"
#               .format(np.mean(val_losses), avg_dict['accuracy'], avg_dict['precision'], avg_dict['recall'], avg_dict['fscore'])
#              )

    # Return validation loss
    return val_loss

