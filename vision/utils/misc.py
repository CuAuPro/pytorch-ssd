import time
import torch
import numpy as np

def str2bool(s):
    return s.lower() in ('true', '1')


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval
        

def save_checkpoint(epoch, net_state_dict, optimizer_state_dict, best_score, checkpoint_path, model_path):
    torch.save({
        'epoch': epoch,
        'model': net_state_dict,
        'optimizer': optimizer_state_dict,
        'best_score': best_score
    }, checkpoint_path)
    torch.save(net_state_dict, model_path)
        
        
def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False


def store_labels(path, labels):
    with open(path, "w") as f:
        f.write("\n".join(labels))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save(self.path)
        self.val_loss_min = val_loss