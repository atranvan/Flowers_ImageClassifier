# import libraries
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import util_fn as hfn
import model_fn as modfn
import os

def main():
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Basic usage: python train.py data_directory    
# --save_dir directory to save checkpoints
# --arch "vgg13" choose architecture
# --learning_rate 0.01 --hidden_units 512 --epochs 20 hyperparameters
# --gpu use gpu

    
    
    # get arguments from command line
    args = hfn.get_train_args()
    device = ("cuda" if ((args.gpu) & (torch.cuda.is_available())) else "cpu")

    # load data
    data_transforms, image_datasets, dataloaders, cat_to_name = hfn.load(args.data_dir)
    
    # define model
    model, criterion, optimizer, input_features = modfn.mkmodel(arch = args.arch, hidden_layers = args.hidden_units, device = device, learning_rate = args.learning_rate)
    
    # train model
    model = modfn.train(dataloaders, args.epochs, model, criterion, optimizer, device)
    
    # save checkpoint
    filename = os.path.join(args.save_dir, 'checkpoint.pth')
    modfn.save_checkpoint(model, optimizer, image_datasets, input_features, args.learning_rate, args.epochs, filename)

if __name__ == '__main__':
    main()