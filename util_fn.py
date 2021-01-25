import torch, torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import argparse
import json

def get_train_args():
    """
    Parses command line arguments
    Output : parse_args structure 
    """
    parser = argparse.ArgumentParser(description="arguments for neural network training")

    # data directory
    parser.add_argument('data_dir', type = str, action='store', help = 'Data directory')
    
    # directory to save checkpoint
    parser.add_argument('--save_dir', default = '.', type = str, action='store', help = 'Define directory where to save checkpoints')
    
    # network architecture
    parser.add_argument('--arch', type = str, default='vgg16', action='store',  help='Choose network architecture, default : vgg16')
    
    # model hyperparameters
    parser.add_argument('--learning_rate', default=0.001, type=float, action='store', help='learning rate, default : 0.001' )
    parser.add_argument('--hidden_units', nargs='+', default=[1024, 256], type=int, help='hidden layer sizes for 3 layer model, default : 1024, 512')
    parser.add_argument('--epochs', default=10, type=int, action='store', help='number of training epochs, default : 10')
    
    # use GPU
    parser.add_argument('--gpu', action='store_true', default=False, help='use GPU processing')
    
    args = parser.parse_args()
    return args

def get_pred_args():
    """
    Parses command line arguments
    Output : parse_args structure 
    """
    parser = argparse.ArgumentParser(description="arguments for classifier predictions")
    
    parser.add_argument('image', action='store')
    parser.add_argument('checkpoint', action='store')
    
    # return top classes
    parser.add_argument('--top_k', action='store', type=int, default=5)
    
    # category mapping
    parser.add_argument('--category_names', action='store', default='cat_to_name.json')
    
    # use GPU
    parser.add_argument('--gpu', action='store_true', default=False, help='use GPU processing')
    
    args = parser.parse_args()
    return args


def load(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    data_transforms = {'training' : transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
                       'validation' : transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
                       'testing' : transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
                      }
    
    image_datasets = {'train_data' : datasets.ImageFolder(train_dir, transform = data_transforms['training']),
                     'valid_data': datasets.ImageFolder(valid_dir, transform = data_transforms['validation']),
                     'test_data': datasets.ImageFolder(test_dir, transform = data_transforms['testing'])
                     }  
    
    dataloaders =  {'train_loaders' : torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
                     'valid_loaders': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64),
                     'test_loaders': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=64)
                     }
    
# label mapping
    with open("cat_to_name.json", "r") as f:
        cat_to_name = json.load(f)
    
    return data_transforms, image_datasets, dataloaders, cat_to_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    # resize
    w, h = pil_image.size
    
    if w>h:
        h = 256
        w = int(w*256/h)
    else:
        w = 256
        h = int(h*256/w)
    pil_image = pil_image.resize((w,h))
    
    # crop
    cropsize = 224
    l = (w-cropsize)/2
    t = (h-cropsize)/2
    r = l + cropsize
    b = t + cropsize
    pil_image = pil_image.crop((l, t, r, b))
    
    # color channels and normalize
    np_image = np.array(pil_image)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image)