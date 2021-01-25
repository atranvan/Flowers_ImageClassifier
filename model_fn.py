import torch
from torch import nn, optim
from torchvision import models
import numpy as np

def mkmodel(arch = 'vgg16', hidden_layers = [1024, 256], device = 'cpu', learning_rate = 0.001):
    
    models_dict = {'vgg13': models.vgg13, 'vgg16': models.vgg16, 'vgg19': models.vgg19, 'densenet121': models.densenet121, 'densenet169' : models.densenet169}
    if arch not in models_dict.keys():
        print('not a supported model')
        return None
    else:    
        model = models_dict[arch](pretrained = True)
    if arch.startswith('vgg'):
        input_features = 25088
    elif arch.startswith('densenet'):
        input_features = 1024 
        
    # Freeze weights of feature extractor.
    for param in model.parameters():
        param.requires_grad = False    
        
    # define classifier
    clf = nn.Sequential(nn.Linear(input_features, hidden_layers[0]),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden_layers[0], hidden_layers[1]),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_layers[1], 102),
                                 nn.LogSoftmax(dim=1))
                        

    
    model.classifier = clf
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate )
    model.to(device)
    return model, criterion, optimizer, input_features

def train(loaders, epochs, model, criterion, optimizer, device):
    running_loss = 0
    steps = 0
    print_every = 40

    for epoch in range(epochs):
        for inputs, labels in loaders['train_loaders']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            train_loss = criterion(logps, labels)
            train_loss.backward()
            optimizer.step()
            running_loss += train_loss.item()
            if steps % print_every == 0:
                model.eval()
                val_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in loaders['valid_loaders']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        val_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {val_loss/len(loaders['valid_loaders']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(loaders['valid_loaders']):.3f}")
                running_loss = 0
                model.train()
    return model


def predict(image, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    if device == 'cpu':    
        torch_im = image.type(torch.FloatTensor).unsqueeze_(0)
    else:
        torch_im = image.type(torch.cuda.FloatTensor).unsqueeze_(0)
    model.to(device)
    
    torch_im.to(device)
    
    model.eval()
    model.to(device)
    with torch.no_grad():
        logps = model.forward(torch_im)
    ps = torch.exp(logps)
        
    top_p, top_class = ps.topk(topk, dim=1)
    top_p = top_p.tolist()[0]
    top_class = top_class.tolist()[0]
#     top_p, top_class = np.array(top_p.to(device)[0]), np.array(top_class.to(device)[0])
    d_idxtoclass = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [d_idxtoclass[i] for i in top_class]
    return top_p, top_classes

def save_checkpoint(model, optimizer, datasets, input_features, learning_rate, epochs, filename):
    model.class_to_idx = datasets['train_data'].class_to_idx

    checkpoint = {
        'model': model,
        'input_size': input_features,
        'output_size': 102,
        'learning_rate': learning_rate,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(checkpoint, filename)
    
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    model.learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']    
    return model



