import torch, torchvision
from torch import nn, optim
from torchvision import datasets, transforms, models

import util_fn as hfn
import model_fn as modfn

def main():
# Predict flower name from an image along with the probability of that name
# Basic usage: python predict.py flowers/test/43/image_02365.jpg checkpoint.pth
# --top_k Top K most likely classes
# --category_names use mapping of category names (json file)
# --gpu use GPU for inference

    # parse arguments
    args = hfn.get_pred_args()

    device = ("cuda" if ((args.gpu) and (torch.cuda.is_available())) else "cpu")
    
    # load checkpoint
    model =  modfn.load_checkpoint(args.checkpoint)
    
    # load and process image
    image = hfn.process_image(args.image)

    # prediction
    probabilities, classes = modfn.predict(image, model, device, topk=args.top_k)
    
    # print results
    print(f'The top {args.top_k} probabilities are: {list(probabilities)}')
    if args.category_names:
        import json
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[item] for item in classes]
        print(f'The top {args.top_k} classes are: {list(class_names)}')
    else:
        print(f'The top {args.top_k} classes are: {list(classes)}')
        
        

if __name__ == '__main__':
    main()