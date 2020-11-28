import PIL
import json
import torch
import argparse
import numpy as np
from math import ceil
from PIL import Image
from train import check_gpu
from torchvision import models, transforms

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('image',type=str,help='Point to impage file for prediction.')
    parser.add_argument('checkpoint',type=str,help='Point to checkpoint file as str.')
    parser.add_argument('--top_k',type=int,help='Choose top K matches as int.', default=5)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="cpu", action="store", dest="gpu")

    args = parser.parse_args()
    
    return args


def load_checkpoint(checkpoint_path):
    
    state = torch.load(checkpoint_path)
    
    model = state['model']
    model.class_to_idx = state['class_to_idx']
    model.classifier = state['classifier']
    model.load_state_dict(state['state_dict'])
    
    return model

def process_image(image):
    img = Image.open(image)
    
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transformed_img = transform(img)
    return transformed_img
    
def predict(image, model, device, cat_to_name, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.to(device)
    model.eval();

    # numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(image,
                                                  axis=0)).type(torch.FloatTensor).to('cuda:0')

    # prob
    log_probs = model.forward(torch_image)

    # linear scale
    lin_probs = torch.exp(log_probs)

    # top 5
    top_probs, top_labels = lin_probs.topk(top_k)
    
    ##
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers


def print_probability(probs, flowers):
    #Converts two lists into a dictionary to print on screen
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, likelyhood: {}%".format(j[0], round(j[1]*100)))


def main():
    
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    image_tensor = process_image(args.image)
    device = check_gpu(gpu_arg=args.gpu);
    top_probs, top_labels, top_flowers = predict(image_tensor, model, device, cat_to_name, args.top_k)
    print_probability(top_probs, top_flowers)

if __name__ == '__main__': 
    main()
    