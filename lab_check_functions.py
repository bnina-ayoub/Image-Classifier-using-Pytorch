from torchvision import datasets, transforms, models
import torch
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import json
def check_train_command_line_arguments(args):
    if args is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line args
        print("Command Line Arguments:\n data_directory =", args.data_directory, 
              "\n save_directory =", args.save_dir, "\n arch =", args.arch, 
              "\n learning_rate =", args.learning_rate, "\n hidden_units =", args.hidden_units, 
              "\n epochs =", args.epochs, "\n print_every =", args.print_every, "\n use_gpu =",
              args.gpu)
        
def check_predict_command_line_arguments(args):
    if args is None:
        print("* Doesn't Check the Command Line Arguments because 'get_predict_args' hasn't been defined.")
    else:
        print("Command Line Arguments:\n image_path =", args.image_path,
              "\n checkpoint =", args.checkpoint, "\n top_k =", args.top_k,
              "\n category_names =", args.category_names ,"\n use_gpu =", args.gpu)

def train_transform():

    return transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Resize(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                              ])
def val_test_transform():
    return transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                              ])
# TODO: Load the datasets with ImageFolder
def load_data(data_directory):
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform())
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform())
    val_dataset = datasets.ImageFolder(valid_dir, transform=val_test_transform())
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loaders = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loaders = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    val_loaders = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

    return train_dataset.class_to_idx, train_loaders, test_loaders, val_loaders


def load_model_for_training(arch, lr, hidden_units, device):
    if arch == 'vgg19' : 
        model = models.vgg19(pretrained = True)
    else:
        model = models.alexnet(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    layers = [model.classifier[i] for i in range(2)]
    layers.append(nn.Dropout(p=0.05))
    layers.append(nn.Linear(4096,hidden_units))
    layers.append(nn.LogSoftmax(dim=1))
    
    classifier = nn.Sequential(*layers)
    
    model.classifier = classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
    print(model)
    model.to(device)                             
    return model, optimizer, criterion

def save_model_checkpoint(model, optimizer, arch, hidden_units, epoch, loss, class_index):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx': class_index,
            'loss' : loss,
            'arch' : arch,
            'hidden_units' : hidden_units
    }, f"Model_Checkpoint_{epoch}.pth")
    print("Model Saved!")

def load_checkpoint(checkpoint_model, device):
    if device=="gpu":
        map_location=lambda device, loc: device.cuda()
    else:
        map_location='cpu'
    
    checkpoint = torch.load(f=checkpoint_model, map_location=map_location)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    
    print(arch, hidden_units)
    if arch == 'vgg19' : 
        model = models.vgg19(pretrained = True)
    else:
        model = models.alexnet(pretrained = True)
        
        
    layers = [model.classifier[i] for i in range(2)]
    layers.append(nn.Dropout(p=0.05))
    layers.append(nn.Linear(4096,hidden_units))
    layers.append(nn.LogSoftmax(dim=1))
    
    classifier = nn.Sequential(*layers)
    
    model.classifier = classifier
        
    model.to(device)
        # redefines checkpoint       
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(model)
    return model


def process_image(image):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model,
       returns an Numpy array
    '''
    with Image.open(image).convert('RGB') as M:
        trsfm = val_test_transform()
        np_image = trsfm(M).numpy()
        '''
        M = M.resize((256, 256))
       
        crop_top = (M.height - 224) / 2
        crop_left = (M.width - 224) / 2
        crop_bot = (M.height + 224) / 2
        crop_right = (M.width + 224) / 2
        M.crop((crop_left, crop_top, crop_right, crop_bot))
        np_image = np.array(M) / 255.0
       
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std
       
        np_image = np_image.transpose((2, 0, 1))
        #np_image = np.expand_dims(np_image, axis=0)
        print(np_image.shape)
        '''
        return np_image
    

def read_cat_name(file_json):
    with open(file_json, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def class_mapping(class_to_idx, labels):
        names = []
        class_to_idx = read_cat_name(class_to_idx)
        for class_idx in labels:
            names.append(class_to_idx[str(class_idx)])
        return names