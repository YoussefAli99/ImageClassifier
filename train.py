  
import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args


def transformer(train_dir, valid_dir, test_dir):
    data_transforms = { 
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
    }


    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }
    
    return image_datasets['train'], image_datasets['valid'], image_datasets['test']

def data_loader(train_data, valid_data, test_data):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size =64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)
    return train_loader, valid_loader, test_loader





def check_gpu(gpu_arg):
    return torch.device('cuda:0') if (gpu_arg and torch.cuda.is_available()) else torch.device('cpu')


def loader_model(arch):
    archs = {
        'resnet18': models.resnet18(pretrained=True),
        'alexnet': models.alexnet(pretrained=True),
        'vgg16': models.vgg16(pretrained=True),
        'squeezenet': models.squeezenet1_0(pretrained=True),
        'densenet': models.densenet161(pretrained=True),
        'inception': models.inception_v3(pretrained=True)
    }
    try:
        model = archs[arch]
    except KeyError: 
        print(f'Model Not found: {arch}')
        model = archs['vgg16']
        
    for param in model.parameters():
        param.requires_grad = False
    return model


def initial_classifier(model, hidden_units):
    from collections import OrderedDict

    classifier = nn.Sequential(OrderedDict([
                ('inputs', nn.Linear(25088, 120)), 
                ('relu1', nn.ReLU()),
                ('dropout',nn.Dropout(0.5)), 
                ('hidden_layer1', nn.Linear(120, 90)), 
                ('relu2',nn.ReLU()),
                ('hidden_layer2',nn.Linear(90,70)), 
                ('relu3',nn.ReLU()),
                ('hidden_layer3',nn.Linear(70,102)), 
                ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    return classifier



def validation(model, testloader, criterion, device='cuda:0'):
    cor,total = 0,0
    with torch.no_grad():
        model.eval()
        for data in train_loader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            cor += (predicted == labels).sum().item()

    print('acc = {}'.format(100 * cor/total))




def network_trainer(model, trainloader, testloader, device, criterion, optimizer, epochs, print_every=10):
    model.train()
    print("Training Mode Anabled ✅")
    for epoch in range(1, epochs+1):
        print(f"Epoch: {epoch}/{epochs}")

        train_loss, train_acc = 0.0, 0.0
        valid_loss, valid_acc = 0.0, 0.0

        for i,(inputs, labels) in enumerate(trainloader):

            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            #Accuracy
            _,pred = torch.max(outputs.data, 1)
            c_counts = pred.eq(labels.data.view_as(pred))

            #Compute the mean
            acc_ = torch.mean(c_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc_.item() * inputs.size(0)
            if not (i%print_every):
                print("--- : {:03d}, loss: {:.4f}, accuracy: {:.4f}".format(i, loss.item(), acc_.item()))

    return model


def validate_model(model, loader, criterion, device):
    with torch.no_grad():
        model.eval()
        valid_loss, valid_acc = 0,0
        for k, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # outputs
            outputs = model(inputs)
            # loss
            loss = criterion(outputs, labels)

            # Total loss
            valid_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            valid_acc += acc.item() * inputs.size(0)

            print("--- : {:03d}, validation loss: {:.4f}, acc: {:.4f}".format(k, loss.item(), acc.item()))
    

def initial_checkpoint(model, structure, train_data, criterion, optimizer, lr, epochs, save_dir='./checkpoint.pth'):
    print("Trying to checkpoint")
    model.class_to_idx = train_data.class_to_idx
    torch.save({'model': structure,
                'classifier': model.classifier,
                'hidden_layer': 120,
                 'droupout': 0.5,
                 'epochs': epochs,
                 'criterion': criterion,
                 'learning_rate': lr,
                 'state_dict': model.state_dict(),
                 'class_to_idx': model.class_to_idx,
                 'optimizer_dict': optimizer.state_dict()},
                 'checkpoint.pth')
    print("CHECKPOINT COMPLETED ✅")

def main():
    args = arg_parser()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data, valid_data, test_data = transformer(train_dir, valid_dir, test_dir)
    
    trainloader, validloader, testloader = data_loader(train_data, valid_data, test_data)
    
    model = loader_model(args.arch)
    
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    
    device = check_gpu(gpu_arg=args.gpu);
    model.to(device);
    
    lr = (args.learning_rate if isinstance(args.learning_rate, (float, int)) else 0.001)
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    trained_model = network_trainer(model, trainloader, validloader, device, criterion, optimizer, args.epochs)
    print("---------------TRAIN COMPLETED--------------------")
    print("---------------VALIDATION STARTED--------------------")
    validate_model(trained_model, testloader, criterion, device)
    print("---------------VALIDATION COMPLETED--------------------")
    initial_checkpoint(trained_model, model, train_data, criterion, optimizer, lr, args.epochs, args.save_dir)
    
if __name__ == '__main__':
    main()