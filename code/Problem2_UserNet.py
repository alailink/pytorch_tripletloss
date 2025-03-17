# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:56:06 2019

@author: ck7w2
"""


import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import time
import os
import copy
from torch.optim import lr_scheduler
import torch.nn.functional as F

####################
### Data Loading ###
####################

os.chdir(r'C:\Users\ck7w2\Documents\Computer Vision\Homework 4')
if __name__ == "__main__":
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
         ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }
    
    data_dir = 'NWPU_split'
    image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'),  data_transforms['train'])
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val','test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val','test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes
    
    
    
    ###########################
    ### Net  Initialization ###
    ###########################
    
    class UserNet(nn.Module):
    
        def __init__(self, num_classes=45):
            super(UserNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=2), #convolution layer 1. Input channels = 3 (RGB)
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=2), #convolution layer 2
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=2), #convlution layer 3
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=2), #convolution layer 4
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(64 * 6 * 6, 4096), #first fully connected layer
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096), #second fully connected layer
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 15) #second fully connected layer
                #Softmax is contained within the CrossEntropyLoss function
            )
    
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), 64 * 6 * 6)
            x = F.softmax(self.classifier(x),dim=1)
            return x
    
    lr = 0.001
    weight_decay = 0.00001
    momentum = 0.9
    epochs = 50
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device = torch.device("cpu")
    model = UserNet(num_classes=15)
    model = model.to(device)
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum, weight_decay)
    validation_scores = np.empty(epochs)
    model.load_state_dict(torch.load("Usernet_model.pth"))
    #######################
    ### Training Setup #### 
    #######################
    
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                running_loss = 0.0
                running_corrects = 0
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
    
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad() # zero the parameter gradients
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
#                        print("loss: ", loss.item())
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if phase == 'val':
                    validation_scores[epoch] = epoch_acc
    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) # Decay LR by a factor of 0.1 every 7 epochs
    model = train_model(model, ce_loss, optimizer, exp_lr_scheduler,num_epochs=epochs)
    torch.save(model.state_dict(), "Usernet_model.pth")
    
    
    #######################
    ### Some graphics ##### 
    #######################
    def visualize_model(model, num_images=16):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()
    
        with torch.no_grad():
                inputs, labels = next(iter(dataloaders['test']))
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
    
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(4, num_images/4, images_so_far)
                    ax.axis('off')
                    ax.set_title('{}'.format(class_names[preds[j]]))
                    img = inputs.cpu().numpy()[j,:,:,:].transpose((1,2,0))
                    ax = plt.imshow(img)
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return

        model.train(mode=was_training)
    
    visualize_model(model)
    
    
    ###########################
    ### Test set Accuracy ##### 
    ###########################
    running_correct = 0.0;
    with torch.no_grad():
        model.eval()
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # zero the parameter gradients
            output = model(inputs)
            _, pred = torch.max(output, 1) # output.topk(1) *1 = top1
    
            running_correct += torch.sum(pred == labels.data)
    test_acc = running_correct.double() / dataset_sizes['test'] * 100
    print('Test Acc: {:4f}'.format(test_acc))
        
    
    
    






