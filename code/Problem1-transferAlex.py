# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:57:32 2019

@author: ck7w2
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

plt.ion()   # interactive mode
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
    ]),
}

data_dir = 'NWPU_split'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
class_names = image_datasets['train'].classes

## Lets visualize few image
#def imshow(inp, title=None):
#    """Imshow for Tensor."""
#    inp = inp.numpy().transpose((1, 2, 0))
#    mean = np.array([0.485, 0.456, 0.406])
#    std = np.array([0.229, 0.224, 0.225])
#    inp = std * inp + mean
#    inp = np.clip(inp, 0, 1)
#    plt.imshow(inp)
#    if title is not None:
#        plt.title(title)
#    plt.pause(0.001)  # pause a bit so that plots are updated
#
#
## Get a batch of training data
#inputs, classes = next(iter(dataloaders['train']))
#
## Make a grid from batch
#out = torchvision.utils.make_grid(inputs)
#
#imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    print("loss: ", loss.item())
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

model_ft = models.alexnet(pretrained=True)
model_ft.classifier[6] = nn.Linear(4096,15)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=2)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=2)

torch.save(model_ft.state_dict(), "transfer_model.pth")

# Prediction on test data
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('{}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

visualize_model(model_ft)

######################################################
#### PULL FC LAYER FOR TRAINING / VAL / TESTING ######
######################################################
class fclayer(nn.Module):
        def __init__(self):
            super(fclayer,self).__init__()
            self.features = nn.Sequential(
                    *list(model_ft.features.children()))
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifer = nn.Sequential(
                    *list(model_ft.classifier.children())[:-1])
            
        def forward(self,x):
            x = self.features(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifer(x)
            return x
        

fclayer_pull = fclayer()
fclayer_pull = fclayer_pull.to(device)
                
def evaluate(split):
    output = np.empty((int(dataset_sizes[split]),4096))
    labels = np.empty(int(dataset_sizes[split]))
    loader = dataloaders[split]
    index = 0;
    with torch.no_grad():
        fclayer_pull.eval()
        for idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            new_data = fclayer_pull(data).cpu()
            new_data = new_data.numpy()
            output[index*32:index*32+len(new_data),:] = new_data
            labels[index*32:index*32+len(new_data)] = target.cpu()
            index = index+1
    return output, labels

train_output, train_labels = evaluate('train')
print(train_labels)
val_output, val_labels = evaluate('val')
print(val_labels)
test_output, test_labels = evaluate('test')
print(test_labels)


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm

pca = PCA(n_components=128)
pca.fit(train_output)
training_reduced = pca.transform(train_output)
test_reduced = pca.transform(test_output)
val_reduced = pca.transform(val_output)

lda = LDA(solver='eigen')
lda.fit(training_reduced, train_labels)
training_lda = lda.transform(training_reduced)
test_lda = lda.transform(test_reduced)
val_lda = lda.transform(val_reduced)

classifier = svm.SVC(gamma='scale')
classifier.fit(training_lda, train_labels)
test_accuracy = np.mean(classifier.predict(test_lda)==test_labels)
val_accuracy = np.mean(classifier.predict(val_lda)==val_labels)

np.savetxt('pca_train', training_reduced, delimiter=',')
np.savetxt('lda_train', training_lda, delimiter=',')
np.savetxt('labels_train', train_labels, delimiter=',')

np.savetxt('pca_val', val_reduced, delimiter=',')
np.savetxt('lda_val', val_lda, delimiter=',')
np.savetxt('labels_val', val_labels, delimiter=',')

np.savetxt('pca_test', test_reduced, delimiter=',')
np.savetxt('lda_test', test_lda, delimiter=',')
np.savetxt('labels_test', test_labels, delimiter=',')
