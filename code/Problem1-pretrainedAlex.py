# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:12:12 2019

@author: ck7w2
"""


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm

os.chdir(r'C:\Users\ck7w2\Documents\Computer Vision\Homework 4')
normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))

augmentation = transforms.RandomApply([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10)], p=.8)

training_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    augmentation,
    transforms.ToTensor(),
    normalize])

valid_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    normalize])

dataset = torchvision.datasets.ImageFolder('NWPU',transform=training_transform)
training, val, test = torch.utils.data.random_split(dataset, (500*15,100*15,100*15)) #not guaranteed stratified (equal) but it's a large sample set so it should be fine
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=16,
                                          shuffle=True,)

# Get a batch of training data
tmpiter = iter(data_loader)
images, labels = tmpiter.next()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
alex = models.alexnet(pretrained=True)
alex = alex.to(device)
class fclayer(nn.Module):
        def __init__(self):
            super(fclayer,self).__init__()
            self.features = nn.Sequential(
                    *list(alex.features.children()))
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifer = nn.Sequential(
                    *list(alex.classifier.children())[:-1])
            
        def forward(self,x):
            x = self.features(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifer(x)
            return x



fclayer_pull = fclayer()
fclayer_pull = fclayer_pull.to(device)
                
def evaluate(dataset):
    output = np.empty((len(dataset),4096))
    labels = np.empty(len(dataset))
    loader = DataLoader(dataset, batch_size=64,shuffle=True)
    index = 0;
    with torch.no_grad():
        fclayer_pull.eval()
        for idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            new_data = fclayer_pull(data).cpu()
            new_data = new_data.numpy()
            output[index*64:index*64+len(new_data),:] = new_data
            labels[index*64:index*64+len(new_data)] = target.cpu()
            index = index+1
    return output, labels

train_output, train_labels = evaluate(training)
val_output, val_labels = evaluate(val)
test_output, test_labels = evaluate(test)
        
#np.savetxt('train_output', train_output, delimiter=',')
#np.savetxt('train_labels', train_labels, delimiter=',')
#np.savetxt('val_output', val_output, delimiter=',')
#np.savetxt('val_labels', val_labels, delimiter=',')
#np.savetxt('test_output', test_output, delimiter=',')
#np.savetxt('test_labels', test_labels, delimiter=',')

pca = PCA(n_components=128)
pca.fit(train_output)
training_reduced = pca.transform(train_output)
test_reduced = pca.transform(test_output)
val_reduced = pca.transform(val_output)

lda = LDA()
lda.fit(training_reduced, train_labels)
training_lda = lda.transform(training_reduced)
test_lda = lda.transform(test_reduced)
val_lda = lda.transform(val_reduced)

classifier = svm.SVC(gamma='scale')
classifier.fit(training_lda, train_labels)
test_accuracy = np.mean(classifier.predict(test_lda)==test_labels)
val_accuracy = np.mean(classifier.predict(val_lda)==val_labels)
