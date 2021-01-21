"""
    PyTorch training code for TFeat shallow convolutional patch descriptor:
    http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    The code reproduces *exactly* it's lua anf TF version:
    https://github.com/vbalnt/tfeat
    2017 Edgar Riba
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os 
from PIL import Image
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.nn.modules.loss import TripletMarginLoss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch TFeat Example')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--n_triplets', type=int, default=20000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train', type=bool, default=True, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


class TripletData(datasets.ImageFolder):
    """From the MNIST Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, *arg, **kw):
        super(TripletData, self).__init__(*arg, **kw)

        print('Generating triplets ...')
        self.n_triplets = args.n_triplets
        self.train_triplets = self.generate_triplets(self.targets)
        self.train = args.train
        self.train_data, _ = zip(*self.imgs)
    def generate_triplets(self, labels):
        triplets = []
        labels = torch.tensor(labels).numpy()

        ulabels = np.unique(labels)
        matches, no_matches = dict(), dict()
        for x in ulabels:
            matches[x] = np.where(labels == x)[0]
            no_matches[x] = np.where(labels != x)[0]

        candidates = np.random.randint(0, len(labels), size=self.n_triplets)
        candidates = labels[candidates]
        for x in candidates:
            idx_a, idx_p = np.random.choice(matches[x], 2, replace=False)
            
            idx_n = np.random.choice(no_matches[x], 1)[0]
            triplets.append([idx_a, idx_p, idx_n])
        return np.array(triplets)

    def __getitem__(self, index):
        if self.train:
            t = self.train_triplets[index]
            a, p, n = io.imread(self.train_data[t[0]]), io.imread(self.train_data[t[1]]),\
                      io.imread(self.train_data[t[2]])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_a = Image.fromarray(a, mode='RGB')
        img_p = Image.fromarray(p, mode='RGB')
        img_n = Image.fromarray(n, mode='RGB')

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)

        return img_a, img_p, img_n

    def __len__(self):
        if self.train:
            return self.train_triplets.shape[0]



class TNet(nn.Module):
    """TFeat model definition
    """
    def __init__(self, num_classes=15):
        super(TNet, self).__init__()
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


#class TripletMarginLoss(nn.Module):
#    """Triplet loss function.
#    Based on: http://docs.chainer.org/en/stable/_modules/chainer/functions/loss/triplet.html
#    """
#    def __init__(self, margin):
#        super(TripletMarginLoss, self).__init__()
#        self.margin = margin
#
#    def forward(self, anchor, positive, negative):
#        pos = (anchor - positive)**2
#        nega = (anchor - negative)**2
#        negb = (positive - negative)**2
#        dista = torch.sum(pos - nega,dim=1) + self.margin
#        distb = torch.sum(pos - negb,dim=1) + self.margin
#        dist = torch.max(dista,distb)
##        dist = torch.sum((anchor - positive)**2 - (anchor - negative)**2, dim=1) + self.margin
#        dist_hinge = torch.clamp(dist, min=0.0)  # maximum between 'dist' and 0.0
#        loss = torch.mean(dist_hinge)
#        return loss


def triplet_loss(input1, input2, input3, margin=0.6):
    """Interface to call TripletMarginLoss
    """
    return TripletMarginLoss(margin,swap=True)(input1, input2, input3)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
os.chdir(r'C:\Users\ck7w2\Documents\Computer Vision\Homework 4')
data_dir = 'NWPU_split'

train_data = TripletData(os.path.join(data_dir, 'train'),
                 transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                 ]))
train_loader = torch.utils.data.DataLoader(train_data,
        batch_size=args.batch_size, shuffle=True, **kwargs)

val_data = datasets.ImageFolder(os.path.join(data_dir, 'val'),
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                ]))
val_loader = torch.utils.data.DataLoader(val_data,
        batch_size=128, shuffle=True, **kwargs)

model = TNet()
if args.cuda:
    model.cuda()
model.load_state_dict(torch.load("triplet.pth"))

lr_gridsearch = 10**(-6)
best_lr = lr_gridsearch
optimizer = optim.Adam(model.parameters(), lr = lr_gridsearch)
plot_train = []
plot_val = []
plot_loss = []
best_acc = 86.7
val_acc = 0
dud= 0 
marg_gridsearch = 0.6
best_margin = marg_gridsearch


def train(epoch, marg_gridsearch):
    model.train()
#    running_correct = 0.0;
    for batch_idx, (data_a, data_p, data_n) in enumerate(train_loader):
        if args.cuda:
            data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
#        data_a, data_p, data_n = Variable(data_a), Variable(data_p), Variable(data_n)
        optimizer.zero_grad()
        with torch.enable_grad():
            out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)
    #            pred = torch.argmax(out_a, 1)
            loss = triplet_loss(out_a, out_p, out_n,margin=marg_gridsearch)
            plot_loss.append(loss.data.item())
            loss.backward()
            optimizer.step()
#            running_correct += torch.sum(pred == labels.data)
    
#    train_acc = running_correct.double() / args.n_triplets * 100    
#    plot_train.append(train_acc)
    print('Train Epoch: {}\tLoss: {:.6f}'.format(
        epoch, loss.data.item()))

def validate(epoch,best_acc):
    running_correct = 0.0;
    with torch.no_grad():
        model.eval()
        for inputs, labels in val_loader:
            optimizer.zero_grad() # zero the parameter gradients
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            output = model(inputs)
            pred = torch.argmax(output, 1) # output.topk(1) *1 = top1
    
            running_correct += torch.sum(pred == labels.data)
    val_acc = running_correct.double() / len(val_data) * 100
    plot_val.append(val_acc.cpu().numpy())
    if val_acc > best_acc:
        torch.save(model.state_dict(), "triplet.pth")
    
    print('Val Acc: {:4f}'.format(val_acc))
    return val_acc
    

if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch, marg_gridsearch)
        val_acc = validate(epoch,best_acc)
        if val_acc>best_acc:
            best_acc = val_acc
    plt.figure(1)
    ax1 = plt.plot(plot_val)
    plt.title("Validation Accuracy per Epoch")
    plt.ylim(60, 100)
##some code for a gridsearch
#            dud = 0
#            best_lr = lr_gridsearch
#            print("Best lr is:\t{}".format(best_lr))
#        else:
#            dud +=1
            
#        if dud == 5:
#            lr_gridsearch *= 0.1
#            optimizer = optim.Adam(model.parameters(), lr = lr_gridsearch)
#            model.load_state_dict(torch.load("triplet.pth"))
#            
#        if val_acc < 10:
#            lr_gridsearch *= 0.1
#            optimizer = optim.Adam(model.parameters(), lr = lr_gridsearch)
#            model.load_state_dict(torch.load("triplet.pth"))
#        
#        
    
    
#running_correct = 0.0;
#with torch.no_grad():
#    model.eval()
#    for inputs, labels in dataloaders['test']:
#        inputs = inputs.to(device)
#        labels = labels.to(device)
#        optimizer.zero_grad() # zero the parameter gradients
#        output = model(inputs)
#        _, pred = torch.max(output, 1) # output.topk(1) *1 = top1
#
#        running_correct += torch.sum(pred == labels.data)
#test_acc = running_correct.double() / dataset_sizes['test'] * 100
#print('Test Acc: {:4f}'.format(test_acc))

#graphs
#import pandas as pd
#plots = pd.read_excel("plots.xlsx").values
#loss = plots[:,0]
#val = plots[:,1]
#plt.plot(loss)
#plt.title("Loss over epochs")
#plt.figure(2)
#plt.plot(val)
#plt.title("Validation, Starting with trained parameters")
        