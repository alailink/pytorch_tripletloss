# Problem 1a - Pretrained AlexNet

The rationale here is to compute a baseline accuracy of how well a pretrained CNN can perform on the dataset. However, these images are not included in AlexNet's outputs, so we have decided to use it's 4096-feature fully connected layer to generate our own simple classifier.  
Here I am sending the images from our dataset through a pretrained alexnet, and pulling the values from the 4096-feature **fully connected** layer. These values are then put into a lower-dimensional embedding through a two-step principal component (PCA) then linear discriminant (LDA) reduction. A support vector machine (SVM) is then used for the final classification, with validation set used for reporting.  
We are using the NWPU-RESISC45 dataset (Northwest Polytechnic Institute: REmote Sensing Image Scene Classification)1. We are using the first fifteen classes of environmental scenes, which includes 700 images per class for a total of 10500 image (our folder structure can be found in the zip file, but without image files to save memory.) All coding is done in python using pytorch and the scikit-learn packages.  
[For code please click here.](https://github.com/alailink/pytorch_tripletloss/blob/main/code/Problem1-pretrainedAlex.py) As an overview, here is the architecture of AlexNet:  
```python
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace)
    (3): Dropout(p=0.5)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)  
```
**The final baseline accuracy is 87%.**  

# Problem 1b - transfer learning from AlexNet
