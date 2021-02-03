# Problem 2 - Claynet

In this problem, I design a SoftMax loss network with 4 conv layers of 64 3x3 filters each followed by 2 full connection layers to be connected with SoftMax layer, and report validation accuracy.  
![alt text](https://github.com/alailink/pytorch_tripletloss/blob/main/images/problem2_net.png)  
Very similar to problem 1, except in this problem a simpler, unique architecture was used:  
```python
ClayNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    (7): ReLU(inplace)
    (8): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    (9): ReLU(inplace)
    (10): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5)
    (1): Linear(in_features=2304, out_features=4096, bias=True)
    (2): ReLU(inplace)
    (3): Dropout(p=0.5)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
  )
)
```  

This was trained over 140 epochs with a 2GB GPU. The increase in validation accuracy peaked out at about the same time. Instead of compression with PCA/LDA, we calculated the accuracy directly from the neural model with a test set, and that turned out to be 87.13%. This is just about on par with the pre-trained AlexNet with no modifications/transfer learning.  
![claynet](https://github.com/alailink/pytorch_tripletloss/blob/main/images/claynet.png)  
| Method  | Accuracy (Validation Set) |
| ------------- | ------------- |
| Pre-trained AlexNet Baseline -> LDA  | 87.0%  |
| Transfer Learning Alexnet -> LDA  | 92.1%  |
| ClayNet trained (unique network) | 87.13% |
