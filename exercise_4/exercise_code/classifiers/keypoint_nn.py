import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=7, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0, dilation=1, groups=1, bias=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fcl1 = nn.Linear(256, 128)
        self.fcl2 = nn.Linear(128, 100)
        self.fcl3 = nn.Linear(100, 30)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.15)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.15)
        self.dropout5 = nn.Dropout(p=0.3)
        self.dropout6 = nn.Dropout(p=0.35)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        x = self.conv1(x)           # 2 - Convolution2d1
        x = F.elu(x)                # 3 - Activation1
        x = self.pool(x)            # 4 - Maxpooling2d1     
        x = self.dropout1(x)        # 5 - Dropout1      
        x = self.conv2(x)           # 6 - Convolution2d2
        x = F.elu(x)                # 7 - Activation2
        x = self.pool(x)            # 8 - Maxpooling2d2
        x = self.dropout2(x)        # 9 - Dropout2
        x = self.conv3(x)           # 10 - Convolution2d3
        x = F.elu(x)                # 11 - Activation3
        x = self.pool(x)            # 12 - Maxpooling2d3
        x = self.dropout3(x)        # 13 - Dropout3
        x = self.conv4(x)           # 14 - Convolution2d4
        x = F.elu(x)                # 15 - Activation4
        x = self.pool(x)            # 16 - Maxpooling2d4
        x = self.dropout4(x)        # 17 - Dropout4
        x = x.view(x.size(0), -1)   # 18 - Flatten1
        x = self.fcl1(x)            # 19 - Dense1
        x = F.elu(x)                # 20 - Activation5
        x = self.dropout5(x)        # 21 - Dropout5
        x = self.fcl2(x)            # 22 - Dense2
        x = F.elu(x)                # 23 - Activation6
        x = self.dropout6(x)        # 24 - Dropout6
        x = self.fcl3(x)            # 25 - Dense3 
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
