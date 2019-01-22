"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        # get pre-trained model
        self.vgg = models.vgg11_bn(pretrained=True).features
        # append 2 convs
        self.segmentation = nn.Sequential(
            nn.Conv2d(512,256,5),
            nn.ReLU(),
            nn.Conv2d(256,128,1)      
        )
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        old_shape = x.shape
        x = self.vgg(x)
        x = self.segmentation(x)
        # upsample
        x = F.interpolate(x, old_shape[2:], mode='bilinear', align_corners=True)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
