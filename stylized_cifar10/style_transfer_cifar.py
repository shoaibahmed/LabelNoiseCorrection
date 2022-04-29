"""A PyTorch version of AdaIN Style Transfer."""
import argparse

import numpy as np
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn as nn

from . import AdaptiveInstanceNormalization as Adain
from . import decoder
from . import vgg_normalised


class StyleTransfer:
    def __init__(self):
        self.vgg = self.load_vgg_model()
        self.decoder = self.load_decoder_model()
        self.adain = Adain.AdaptiveInstanceNormalization()

    def style_transfer(self, style_img, content_img):
        """Style transfer between content image and style image."""
        style_img = self.normalize_img(style_img)
        content_img = self.normalize_img(content_img)
        
        style_feature = self.vgg(style_img)  # torch.Size([1, 512, 16, 16])
        content_feature = self.vgg(content_img)  # torch.Size([1, 512, 16, 16])
        input = torch.cat((content_feature, style_feature), 0)
        
        target_feature = self.adain(input)
        alpha = 0.75
        target_feature = alpha * target_feature + (1 - alpha) * content_feature
        out = self.decoder(target_feature)
        out = out.data.squeeze(0).permute(1, 2, 0).numpy()
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        return out
    
    def normalize_img(self, image):
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert image.shape[-1] == 3
        image = image.astype(np.float32) / 255.0  # Normalize
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        image = image.unsqueeze(0)  # Change from 3D to 4D
        image = image.float()
        assert len(image.shape) == 4 and image.shape[1] == 3
        return image

    def load_vgg_model(self):
        """Load the VGG model."""
        this_vgg = vgg_normalised.vgg_normalised
        this_vgg.load_state_dict(torch.load('stylized_cifar10/models/vgg_normalised.pth'))
        """
        This is to ensure that the vgg is the same as the model used in PyTorch lua as below:
        vgg = torch.load(opt.vgg)
        for i=53,32,-1 do
            vgg:remove(i)
        end
        This actually removes 22 layers from the VGG model.
        """
        this_vgg = nn.Sequential(*list(this_vgg)[:-22])
        this_vgg.eval()
        return this_vgg

    def load_decoder_model(self):
        """Load the decoder model which is converted from the Torch lua model using
        git@github.com:clcarwin/convert_torch_to_pytorch.git.

        :return: The decoder model as described in the paper.
        """
        this_decoder = decoder.decoder
        this_decoder.load_state_dict(torch.load('stylized_cifar10/models/decoder.pth'))
        this_decoder.eval()
        return this_decoder
