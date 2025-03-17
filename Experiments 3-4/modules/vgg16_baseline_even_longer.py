"""                                                                                                          
VGG 16 model without Predify                                                                                 
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16_bn


class VGG16Baseline(nn.Module):
    def __init__(self, pretrain=False, freeze_pretain=False):
        super(VGG16Baseline, self).__init__()
        # configs                                                                                            
        self.pretrain = pretrain
        self.freeze_pretain = freeze_pretain

        # model architecture                                                                                 
        self.proc_depth = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.proc_rgb = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.features_compress = nn.Sequential(
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )

        #        self.features_expand = nn.Sequential(                                                               
#            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),                                          
#            nn.ReLU(inplace=True),                                                                          
#            nn.Conv2d(128, 128, kernel_size=5, padding=2),                                                  
#            nn.ReLU(inplace=True),                                                                          
#            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=2, output_padding=1),              
#            nn.ReLU(inplace=True)                                                                           
#        )                                                                                                   

        self.features_expand = nn.Sequential(

            # Expand the feature maps from 512 to 256, upsample by 2                                                            
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),

            # 2D convolution and ReLU activation                                                                                
            nn.Conv2d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            
            # Expand the feature maps from 256 to 128, upsample by 2                                         
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),

            # 2D convolution and ReLU activation                                                             
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            # Expand the feature maps from 128 to 64, upsample by 2                                          
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),

            # 2D convolution and ReLU activation                                                             
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            # Final upsampling to get to the desired 224x224 size                                            
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )


        self.grasp = nn.Sequential(
            nn.ConvTranspose2d(64, 5, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh()
        )

        self.confidence = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

        # initialize weights                                                                                 
        self._initialize_weights()


         # pretrain                                                                                           
        if self.pretrain:
            self.load_pretrain()
        if self.freeze_pretain:
            self.freeze_pretrain()

    def load_pretrain(self):
        vgg16 = vgg16_bn(pretrained=True)
        self.proc_rgb = vgg16.features[0]
        self.features_compress = vgg16.features[1:23]

    def freeze_pretrain(self):
        for m in self.features_compress.modules():
            m.requires_grad = False
        self.proc_rgb.requires_grad = False

    def unfreeze_pretrain(self):
        for m in self.features_compress.modules():
            m.requires_grad = True
        self.proc_rgb.requires_grad = True

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)

        rgb = self.proc_rgb(rgb)
        d = self.proc_depth(d)
        x = rgb+d
        #print("debug1:", x.shape)                                                                           

        x = self.features_compress(x)
        #print("debug2:", x.shape)                                                                           
        x = self.features_expand(x)
        #print("debug3:", x.shape)                                                                           
        grasp = self.grasp(x)
        confidence = self.confidence(x)
        out = torch.cat((grasp, confidence), dim=1)
        #print("debug4:", out.shape)                                                                         
        return out

    def _initialize_weights(self):
        # xavier initialization                                                                              
        if not self.pretrain:
            for m in self.features_compress.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight, gain=1)

            nn.init.xavier_uniform_(self.proc_rgb.weight, gain=1)

        nn.init.xavier_uniform_(self.proc_depth.weight, gain=1)

        for m in self.features_expand.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1)

        for m in self.grasp.modules():
            if isinstance(m, (nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

        for m in self.confidence.modules():
            if isinstance(m, (nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
 
