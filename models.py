import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.downsample = downsample

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = nn.functional.relu(self.bn1(out))
        out = nn.functional.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = nn.functional.relu(out)

        return out

class PushFCN(nn.Module):
    def __init__(self):
        super(PushFCN, self).__init__()
        self.device = 'cuda'
        self.nr_rotations = 2
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False, groups=2)
        self.rb1 = self.make_layer(64, 128)
        self.rb2 = self.make_layer(128, 256)
        self.rb3 = self.make_layer(256, 512)
        self.rb4 = self.make_layer(512, 256)
        self.rb5 = self.make_layer(256, 128)
        self.rb6 = self.make_layer(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.push_prob = []
        
    def make_layer(self, in_channels, out_channels, blocks=1, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            # downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride),
            #                            nn.BatchNorm2d(out_channels))
            downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride))

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def predict(self, input):
        x = nn.functional.relu(self.conv1(input)) # ReLu is needed?
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.rb1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.rb2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.rb3(x)
        x = self.rb4(x)
        
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.rb5(x)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.rb6(x)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final_conv(x) # activation function?
        return out

    def forward(self,input,specific_rotation=-1,is_volatile=[]):
        self.push_prob = []

        if is_volatile:
            batch_rot_input = torch.zeros((self.nr_rotations,input.shape[1],input.shape[2],input.shape[3])).to('cuda')
            for rot_id in range(self.nr_rotations):
                theta = np.radians(rot_id*(360/self.nr_rotations))
                affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                                [-np.sin(theta), np.cos(theta), 0.0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()

                flow_grid_before = F.affine_grid(
                    Variable(affine_mat_before, requires_grad=False).to(self.device),
                    input.size(),
                    align_corners=True)

                rotate_input = F.grid_sample(
                        Variable(input, requires_grad=False).to(self.device),
                        flow_grid_before,
                        mode='nearest',
                        align_corners=True,
                        padding_mode="border")
                batch_rot_input[rot_id] = rotate_input[0]

            prob = self.predict(batch_rot_input)

                # Undo rotation
            affine_after = torch.zeros((self.nr_rotations, 2, 3))
            for rot_id in range(self.nr_rotations):
                    # Compute sample grid for rotation before neural network
                theta = np.radians(rot_id * (360 / self.nr_rotations))
                affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
                                                [-np.sin(-theta), np.cos(-theta), 0.0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_after[rot_id] = affine_mat_after

            flow_grid_after = F.affine_grid(Variable(affine_after, requires_grad=False).to(self.device),
                                                prob.size(), align_corners=True)
            out_prob = F.grid_sample(prob, flow_grid_after, mode='nearest', align_corners=True)

            return out_prob
        else:
            thetas = [np.radians(specific_rotation * (360 / self.nr_rotations))]
            affine_before = torch.zeros((input.shape[0], 2, 3))
            for i in range(len(thetas)):
                # Compute sample grid for rotation before neural network
                theta = thetas[i]
                affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                              [-np.sin(theta), np.cos(theta), 0.0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                affine_before[i] = affine_mat_before

            flow_grid_before = F.affine_grid(Variable(affine_before, requires_grad=False).to(self.device),
                                             input.size(), align_corners=True)

            # Rotate image clockwise_
            rotate_depth = F.grid_sample(Variable(input, requires_grad=False).to(self.device),
                                         flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")

            # Compute intermediate features
            prob = self.predict(rotate_depth)

            # Compute sample grid for rotation after branches
            affine_after = torch.zeros((input.shape[0], 2, 3))
            for i in range(len(thetas)):
                theta = thetas[i]
                affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
                                             [-np.sin(-theta), np.cos(-theta), 0.0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_after[i] = affine_mat_after

            flow_grid_after = F.affine_grid(Variable(affine_after, requires_grad=False).to(self.device),
                                            prob.size(), align_corners=True)

            # Forward pass through branches, undo rotation on output predictions, upsample results
            out_prob = F.grid_sample(prob, flow_grid_after, mode='nearest', align_corners=True)
            self.push_prob = out_prob

            return out_prob

class PickFCN(nn.Module):
    def __init__(self):
        super(PickFCN, self).__init__()
        self.device = 'cuda'

        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False, groups=2)
        self.rb1 = self.make_layer(64, 128)
        self.rb2 = self.make_layer(128, 256)
        self.rb3 = self.make_layer(256, 512)
        self.rb4 = self.make_layer(512, 256)
        self.rb5 = self.make_layer(256, 128)
        self.rb6 = self.make_layer(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.pick_prob = []

    def make_layer(self, in_channels, out_channels, blocks=1, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            # downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride),
            #                            nn.BatchNorm2d(out_channels))
            downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride))

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def predict(self, input):
        
        
        x = nn.functional.relu(self.conv1(input)) # ReLu is needed?
        
        #Encoder
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.rb1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.rb2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.rb3(x)

        #Decoder
        x = self.rb4(x)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.rb5(x)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.rb6(x)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final_conv(x) # activation function?
        return out

    def forward(self,input,is_volatile=[]):
        self.pick_prob = []
        if is_volatile:
            with torch.no_grad():
                prob = self.predict(input.to(self.device))
            return prob
        else:
           prob = self.predict(input.to(self.device))
           self.pick_prob = prob
           return prob 
