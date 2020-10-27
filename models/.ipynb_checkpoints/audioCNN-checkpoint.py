import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.maxpool = nn.AdaptiveAvgPool2d(32)
        self.conv1 = nn.Conv2d(1,8,3)
        self.maxpool1 = nn.AdaptiveAvgPool2d(16)
        self.dropout = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv2d(8,16,3)
        self.maxpool2 = nn.AdaptiveAvgPool2d(8)
        
        self.fc1 = nn.Linear(1024, 512) #8*8 = adaptivemaxpool, *16 = conv2d op
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.maxpool(x)
        x = x.view(-1,1,32,32)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x.view(-1,8*8*16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x