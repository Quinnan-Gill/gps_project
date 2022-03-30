import torch
import torch.nn as nn

from cnn_model import DoubleConv, Down, Up, OutConv

class TDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AutoEnc(nn.Module):
    def __init__(self, n_channels, n_classes=2, embedding_size=16):
        super(AutoEnc, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 128)
        self.enc1 = DoubleConv(128, 64)
        self.enc2 = DoubleConv(64, 32)
        #self.enc3 = DoubleConv(32, 16)

        self.fc1 = nn.Linear(32, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 32)

        # factor = 2 if bilinear else 1
        # 
        #self.decode1 = TDoubleConv(16, 32)
        self.decode2 = TDoubleConv(32, 64)
        self.decode3 = TDoubleConv(64, 128)
        self.outc = DoubleConv(128, n_channels)
        # self.outc = OutConv(64, n_classes)

    def forward(self, x):

        encoding_output1 = self.inc(x)

        encoding_output2 = self.enc1(encoding_output1)

        encoding_output3 = self.enc2(encoding_output2)

        #encoding_output4 = self.enc3(encoding_output3)

        fc_output1 = self.fc1(encoding_output3)
        fc_output2 = self.fc2(fc_output1)
        
        #decoding_output1 = self.decode1(fc_output2)
        decoding_output2 = self.decode2(fc_output2)
        decoding_output3 = self.decode3(decoding_output2)

        logits = self.outc(decoding_output3)
        return logits