# Replication of the model architecture used in the paper DeepDTA in pytorch
# Author: @ksun63
# Date: 2023-04-14

import torch
import torch.nn as nn

class Conv1d(nn.Module):
    """
    Three 1d convolutional layer with relu activation stacked on top of each other
    with a final global maxpooling layer
    """
    def __init__(self, vocab_size, channel, kernel_size, stride=1, padding=0):
        super(Conv1d, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=128)
        self.conv1 = nn.Conv1d(128, channel, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(channel, channel*2, kernel_size, stride, padding)
        self.conv3 = nn.Conv1d(channel*2, channel*3, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.globalmaxpool(x)
        x = x.squeeze(-1)
        return x

class DeepDTA(nn.Module):
    """DeepDTA model architecture, Y-shaped net that does 1d convolution on 
    both the ligand and the protein representation and then concatenates the
    result into a final predictor of binding affinity"""

    def __init__(self, pro_vocab_size, lig_vocab_size, channel, protein_kernel_size, ligand_kernel_size):
        super(DeepDTA, self).__init__()
        self.ligand_conv = Conv1d(lig_vocab_size, channel, ligand_kernel_size)
        self.protein_conv = Conv1d(pro_vocab_size, channel, protein_kernel_size)
        self.fc1 = nn.Linear(channel*6, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, protein, ligand):
        x1 = self.ligand_conv(ligand)
        x2 = self.protein_conv(protein)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x.squeeze()
