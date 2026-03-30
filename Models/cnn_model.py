import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, vocab_size, embed_dim=200, num_classes=3):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.conv3 = nn.Conv1d(embed_dim, 256, kernel_size=3)
        self.conv4 = nn.Conv1d(embed_dim, 256, kernel_size=4)
        self.conv5 = nn.Conv1d(embed_dim, 256, kernel_size=5)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x3 = F.relu(self.conv3(x))
        x4 = F.relu(self.conv4(x))
        x5 = F.relu(self.conv5(x))

        x3 = F.max_pool1d(x3, x3.shape[2]).squeeze(2)
        x4 = F.max_pool1d(x4, x4.shape[2]).squeeze(2)
        x5 = F.max_pool1d(x5, x5.shape[2]).squeeze(2)

        x = torch.cat((x3, x4, x5), dim=1)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x