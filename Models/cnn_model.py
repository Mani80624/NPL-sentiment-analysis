import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, vocab_size, embed_dim=100, num_classes=3):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(embed_dim, 100, 3)
        self.conv2 = nn.Conv1d(embed_dim, 100, 4)
        self.conv3 = nn.Conv1d(embed_dim, 100, 5)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):

        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(x))
        c3 = F.relu(self.conv3(x))

        p1 = F.max_pool1d(c1, c1.shape[2]).squeeze(2)
        p2 = F.max_pool1d(c2, c2.shape[2]).squeeze(2)
        p3 = F.max_pool1d(c3, c3.shape[2]).squeeze(2)

        out = torch.cat([p1, p2, p3], dim=1)
        out = self.dropout(out)

        return self.fc(out)