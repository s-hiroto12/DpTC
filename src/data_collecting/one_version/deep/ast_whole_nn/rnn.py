import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn as nn


class GRUClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, use_gpu):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu else "cpu"
        # gru
        self.lstm = nn.GRU(embedding_dim, hidden_dim, num_layers = 1, bidirectional=True, 
                            batch_first = True)
        # linear
        self.hidden2label = nn.Linear(hidden_dim * 2, label_size)
        # hidden
        self.hidden = self.init_hidden()
        

    def init_hidden(self):
        if self.use_gpu:
            # h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            # c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            return torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim, device=self.device)
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, x):
        # gru
        gru_out, self.hidden = self.lstm(x, self.hidden)

        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)

        # linear
        y = self.hidden2label(gru_out)

        return y
