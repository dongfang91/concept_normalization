import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.dropout = nn.Dropout(0.5)


    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        if torch.cuda.is_available():
            idx = idx.cuda()

        return unpacked.gather(1, idx).squeeze()

    def init_hidden(self):
        if torch.cuda.is_available():
             h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).cuda()
             c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).cuda()
        else:
             h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
             c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence,lengths):

        packed = torch.nn.utils.rnn.pack_padded_sequence(sentence, lengths,batch_first=True)
        lstm_out, self.hidden = self.lstm(packed, self.hidden)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,batch_first=True)
        # get the outputs from the last *non-masked* timestep for each sentence
        last_outputs = self.last_timestep(unpacked, unpacked_len)
        last_outputs = self.dropout(last_outputs)
        hidden_1 = self.hidden2hidden1(last_outputs)
        hidden_1 = self.relu1(hidden_1)
        y = self.hidden2label(hidden_1)
        return y
