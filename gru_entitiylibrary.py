import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.label_size = label_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2,bidirectional=True)
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
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim//2), requires_grad = False).cuda()
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim//2), requires_grad = False).cuda()
            h1 = Variable(torch.zeros(2, self.label_size, self.hidden_dim//2), requires_grad = False).cuda()
            c1 = Variable(torch.zeros(2, self.label_size, self.hidden_dim//2), requires_grad = False).cuda()

        else:
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim//2), requires_grad = False)
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim//2), requires_grad = False)
            h1 = Variable(torch.zeros(2, self.label_size, self.hidden_dim//2), requires_grad = False)
            c1 = Variable(torch.zeros(2, self.label_size, self.hidden_dim//2), requires_grad = False)
        return (h0,c0),(h1,c1)

    def forward(self, sentence,lengths,label_input,label_seq_input):
        (h0, c0), (h1, c1) = self.init_hidden()

        embeds = self.word_embeddings(sentence)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths,batch_first=True)
        lstm_out, (h0,c0) = self.lstm(packed,(h0,c0) )
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,batch_first=True)
        # get the outputs from the last *non-masked* timestep for each sentence
        last_outputs = self.last_timestep(unpacked, unpacked_len)
        last_outputs = self.dropout(last_outputs)
        hidden_1 = self.hidden2hidden1(last_outputs)
        hidden_1 = self.relu1(hidden_1)

        label_embeds = self.word_embeddings(label_input)
        label_packed = torch.nn.utils.rnn.pack_padded_sequence(label_embeds, label_seq_input,batch_first=True)
        label_lstm_out, (h1, c1) = self.lstm(label_packed, (h1, c1) )
        label_unpacked, label_unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(label_lstm_out,batch_first=True)
        # get the outputs from the last *non-masked* timestep for each sentence
        label_last_outputs = self.last_timestep(label_unpacked, label_unpacked_len)
        label_last_outputs = self.dropout(label_last_outputs)
        label_hidden_1 = self.hidden2hidden1(label_last_outputs)
        label_hidden_1 = self.relu1(label_hidden_1)
        score = torch.mm(hidden_1,torch.t(label_hidden_1))


        #y = self.hidden2label(hidden_1)
        return score
