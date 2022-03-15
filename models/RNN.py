from abc import abstractmethod

import torch
import torch.nn as nn


class RNNParent(nn.Module):
    def __init__(self, distinct_word_cnt, embd_vector_dim, hidden_size, num_layers, dropout=0.5):
        super(RNNParent, self).__init__()

        self.encoder = nn.Embedding(num_embeddings=distinct_word_cnt,
                                    embedding_dim=embd_vector_dim)
        self.decoder = nn.Linear(hidden_size, distinct_word_cnt)
        self.dropout_layer = nn.Dropout(dropout)

        self.init_weights()

        self.distinct_word_cnt = distinct_word_cnt
        self.embd_vector_dim = embd_vector_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, input, hidden):
        emb = self.encoder(input)
        emb = self.dropout_layer(emb)

        output, hidden = self.rnn(emb, hidden)
        output = self.dropout_layer(output)

        output_decoded = self.decoder(output.view(-1, self.hidden_size))
        return output_decoded.view(output.size(0), output.size(1), -1), hidden

    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-init_range, init_range)

    @abstractmethod
    def init_hidden(self, batch_size):
        pass


class LSTM(RNNParent):
    def __init__(self, distinct_word_cnt, embd_vector_dim, hidden_size, num_layers, dropout=0.5):
        super(LSTM, self).__init__(distinct_word_cnt, embd_vector_dim, hidden_size, num_layers, dropout)
        self.rnn = nn.LSTM(input_size=self.embd_vector_dim, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, dropout=self.dropout)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class GRU(RNNParent):
    def __init__(self, distinct_word_cnt, embd_vector_dim, hidden_size, num_layers, dropout=0.5):
        super(GRU, self).__init__(distinct_word_cnt, embd_vector_dim, hidden_size, num_layers, dropout)
        self.rnn = nn.GRU(input_size=self.embd_vector_dim, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, dropout=self.dropout)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


if __name__ == '__main__':
    batch_size = 128
    seq_len = 100
    embd_vector_dim = 5
    hidden_size = 10
    num_layers = 3

    from corpus import Corpus
    from trainer import lm_batch

    cp = Corpus(name='ptb')
    train_idx_list = cp.train_idx_list + cp.valid_idx_list
    test_idx_list = cp.test_idx_list
    train_data_batch, train_target_batch = lm_batch(train_idx_list, batch_size, seq_len, 0)

    model = GRU(distinct_word_cnt=len(cp.idx2word_dict), embd_vector_dim=embd_vector_dim,
                hidden_size=hidden_size, num_layers=num_layers)
    initial_hidden = model.init_hidden(batch_size)
    output, hidden = model.forward(train_data_batch, initial_hidden)
