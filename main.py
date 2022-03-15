import torch
import torch.nn as nn
import torch.optim as optim

from models import *
from corpus import Corpus
from trainer import Trainer


if __name__ == '__main__':
    cp = Corpus(name='ptb')
    train_data = cp.train_idx_list + cp.valid_idx_list
    test_data = cp.test_idx_list

    model_list = [
        LSTM(distinct_word_cnt=len(cp.idx2word_dict), embd_vector_dim=500, hidden_size=200, num_layers=3),
        GRU(distinct_word_cnt=len(cp.idx2word_dict), embd_vector_dim=500, hidden_size=200, num_layers=3)
    ]

    for model in model_list:
        model.to(torch.device('cuda')) # necessary to load optimizer's state_dict with proper device

        batch_size = 256
        seq_len = 25
        num_epochs = 500

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        load_path = None
        # load_path = 'saved_models/best_GRU_20220312_2035.pth'

        if load_path:
            state_dict = torch.load(load_path)
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            optimizer.load_state_dict(state_dict['scheduler'])

        tr = Trainer(train_data, test_data, batch_size, seq_len, num_epochs, model, criterion, optimizer, scheduler)
        tr.run()
        tr.plot()
