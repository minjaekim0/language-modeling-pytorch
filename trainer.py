import math
import time
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from corpus import *


class Trainer:
    def __init__(self, train_data, valid_data, batch_size, seq_len, num_epochs,
                 model, criterion, optimizer, scheduler=None):
        self.train_data = train_data
        self.valid_data = valid_data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_epochs = num_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.num_batches_train = len(self.train_data) // (batch_size * seq_len) + 1
        self.num_batches_valid = len(self.valid_data) // (batch_size * seq_len) + 1

        self.device = torch.device('cuda')
        self.model.to(self.device)

        self.train_loss_list = []
        self.train_pplx_list = []
        self.valid_loss_list = []
        self.valid_pplx_list = []
        self.last_save_path = None

        self.start = time.time()


    def train_one_epoch(self, epoch):
        """train model for one epoch"""

        self.model.train()

        train_loss = 0
        hiddens = self.model.init_hidden(self.batch_size) # initial hidden state

        if type(hiddens) == torch.Tensor:
            hiddens = hiddens.to(self.device)
        else:
            hiddens = tuple([h.to(self.device) for h in hiddens])

        for _batch_idx in range(self.num_batches_train):
            # _batch_idx starts from 0
            # batch_idx = _batch_idx + 1 starts from 1
            batch_idx = _batch_idx + 1

            inputs, targets = lm_batch(self.train_data, self.batch_size, self.seq_len, _batch_idx)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if type(hiddens) == torch.Tensor:
                hiddens = hiddens.detach() # truncated bptt
            else:
                hiddens = tuple([h.detach() for h in hiddens])

            self.optimizer.zero_grad()
            outputs, hiddens = self.model(inputs, hiddens)

            outputs_flattened = outputs.view(-1, self.model.distinct_word_cnt)
            targets_flattened = targets.flatten()
            loss = self.criterion(outputs_flattened, targets_flattened)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            actual_train_loss = train_loss / batch_idx
            train_pplx = np.exp(actual_train_loss)

            now = time.time()
            print('\r', end='')
            print(self.status_bar(now-self.start, epoch, 'train', batch_idx,
                                  actual_train_loss, train_pplx), end='  ')
        print('\n', end='')

        return {
            'epoch': epoch,
            'loss': train_loss / self.num_batches_train,
            'pplx': np.exp(train_loss / self.num_batches_train)
        }


    def valid_one_epoch(self, epoch):
        """valid model for one epoch"""

        self.model.eval()

        valid_loss = 0
        hiddens = self.model.init_hidden(self.batch_size) # initial hidden state

        if type(hiddens) == torch.Tensor:
            hiddens = hiddens.to(self.device)
        else:
            hiddens = tuple([h.to(self.device) for h in hiddens])

        with torch.no_grad():
            for _batch_idx in range(self.num_batches_valid):
                # _batch_idx starts from 0
                # batch_idx = _batch_idx + 1 starts from 1
                batch_idx = _batch_idx + 1

                inputs, targets = lm_batch(self.valid_data, self.batch_size, self.seq_len, _batch_idx)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs, hiddens = self.model(inputs, hiddens)

                outputs_flattened = outputs.view(-1, self.model.distinct_word_cnt)
                targets_flattened = targets.flatten()
                loss = self.criterion(outputs_flattened, targets_flattened)
                self.optimizer.step()

                valid_loss += loss.item()
                actual_valid_loss = valid_loss / batch_idx
                valid_pplx = np.exp(actual_valid_loss)

                now = time.time()
                print('\r', end='')
                print(self.status_bar(now-self.start, epoch, 'valid', batch_idx,
                                      actual_valid_loss, valid_pplx), end='  ')
            print('\n\n', end='')

            return {
                'epoch': epoch,
                'loss': valid_loss / self.num_batches_valid,
                'pplx': np.exp(valid_loss / self.num_batches_valid)
            }


    def save(self, epoch, loss, pplx):
        """model save"""

        # remove last saved model
        if self.last_save_path:
            os.remove(self.last_save_path)

        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'pplx': pplx
        }
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()

        path = f'saved_models/best_{self.model._get_name()}_{datetime.now().strftime("%Y%m%d_%H%M")}.pth'
        torch.save(state, path)

        self.last_save_path = path


    def run(self):
        """execute overall training process"""

        for i in range(1, self.num_epochs+1):
            train_result = self.train_one_epoch(i)
            valid_result = self.valid_one_epoch(i)

            if self.scheduler:
                self.scheduler.step()
            self.train_loss_list.append(train_result['loss'])
            self.train_pplx_list.append(train_result['pplx'])
            self.valid_loss_list.append(valid_result['loss'])
            self.valid_pplx_list.append(valid_result['pplx'])

            last_loss = self.valid_loss_list[-1]
            last_pplx = self.valid_pplx_list[-1]
            if last_pplx == min(self.valid_pplx_list):
                self.save(i, last_loss, last_pplx)


    def plot(self):
        """plot train/valid loss and accuracy vs epochs"""

        sns.set_style("darkgrid")
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        x = list(range(1, self.num_epochs + 1))
        sns.lineplot(x=x, y=self.train_loss_list, label='train_loss', ax=ax1)
        sns.lineplot(x=x, y=self.valid_loss_list, label='valid_loss', ax=ax1)
        sns.lineplot(x=x, y=self.train_pplx_list, label='train_acc', ax=ax2)
        sns.lineplot(x=x, y=self.valid_pplx_list, label='valid_acc', ax=ax2)

        ax1.set_ylabel('loss')
        ax2.set_ylabel('perplexity')
        ax2.set_xlabel('epoch')

        ax1.legend(ax1.get_legend_handles_labels()[1], loc='best')
        ax2.legend(ax2.get_legend_handles_labels()[1], loc='best')

        fig.savefig(f'plots/loss_acc_plot_{self.model._get_name()}_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        fig.show()


    def status_bar(self, time_elapsed, epoch, train_or_valid, batch_idx, loss, pplx, length=50):
        """show status at train_one_epoch & valid_one_epoch"""
        if train_or_valid == 'train':
            num_batches = self.num_batches_train
        else:
            num_batches = self.num_batches_valid

        rate = batch_idx / num_batches
        num_equals = math.floor(length * rate) - 1
        num_dots = length - num_equals - 1

        def fm(x):
            if x < 1e3:
                return f'{x:.2f}'
            else:
                return f'{x:.2e}'
        status = f'[{time_elapsed // 60:3.0f}m {time_elapsed % 60:5.2f}s | ' \
                 f'epoch: {epoch}/{self.num_epochs} | {train_or_valid:5s}]' \
                 f'[{"=" * num_equals}>{"." * num_dots}]' \
                 f'[loss:{loss:6.3f} | pplx: {fm(pplx)}]'

        return status


def lm_batch(idx_list, batch_size, seq_len, batch_idx):
    cnt_idx_one_batch = batch_size * seq_len
    initial_idx = batch_idx * cnt_idx_one_batch
    final_idx = (batch_idx + 1) * cnt_idx_one_batch

    if final_idx >= len(idx_list):
        final_idx = initial_idx + (len(idx_list) - initial_idx) // batch_size * batch_size

    data = idx_list[initial_idx : final_idx]
    target = idx_list[initial_idx+1 : final_idx+1]

    data = torch.tensor(data).view(batch_size, -1).t().contiguous()
    target = torch.tensor(target).view(batch_size, -1).t().contiguous()

    return data, target
