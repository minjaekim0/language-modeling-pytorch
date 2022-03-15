import torch
import torchnlp.datasets as datasets


def word_list(name):
    """return train / valid / test dataset"""
    if name == 'ptb':
        train_word_list = datasets.penn_treebank_dataset(directory='dataset/ptb', train=True)
        valid_word_list = datasets.penn_treebank_dataset(directory='dataset/ptb', dev=True)
        test_word_list = datasets.penn_treebank_dataset(directory='dataset/ptb', test=True)

        return train_word_list, valid_word_list, test_word_list

    else:
        # to be added
        return


class Corpus:
    def __init__(self, name):
        self.train_word_list, self.valid_word_list, self.test_word_list = word_list(name)

        self.word2idx_dict = {}
        self.idx2word_dict = {}
        self.make_dictionaries()

        self.train_idx_list = self.make_idx_list(self.train_word_list)
        self.valid_idx_list = self.make_idx_list(self.valid_word_list)
        self.test_idx_list = self.make_idx_list(self.test_word_list)

    def make_dictionaries(self):
        idx = 0
        for word_list in [self.train_word_list, self.valid_word_list, self.test_word_list]:
            for word in word_list:
                if word not in self.word2idx_dict.keys():
                    self.word2idx_dict[word] = idx
                    self.idx2word_dict[idx] = word
                    idx += 1

    def make_idx_list(self, word_list):
        idx_list = []
        for word in word_list:
            idx_list.append(self.word2idx_dict[word])

        return idx_list
