import torch
import torch.nn.functional as F

from models import *
from corpus import Corpus


class TestWorking:
    def __init__(self, dataset_name, path, model_class, **kwargs):
        self.__dict__.update(kwargs)

        self.cp = Corpus(dataset_name)

        model_kwargs = {
            'distinct_word_cnt': len(self.cp.idx2word_dict),
            'embd_vector_dim': self.embd_vector_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }
        self.model = model_class(**model_kwargs)
        self.model.load_state_dict(torch.load(path)['model'])
        self.model.eval()
        print(f"pplx: {torch.load(path)['pplx']:.2f}")

    def next_word_list(self, init_sentence, k, stochastic=True):
        init_indices = [self.cp.word2idx_dict[w] for w in init_sentence.split(' ')]
        input = torch.tensor(init_indices).view(1, -1).t()

        hidden = self.model.init_hidden(1)  # initial hidden state

        output, hidden = self.model.forward(input, hidden)

        if stochastic:
            if len(init_indices) > 1:
                next_indices = torch.multinomial(F.softmax(output.squeeze(), 1), k).tolist()
                next_indices = next_indices[-1] # last ouptut should be the last word of input sentence
            else:
                next_indices = torch.multinomial(F.softmax(output.squeeze(), 0), k).tolist()
            next_words = [self.cp.idx2word_dict[i] for i in next_indices]

        else: # i.e. deterministic
            if len(init_indices) > 1:
                next_indices = torch.topk(F.softmax(output.squeeze(), 1), k).indices.tolist()
                next_proba = torch.topk(F.softmax(output.squeeze(), 1), k).values.tolist()
                next_indices = next_indices[-1]
                next_proba = next_proba[-1]
            else:
                next_indices = torch.topk(F.softmax(output.squeeze(), 0), k).indices.tolist()
                next_proba = torch.topk(F.softmax(output.squeeze(), 0), k).values.tolist()
            next_words = [(self.cp.idx2word_dict[idx], next_proba[i]) for i, idx in enumerate(next_indices)]

        return next_words

    def generate_sentence(self, init_sentence, add_len=10, stochastic=True):
        init_indices = [self.cp.word2idx_dict[w] for w in init_sentence.split(' ')]

        indices = init_indices.copy()
        words = init_sentence.split(' ')

        hidden = self.model.init_hidden(1)  # initial hidden state

        for _ in range(add_len):
            input = torch.tensor(indices).view(1, -1).t()

            output, hidden = self.model.forward(input, hidden)

            if stochastic:
                next_idx = torch.multinomial(F.softmax(output.squeeze(), 0), 1)
            else:
                next_idx = torch.topk(F.softmax(output.squeeze(), 0), 1).indices.tolist()

            if len(indices) > 1:
                next_idx = int(next_idx[-1][0])
            else:
                next_idx = int(next_idx[0])

            indices.append(next_idx)
            words.append(self.cp.idx2word_dict[next_idx])

        return ' '.join(words)


if __name__ == '__main__':
    tw = TestWorking(dataset_name='ptb',
                     path='saved_models/best_GRU_20220314_0606.pth',
                     model_class=GRU,
                     **{'embd_vector_dim': 500, 'hidden_size': 200, 'num_layers': 3, 'batch_size': 256, 'seq_len': 25})

    init = "seoul is the largest city"
    result_sentence = tw.generate_sentence(init_sentence=init, add_len=20, stochastic=False)
    print(result_sentence)

    """
    generated sentences example (not stochastically)
    - seoul is : n't disclosed by least half of goods ' ability to keep them out of dollars us$ 300-a-share bid was quoted
    - seoul is the largest : stake by moody lehman hutton inc </s> moreover there are no longer be able to keep them out of dollars
    - seoul is the largest city : department store makers ' own account force by moody lehman hutton inc </s> moreover there are no longer be able
    """
