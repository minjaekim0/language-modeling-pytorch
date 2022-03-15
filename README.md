# language-modeling-pytorch

### implemented models
- LSTM
- GRU

### result for PTB
- seq_len = 25
- \# of epochs = 500
- Adam optimizer: lr = 1e-2, weight_decay = 1e-4
- CosineAnnealingLR scheduler: T_max = 500
- each model: embd_vector_dim = 500, hidden_size = 200, num_layers = 3

| model | perplexity |
|-------|------------|
| LSTM  | 153.66     |
| GRU   | 134.45     |
