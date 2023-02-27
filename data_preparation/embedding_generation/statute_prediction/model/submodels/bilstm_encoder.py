#!usr/bin/env python

"""BiLSTM-based document/sentence encoder with configurable vocabulary"""

import torch
import torch.nn as nn

__author__ = "Upal Bhattacharya"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


class BiLSTMEncoder(nn.Module):
    """BiLSTM-based document/sentence encoder with configurable vocabulary"""

    def __init__(
            self,
            vocab_size: int,
            emb_dim: int,
            hidden_dim: int,
            num_layers: int = 1,
            drop: float = 0.5,
            device: str = 'cuda',
            vocab_path: str = None):

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop = drop
        self.device = device
        self.vocab_path = vocab_path

        self.bilstm = nn.LSTM(
                input_size=self.emb_dim,
                hidden_size=self.hidden_dim // 2,
                num_layers=self.num_layers,
                bias=True,
                batch_first=True,
                dropout=self.droput,
                bidirectional=True)

        self.initial_states = None  # Updated in forward()

        if self.vocab_path is not None:
            self.emb = torch.load(self.vocab_path)
        else:
            self.emb = nn.Embedding(vocab_size, emb_dim)
        if self.emb.shape != (self.vocab_size, self.emb_dim):
            raise ValueError(
                    ("Incorrect shape for vocabulary embeddings."
                     f"Expected ({self.vocab_size}, {self.emb_dim})"
                     f"Got {self.emb.shape}"))

    def _initialise_states(self, batch_size: int):
        return (torch.randn(
                2, batch_size, self.hidden_dim // 2)).to(self.device)
