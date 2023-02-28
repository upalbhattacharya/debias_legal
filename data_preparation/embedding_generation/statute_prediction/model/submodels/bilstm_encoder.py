#!usr/bin/env python

"""BiLSTM-based document/sentence encoder with configurable vocabulary"""

import torch
import torch.nn as nn
from einops import rearrange

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
            vocab_path: str = None,
            update_emb: bool = False):
        """
        Initialisation of BiLSTM-based sentence encoder

        Parameters
        ----------

        vocab_size : int
            Size of vocabulary.
        emb_dim : int
            Dimensionality of word embeddings.
        hidden_dim : int
            Dimensionality of hidden layer of BiLSTM.
        num_layers : int, default = 1
            Number of stacked biLSTMs.
        drop : float, default = 0.5
            Dropout value for biLSTMs.
        device : str, default = 'cuda'
            Device to execute on.
        vocab_path : str, default = None
            Path to loading vocabulary tensor when using pretrained embeddings.
        update_emb : bool, default = False
            Whether to allow updation of word embeddings during training.
        """

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop = drop
        self.device = device
        self.vocab_path = vocab_path
        self.update_emb = update_emb

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
            pretrained = torch.load(self.vocab_path)
            self.emb = nn.Embedding.from_pretrained(pretrained)
            self.emb.weight.required_grad = self.update_emb
        else:
            self.emb = nn.Embedding(vocab_size, emb_dim)

        if self.emb.shape != (self.vocab_size, self.emb_dim):
            raise ValueError(
                    ("Incorrect shape for vocabulary embeddings."
                     f"Expected ({self.vocab_size}, {self.emb_dim})"
                     f"Got {self.emb.shape}"))

    def _initialise_states(self, batch_size: int):
        self.initial_states = (torch.randn(
                2, batch_size, self.hidden_dim // 2)).to(self.device)

    def forward(self, sents, lengths):
        batch_size = sents.shape[0]
        self._initialise_states(batch_size)
        x = self.emb(sents)
        x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True,
                    enforce_sorted=False)

        _, (x, _) = self.bilstm(x, self.initial_states)

        x = rearrange(x, 'd b h -> b (d h)')
        return x
