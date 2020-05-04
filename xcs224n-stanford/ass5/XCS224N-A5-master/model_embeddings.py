#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool1d

from cnn import CNN
from highway import Highway
# End "do not change"
from vocab import VocabEntry


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)


class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self,
                 w_embed_size,
                 vocab: VocabEntry,
                 c_embed_size: int = 50,
                 dropout_prob: float = 0.3):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for
        documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = w_embed_size
        self.charEmbeddings = nn.Embedding(len(vocab.char2id),
                                           c_embed_size,
                                           padding_idx=vocab.char2id['<pad>'])
        self.cnn = CNN(c_embed_size, w_embed_size)
        self.highway = Highway(w_embed_size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x_padded):
        """
        Looks up character-based CNN embeddings for the words in a batch of
        sentences.
        @param x_padded: Tensor of integers of shape (sentence_length,
        batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size,
        embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        # (sentence_length, batch_size, max_word_length)
        x_padded = x_padded.permute(1, 0, 2)
        # (batch_size, sentence_length, max_word_length)
        x_emb = self.charEmbeddings(x_padded)
        # (batch_size, sentence_length, max_word_length, c_embed_size)
        x_reshaped = x_emb.permute(0, 1, 3, 2)
        x_word_emb = []
        for i in range(x_reshaped.size()[0]):
            x_word_emb_i = self._words_batch(x_reshaped[i, :, :, :])
            x_word_emb.append(x_word_emb_i)
        x_word_emb = torch.stack(x_word_emb)
        # (batch_size, sentence_length, w_embed_size)
        x_word_emb = x_word_emb.permute(1, 0, 2)
        # (sentence_length, batch_size, w_embed_size)
        return x_word_emb

    def _words_batch(self, x_reshaped):
        # (sentence_length, c_embed_size, max_word_length)
        x_conv = self.cnn(x_reshaped)
        # (sentence_length, w_embed_size, max_word_length-k+1)
        x_conv_out = max_pool1d(relu(x_conv), kernel_size=x_conv.size()[-1])
        # (sentence_length, w_embed_size, 1)
        x_conv_out = x_conv_out.reshape(x_conv_out.size()[:-1])
        # (sentence_length, w_embed_size)
        x_highway = self.highway(x_conv_out)
        # (sentence_length, w_embed_size)
        x_word_emb = self.dropout(x_highway)
        # (sentence_length, w_embed_size)
        return x_word_emb