#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language.
        See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.vocab_size = len(target_vocab.char2id)
        self.charDecoder = nn.LSTM(input_size=char_embedding_size,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   bias=True,
                                   bidirectional=False)
        self.char_output_projection = \
            nn.Linear(in_features=hidden_size,
                      out_features=self.vocab_size,
                      bias=True)
        self.decoderCharEmb = \
            nn.Embedding(num_embeddings=self.vocab_size,
                         embedding_dim=char_embedding_size,
                         padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the
        input characters. A tuple of two tensors of shape (1, batch,
        hidden_size)

        @returns scores: called s in the PDF, shape (length, batch,
        self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the
        input characters. A tuple of two tensors of shape (1, batch,
        hidden_size)
        """
        # (length, batch)
        x = self.decoderCharEmb(input)
        # (length, batch, char_embed_size)
        h, dec_hidden = self.charDecoder(x, dec_hidden)
        # (length, batch, char_embed_size)
        s = self.char_output_projection(h)
        # (length, batch, self.vocab_size)
        return s, dec_hidden

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note
        that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from
        the output of the word-level decoder. A tuple of two tensors of shape
        (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of
        cross-entropy losses of all the words in the batch, for every
        character in the sequence.
        """
        batch = char_sequence.shape[-1]
        # (length, batch)
        x = char_sequence[:-1, :]
        # (length-1, batch)
        s, _ = self.forward(x, dec_hidden)
        # (length-1, batch, self.vocab_size)
        char_pad_ix = self.target_vocab.char2id['<pad>']
        lossCalc = nn.CrossEntropyLoss(reduction='sum',
                                       ignore_index=char_pad_ix)
        loss = 0
        for i in range(batch):
            s_i = s[:, i, :].squeeze()
            # (length-1, self.vocab_size)
            tgt_i = char_sequence[1:, i].squeeze()
            # (length-1)
            loss += lossCalc(s_i, tgt_i)
        return loss

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of
        two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or
        GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of
        which has length <= max_length.
                              The decoded strings should NOT contain the
                              start-of-word and end-of-word characters.
        """
        batch = initialStates[0].shape[1]
        start_ix = self.target_vocab.char2id['{']
        curr_char = torch.tensor([start_ix] * batch,
                                 device=device,
                                 dtype=torch.long)
        # (batch)
        curr_char = curr_char.unsqueeze(dim=0)
        # (1, batch)
        dec_state = initialStates
        softmax = nn.Softmax(dim=2)
        outputWords = torch.empty(size=(max_length, batch),
                                  dtype=torch.long)
        for i in range(max_length):
            s, dec_state = self.forward(curr_char, dec_state)
            # (1, batch, self.vocab_size)
            p = softmax(s)
            # (1, batch, self.vocab_size)
            curr_char = torch.argmax(p, dim=2, keepdim=False)
            # (1, batch)
            outputWords[i, :] = curr_char
        return self._decodeWords(batch, outputWords)

    def _decodeWords(self, batch, outputWords):
        end_ix = self.target_vocab.char2id['}']
        id2char = self.target_vocab.id2char
        decodedWords = []
        for j in range(batch):
            word = outputWords[:, j].tolist()
            wordLen = word.index(end_ix)
            word = word[:wordLen]
            decWord = ''.join([id2char[i] for i in word])
            decodedWords.append(decWord)
        return decodedWords
