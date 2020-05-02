#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np


def pad_sents_char(sents, char_pad_token, max_word_lenght: int = 21):
    """ Pad list of sentences according to the longest sentence in the batch
    and max_word_length.
    @param sents (list[list[list[int]]]): list of sentences, result of
    `words2charindices()`
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where
    sentences/words shorter
        than the max length sentence/word are padded out with the appropriate
        pad token, such that
        each sentence in the batch now has same number of words and each word
        has an equal
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
        Words longer than 21 characters should be truncated
    """
    sents_padded = []
    max_sen_len = max((len(s) for s in sents))
    padding_word = [char_pad_token] * max_word_lenght
    for sen in sents:
        sen_pad = []
        for word in sen:
            diff = max_word_lenght - len(word)
            if diff >= 0:
                word += [char_pad_token] * diff
            else:
                word = word[:max_word_lenght]
            sen_pad.append(word)
        diff = max_sen_len - len(sen_pad)
        if diff > 0:
            sen_pad += [padding_word] * diff
        sents_padded.append(sen_pad)
    return sents_padded


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where
    sentences shorter
        than the max length sentence are padded out with the pad_token,
        such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for s in sents:
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)

    return sents_padded


def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length
    (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing
    source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
