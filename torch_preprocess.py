# _*_ coding: utf-8 _*_

import os
import sys
import numpy as np
import random


def load_data(training_file, validation_file, testing_file):
    # Load data from files
    texts = []
    labels = []
    label_texts = []
    for file_n in [training_file, validation_file, testing_file]:
        txts = []
        lbs = []
        lb_txts = []
        with open(file_n,'r',encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                label, label_text, text = line.split("\t")
                txts.append(text)
                lbs.append(label)
                lb_txts.append(label_text)
        texts.append(txts)
        labels.append(lbs)
        label_texts.append(lb_txts)

    return texts, label_texts, labels

def get_idx_from_sent(sent, word_idx_map, max_l):
    """
    Transforms sentence into a list of indices. Post-Pad with zeroes.
    """
    x = []
    # for i in range(pad):
    #     x.append(word_idx_map[padding_char])
    for word in sent:
        word = word.lower()
        if word in word_idx_map.keys():
            x.append(word_idx_map[word])
        else:
            x.append(word_idx_map["UNK"])
    while len(x) < max_l:
        x.append(0)
    return x[:max_l]

def get_idx_from_sent_character(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Post-Pad with zeroes.
    """
    x = []
    # for i in range(pad):
    #     x.append(word_idx_map[padding_char])
    for word in sent:
        word = word.lower()
        if word in word_idx_map.keys():
            x.append(word_idx_map[word])
        else:
            x.append(word_idx_map["UNK"])
    return x

def dataset_preprocess_character(texts,labels, vocab_dict,label_dict):
    text_x = []
    label_y = []
    sent_length = []

    for index, sent in enumerate(texts):
        #word_list = word_tokenize(sent)
        sent_length.append(len(sent))
        text_x.append(get_idx_from_sent_character(sent,vocab_dict))
        # if index <=2000:
        #     label_y.append(0)
        # elif 3000<=index<=7000:
        #     label_y.append(1)
        # else:
        #     label_y.append(2)
        label_y.append(label_dict[labels[index]])
    return text_x,np.asanyarray(label_y),np.asarray(sent_length)

def label_preprocess_character(label_dict,label_texts_dict,vocab_dict):
    label_dict_new = {index:label for label, index in label_dict.items()}
    labels_sorted = [label_texts_dict[label_dict_new[i]] for i in range(len(label_dict))]
    len_max = [len(labels_sorted_single) for labels_sorted_single in labels_sorted]
    label_x = [get_idx_from_sent_character(label,vocab_dict) for label in labels_sorted]
    return label_x,np.asarray(len_max)

def dataset_preprocess_character_pretrained(texts,labels, vocab_dict,label_dict):
    text_x = []
    label_y = []
    sent_length = []

    for index, sent in enumerate(texts):
        #word_list = word_tokenize(sent)
        sent_length.append(len(sent))
        #text_x.append(get_idx_from_sent_character(sent,vocab_dict))
        # if index <=2000:
        #     label_y.append(0)
        # elif 3000<=index<=7000:
        #     label_y.append(1)
        # else:
        #     label_y.append(2)
        label_y.append(label_dict[labels[index]])
    return texts,np.asanyarray(label_y),np.asarray(sent_length)
