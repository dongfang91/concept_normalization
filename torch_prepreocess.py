# _*_ coding: utf-8 _*_

import os
import sys
import numpy as np
from nltk.tokenize import word_tokenize


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

def get_vocab(texts,label_texts,labels):
    x = texts[0]+ texts[1] +texts[2]
    label_texts = label_texts[0]+label_texts[1]
    y = labels[0] + labels[1] +labels[2]
    label_y_dict = {}
    text_x_dict = {}
    for index,text_sent in enumerate(x+label_texts):
        for text_item in word_tokenize(text_sent):
            text_item = text_item.lower()
            if text_item not in text_x_dict:
                text_x_dict[text_item] = 1
            else:
                text_x_dict[text_item] +=1
    vocab_dict = {token_stem.lower(): index for index, token_stem in enumerate(list(text_x_dict.keys()), start=1)}

    for index,label_elem in enumerate(y):
        if label_elem not in label_y_dict:
            label_y_dict[label_elem] = 1
        else:
            label_y_dict[label_elem] += 1

    label_dict = {label: index for index, label in enumerate(list(label_y_dict.keys()), start=0)}
    return vocab_dict, label_dict

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

def dataset_preprocess(texts,labels, vocab_dict,label_dict,max_length):
    text_x = []
    label_y = []
    sent_length = []

    for index, sent in enumerate(texts):
        word_list = word_tokenize(sent)
        sent_length.append(len(word_list))
        text_x.append(get_idx_from_sent(word_tokenize(sent),vocab_dict,max_length))
        # if index <=2000:
        #     label_y.append(0)
        # elif 3000<=index<=7000:
        #     label_y.append(1)
        # else:
        #     label_y.append(2)
        label_y.append(label_dict[labels[index]])
    return np.asanyarray(text_x),np.asanyarray(label_y),np.asarray(sent_length)


