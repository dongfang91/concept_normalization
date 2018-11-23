# _*_ coding: utf-8 _*_

import os
import sys
import numpy as np
from nltk.tokenize import word_tokenize
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

def get_vocab(texts, label_texts, labels):
    x = texts[0]+ texts[1] +texts[2]
    label_texts = label_texts[0]+label_texts[1] +label_texts[2]
    y = labels[0] + labels[1] +labels[2]
    label_y_dict = {}
    text_x_dict = {}
    label_text_dict = {}
    for index, text_sent in enumerate(x+label_texts):
        for text_item in word_tokenize(text_sent):
            text_item = text_item.lower()
            if text_item not in text_x_dict:
                text_x_dict[text_item] = 1
            else:
                text_x_dict[text_item] +=1
    vocab_dict = {token_stem.lower(): index for index, token_stem in enumerate(list(text_x_dict.keys()), start=1)}

    for index,label_elem in enumerate(y):
        if label_elem not in label_text_dict:
            label_text_dict[label_elem] = label_texts[index]
        if label_elem not in label_y_dict:
            label_y_dict[label_elem] = 1
        else:
            label_y_dict[label_elem] += 1
    label_text_key_text = {}
    label_text_dict_dupl = {}
    for conc, text in label_text_dict.items():
        if text in label_text_key_text:
            text_new = text +str(random.randint(1,10))
            while text_new in label_text_key_text:
                text_new = text + str(random.randint(1, 10))
            label_text_dict_dupl[conc] = text_new
        else:
            label_text_key_text[text] = conc

    for conc, text in label_text_dict_dupl.items():
        label_text_dict[conc] = text

    label_text_list = list(label_text_dict.values())
    label_text_list.sort(key=len,reverse=True)

    label_text2int = {label: index for index, label in enumerate(label_text_list, start=0)}

    label_text_key_text = {text: conc for conc, text in label_text_dict.items()}

    label_dict = {label_text_key_text[label]: index for label, index in label_text2int.items()}
    return vocab_dict, label_dict , label_text_dict

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
