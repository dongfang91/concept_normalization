import numpy as np
import torch
from gru import LSTMClassifier
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch_preprocess as function
import math


np.random.seed(123)
torch.manual_seed(123)


## parameter setting


def iterate_minibatches(inputs, targets, sentence_len,batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt],sentence_len[excerpt]

    ### parameter setting





### training procedure
def train(vocab_dict, label_dict, train_x,train_y,train_sentence_len,valid_x,valid_y , valid_sentence_len, dataset,folder):
    embedding_dim = 300
    hidden_dim = 100
    epochs = 60
    batch_size = 50
    learning_rate = 1.0

    model = LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                           vocab_size=len(vocab_dict), label_size=len(label_dict), batch_size=batch_size)
    train_acc_ = []
    valid_acc_ = []
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    valid_loss_ = []
    val_acc = -math.inf
    for epoch in range(epochs):
        ## training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for batch in iterate_minibatches(train_x, train_y,train_sentence_len, batchsize=batch_size, shuffle=True):
            input_batch, target_batch, seq_lens_batch = batch
            train_input_batch = torch.LongTensor(input_batch)
            train_target_batch = torch.LongTensor(target_batch)
            train_seq_lens_batch = torch.LongTensor(seq_lens_batch)
            train_seq_lens_batch, perm_index = train_seq_lens_batch.sort(0, descending=True)
            train_x_batch= train_input_batch[perm_index]
            train_y_batch = train_target_batch[perm_index]
            train_x_varaible = Variable(train_x_batch)
            model.zero_grad()
            model.batch_size = len(train_x_batch)
            model.hidden = model.init_hidden()
            output = model(train_x_batch,train_seq_lens_batch)
            loss = loss_function(output, Variable(train_y_batch))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += np.float((predicted == train_y_batch).sum().item())
            total += len(train_y_batch)
            total_loss += loss.item()

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / total)
        ## validing epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0

        for batch in iterate_minibatches(valid_x, valid_y,valid_sentence_len, batchsize=batch_size, shuffle=False):
            input_batch, target_batch, seq_lens_batch= batch
            valid_input = torch.LongTensor(input_batch)
            valid_label = torch.LongTensor(target_batch)
            valid_seq_lens_batch = torch.LongTensor(seq_lens_batch)
            valid_seq_lens_batch, perm_index = valid_seq_lens_batch.sort(0, descending=True)
            valid_x_batch = valid_input[perm_index]
            valid_y_batch = valid_label[perm_index]
            valid_x_batch = Variable(valid_x_batch)
            model.batch_size = len(valid_x_batch)
            model.hidden = model.init_hidden()
            output = model(valid_x_batch,valid_seq_lens_batch)
            loss = loss_function(output, Variable(valid_y_batch))

            # calc testing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += np.float((predicted == valid_y_batch).sum().item())
            total += len(valid_y_batch)
            total_loss += loss.item()
        if (total_acc/total > val_acc):
            torch.save(model, "data/model/model_" + dataset + "_folder_" + str(folder) + ".pkl")
            val_acc = total_acc/total
        valid_loss_.append(total_loss / total)
        valid_acc_.append(total_acc / total)
        print(
            '[Epoch: %3d/%3d] Training Loss: %.3f, Validating Loss:  %.3f, Training Acc: %.3f, Validing Acc: %.3f'
            % (epoch, epochs, train_loss_[epoch], valid_loss_[epoch],  train_acc_[epoch],
               valid_acc_[epoch]))


#train()

def eval(folder, model,test_x,test_y , test_sentence_len,mode):
    ## testing epoch
    test_loss_ = []
    test_acc_ = []
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0

    batch_size = 50
    for batch in iterate_minibatches(test_x, test_y,test_sentence_len, batchsize=batch_size, shuffle=False):
        input_batch, target_batch, seq_lens_batch= batch
        test_input = torch.LongTensor(input_batch)
        test_label = torch.LongTensor(target_batch)
        test_seq_lens_batch = torch.LongTensor(seq_lens_batch)
        test_seq_lens_batch, perm_index = test_seq_lens_batch.sort(0, descending=True)
        test_x_batch = test_input[perm_index]
        test_y_batch = test_label[perm_index]
        test_x_batch = Variable(test_x_batch)
        model.batch_size = len(test_x_batch)
        model.hidden = model.init_hidden()
        output = model(test_x_batch,test_seq_lens_batch)
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(output, Variable(test_y_batch))

        # calc testing acc
        _, predicted = torch.max(output.data, 1)
        total_acc += np.float((predicted == test_y_batch).sum().item())
        total += len(test_y_batch)
        total_loss += loss.item()
    test_loss_.append(total_loss / total)
    test_acc_.append(total_acc / total)

    # print('%s Loss for folder %s: %.3f, %s Acc: %.3f'
    #       % (mode ,str(folder),test_loss_[0], mode, test_acc_[0]))
    return test_acc_[0]

#test()
def rnn_baseline(dataset,train_model):
    avg_test_acc = 0.0
    avg_dev_acc = 0.0

    print("Run RNN Baseline for %s: ..." %(dataset))

    for i in range(10):
        texts, label_texts, labels = function.load_data("data/"+dataset + "/"+dataset+".fold-"+ str(i) +".train.txt",
                           "data/"+dataset + "/"+dataset+".fold-"+ str(i) +".validation.txt",
                           "data/"+dataset + "/"+dataset+".fold-"+ str(i) +".test.txt")

        vocab_dict, label_dict =function.get_vocab(texts,label_texts,labels)
        vocab_dict["UNK"] = len(vocab_dict)+1
        vocab_dict["PAD"] = 0
        max_length = 56
        train_x,train_y,train_sentence_len = function.dataset_preprocess(texts[0],labels[0], vocab_dict,label_dict,max_length)
        valid_x,valid_y , valid_sentence_len = function.dataset_preprocess(texts[1],labels[1], vocab_dict,label_dict,max_length)
        test_x,test_y , test_sentence_len = function.dataset_preprocess(texts[2],labels[2], vocab_dict,label_dict,max_length)
        if train_model == True:
            train(vocab_dict, label_dict, train_x, train_y, train_sentence_len, valid_x, valid_y, valid_sentence_len,dataset,i)
            model = torch.load("data/model/model_" + dataset + "_folder_" + str(i) + ".pkl")
            dev_acc = eval(i,model, valid_x,valid_y , valid_sentence_len,mode = "Dev")
            test_acc = eval(i,model, test_x, test_y, test_sentence_len,mode = "Test")
        else:
            model = torch.load("data/model/model_" + dataset + "_folder_" + str(i) + ".pkl")
            dev_acc = eval(i, model, valid_x, valid_y, valid_sentence_len, mode="Dev")
            test_acc = eval(i, model, test_x, test_y, test_sentence_len, mode = "Test")
        avg_test_acc += test_acc
        avg_dev_acc +=dev_acc
    print('Average Dev Acc for %s: %.3f'
          % (dataset, avg_dev_acc/10.0))
    print('Average Testing Acc for %s: %.3f'
          % (dataset, avg_test_acc/10.0))

import term_matching_baseline
term_matching_baseline.term_matching_baseline("AskAPatient")
term_matching_baseline.term_matching_baseline("TwADR-L")
rnn_baseline("AskAPatient",train_model=False)
rnn_baseline("TwADR-L",train_model=False)