import numpy as np
import torch
from gru_entitiylibrary import LSTMClassifier
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch_preprocess as function
import math
import read_files as read
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('data/rnn_entity.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

logger.info('##############################Start Training#################################')

np.random.seed(123)
torch.manual_seed(123)

use_gpu = torch.cuda.is_available()



def iterate_minibatches(inputs, targets, sentence_len,batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = indices[start_idx:start_idx + batchsize]
        sentence_len_batch = sentence_len[excerpt]
        max_len = max(sentence_len_batch)
        input_batch = np.zeros((batchsize,max_len))
        for i,text_id in enumerate(excerpt):
            input_batch[i][:len(inputs[text_id])] = inputs[text_id]
        yield input_batch, targets[excerpt],sentence_len_batch

def padding(inputs,seq_len):
    max_len = max(seq_len)
    input_batch = np.zeros((len(inputs), max_len))
    for i, text_id in enumerate(inputs):
        input_batch[i][:len(inputs[i])] = inputs[i]
    return input_batch

    ### parameter setting

### training procedure
def train(vocab_dict, label_dict, train_x,train_y,label_x, label_seq,train_sentence_len,valid_x,valid_y , valid_sentence_len, dataset,folder):
    embedding_dim = 512
    hidden_dim = 256
    epochs = 200
    batch_size = 64
    learning_rate = 1.0

    model = LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                           vocab_size=len(vocab_dict), label_size=len(label_dict), batch_size=batch_size)
    if use_gpu:
        model = model.cuda()
    train_acc_ = []
    valid_acc_ = []
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    valid_loss_ = []
    val_acc = -math.inf
    label_x = padding(label_x, label_seq)
    label_input = torch.LongTensor(label_x)
    label_seq_input = torch.LongTensor(label_seq)
    train_label = Variable(label_input)
    if use_gpu:
        train_label = Variable(label_input).cuda()
        label_seq_input = label_seq_input.cuda()
    for epoch in range(epochs):
        ## training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        logger.info('Starting Epoch: ' + str(epoch))
        batch_count = 0
        for batch in iterate_minibatches(train_x, train_y,train_sentence_len, batchsize=batch_size, shuffle=True):

            input_batch, target_batch, seq_lens_batch = batch
            train_input_batch = torch.LongTensor(input_batch)
            train_target_batch = torch.LongTensor(target_batch)
            train_seq_lens_batch = torch.LongTensor(seq_lens_batch)
            train_seq_lens_batch, perm_index = train_seq_lens_batch.sort(0, descending=True)
            if use_gpu:
                train_x_batch = Variable(train_input_batch[perm_index]).cuda()
                train_y_batch = Variable(train_target_batch[perm_index]).cuda()
                train_seq_lens_batch = train_seq_lens_batch.cuda()
            else:
                train_x_batch = Variable(train_input_batch[perm_index])
                train_y_batch = Variable(train_target_batch[perm_index])

            model.zero_grad()
            model.batch_size = len(train_x_batch)
            output = model(train_x_batch,train_seq_lens_batch,train_label,label_seq_input)
            loss = loss_function(output, train_y_batch)
            loss.backward()
            optimizer.step()


            batch_count += 1
            # calc training acc
            _, predicted = torch.max(output.data, 1)
            batch_acc = np.float((predicted == train_y_batch).sum().item())
            #print("Batch "+ str(batch_count) +" Loss & Acc: " + str(loss.data.cpu().numpy()) + " " +str(batch_acc))
            #logger.info("Batch "+ str(batch_count) +" Loss & Acc: " + str(loss.data.cpu().numpy()) + " " +str(batch_acc))
            total_acc += batch_acc
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
            if use_gpu:
                valid_x_batch = Variable(valid_input[perm_index]).cuda()
                valid_y_batch = Variable(valid_label[perm_index]).cuda()
                valid_seq_lens_batch = valid_seq_lens_batch.cuda()
            else:
                valid_x_batch = Variable(valid_input[perm_index])
                valid_y_batch = Variable(valid_label[perm_index])
                train_label = Variable(label_input)
            model.batch_size = len(valid_x_batch)
            output = model(valid_x_batch,valid_seq_lens_batch, train_label, label_seq_input)
            loss = loss_function(output, valid_y_batch)
            # calc testing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += np.float((predicted == valid_y_batch).sum().item())
            total += len(valid_y_batch)
            total_loss += loss.item()
        if (total_acc/total > val_acc):
            torch.save(model, "data/model_entity/model_" + dataset + "_folder_" + str(folder) + ".pkl")
            val_acc = total_acc/total
        valid_loss_.append(total_loss / total)
        valid_acc_.append(total_acc / total)
        #print(
        #    '[Epoch: %3d/%3d] Training Loss: %.3f, Validating Loss:  %.3f, Training Acc: %.3f, Validing Acc: %.3f'
        #    % (epoch, epochs, train_loss_[epoch], valid_loss_[epoch],  train_acc_[epoch],
        #       valid_acc_[epoch]))
        logger.info('[Epoch: %3d/%3d] Training Loss: %.3f, Validating Loss:  %.3f, Training Acc: %.3f, Validing Acc: %.3f'
            % (epoch, epochs, train_loss_[epoch], valid_loss_[epoch],  train_acc_[epoch],
               valid_acc_[epoch]))



#train()

def eval(folder, model,test_x,test_y, test_sentence_len, label_x, label_seq,mode):
    ## testing epoch
    test_loss_ = []
    test_acc_ = []
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    label_x = padding(label_x, label_seq)
    label_input = torch.LongTensor(label_x)
    label_seq_input = torch.LongTensor(label_seq)
    train_label = Variable(label_input)
    if use_gpu:
        train_label = Variable(label_input).cuda()
        label_seq_input = label_seq_input.cuda()
    batch_size = 64
    for batch in iterate_minibatches(test_x, test_y,test_sentence_len, batchsize=batch_size, shuffle=False):
        input_batch, target_batch, seq_lens_batch= batch
        test_input = torch.LongTensor(input_batch)
        test_label = torch.LongTensor(target_batch)
        test_seq_lens_batch = torch.LongTensor(seq_lens_batch)
        test_seq_lens_batch, perm_index = test_seq_lens_batch.sort(0, descending=True)
        if use_gpu:
            test_x_batch = Variable(test_input[perm_index]).cuda()
            test_y_batch = Variable(test_label[perm_index]).cuda()
            test_seq_lens_batch = test_seq_lens_batch.cuda()
        else:
            test_x_batch = Variable(test_input[perm_index])
            test_y_batch = Variable(test_label[perm_index])
        model.batch_size = len(test_x_batch)
        #model.hidden = model.init_hidden()
        output = model(test_x_batch,test_seq_lens_batch,train_label,label_seq_input)
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(output,test_y_batch)
        # calc testing acc
        _, predicted = torch.max(output.data, 1)
        total_acc += np.float((predicted == test_y_batch).sum().item())
        total += len(test_y_batch)
        total_loss += loss.item()
    test_loss_.append(total_loss / total)
    test_acc_.append(total_acc / total)

    #print('%s Loss for folder %s: %.3f, %s Acc: %.3f'
    #      % (mode ,str(folder),test_loss_[0], mode, test_acc_[0]))
    logger.info('%s Loss for folder %s: %.3f, %s Acc: %.3f'
          % (mode ,str(folder),test_loss_[0], mode, test_acc_[0]))
    return test_acc_[0]

#test()
def rnn_character_entity(dataset,train_model):
    avg_test_acc = 0.0
    avg_dev_acc = 0.0

    print("Run RNN entity library for %s: ..." %(dataset))
    logger.info("Run RNN entity library for %s: ..." %(dataset))
    #lm_f = LanguageModel.load_language_model('/home/dongfang/.flair/embeddings/lm-news-english-forward-v0.2rc.pt')
    #dictionary = lm_f.dictionary
    #vocab_dict = dict()
    #for i, j in dictionary.item2idx.items():
    #    vocab_dict[i.decode('utf-8')] = j
    #vocab_dict["UNK"] = len(vocab_dict) + 1
    #read.savein_json("data/config/char2int",vocab_dict)
    # texts, label_texts, labels = function.load_data(
    #     "data/" + dataset + "/" + dataset + ".fold-" + str(0) + ".train.txt",
    #     "data/" + dataset + "/" + dataset + ".fold-" + str(0) + ".validation.txt",
    #     "data/" + dataset + "/" + dataset + ".fold-" + str(0) + ".test.txt")
    # _, label_dict, label_texts_dict = function.get_vocab(texts, label_texts, labels)
    # read.savein_json("data/config/label_dict_"+dataset,label_dict)
    # read.savein_json("data/config/label_texts_dict_"+dataset,label_texts_dict)

    vocab_dict = read.readfrom_json("data/config/char2int")
    label_dict = read.readfrom_json("data/config/label_dict_"+dataset)
    label_texts_dict = read.readfrom_json("data/config/label_texts_dict_"+dataset)
    label_input, label_input_seq= function.label_preprocess_character(label_dict,label_texts_dict,vocab_dict)
    folder = 0
    for i in range(folder, folder+1):
        texts, label_texts, labels = function.load_data("data/"+dataset + "/"+dataset+".fold-"+ str(i) +".train.txt",
                           "data/"+dataset + "/"+dataset+".fold-"+ str(i) +".validation.txt",
                           "data/"+dataset + "/"+dataset+".fold-"+ str(i) +".test.txt")

        train_x,train_y,train_sentence_len = function.dataset_preprocess_character(texts[0],labels[0], vocab_dict,label_dict)
        valid_x,valid_y , valid_sentence_len = function.dataset_preprocess_character(texts[1],labels[1], vocab_dict,label_dict)
        test_x,test_y , test_sentence_len = function.dataset_preprocess_character(texts[2],labels[2], vocab_dict,label_dict)
        if train_model == True:

            print("#####################   Performance on folder " + str(i) + "  ######################################")
            logger.info("#####################   Performance on folder " + str(i) + "  ######################################")
            train(vocab_dict, label_dict, train_x, train_y, label_input, label_input_seq, train_sentence_len, valid_x, valid_y, valid_sentence_len,dataset,i)
            model = torch.load("data/model_entity/model_" + dataset + "_folder_" + str(i) + ".pkl")
            dev_acc = eval(i,model, valid_x,valid_y , valid_sentence_len, label_input, label_input_seq,mode = "Dev")
            test_acc = eval(i,model, test_x, test_y, test_sentence_len,label_input, label_input_seq, mode = "Test")
        else:
            model = torch.load("data/model_entity/model_" + dataset + "_folder_" + str(i) + ".pkl")
            dev_acc = eval(i, model, valid_x, valid_y, valid_sentence_len, label_input, label_input_seq, mode="Dev")
            test_acc = eval(i, model, test_x, test_y, test_sentence_len, label_input, label_input_seq, mode = "Test")
        avg_test_acc += test_acc
        avg_dev_acc +=dev_acc
    print('Average Dev Acc for %s: %.3f'% (dataset, avg_dev_acc/1.0))
    print('Average Testing Acc for %s: %.3f'% (dataset, avg_test_acc/1.0))
    logger.info('Average Dev Acc for %s: %.3f'% (dataset, avg_dev_acc/1.0))
    logger.info('Average Testing Acc for %s: %.3f'% (dataset, avg_test_acc/1.0))
# import term_matching_baseline
# from flair.embeddings import CharLMEmbeddings
# from flair.data import Sentence



# rnn_character_entity("AskAPatient",train_model=True)
# rnn_character_entity("TwADR-L",train_model=True)
