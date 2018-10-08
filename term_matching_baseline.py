from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk
from gensim.summarization import bm25
nltk.download('punkt')
stemmer = SnowballStemmer("english")


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
    x = label_texts[0]  #+ texts[1]
    label_texts = label_texts[0] #+label_texts[1]
    y = labels[0] # + labels[1]
    label_y_dict = {}
    label_x_dict = {}
    for index, label_text in enumerate(label_texts):
        if label_text not in label_y_dict:
            label_y_dict[label_text] = y[index]
            label_x_dict[label_text] = [x[index]]
        else:
            label_x_dict[label_text] +=[x[index]]

    label_texts_new = list(label_y_dict.keys())
    label_text_new_index = {index:label_text for index,label_text in enumerate(label_texts_new)}
    vocab = list()

    for label, x_texts in label_x_dict.items():
        text_items = x_texts+[label]
        for text_item in text_items:
            for label_token in word_tokenize(text_item):
                label_token_stem = stemmer.stem(label_token.lower())
                if label_token_stem not in vocab:
                    vocab.append(label_token_stem)

    vocab_dict = {token_stem: index for index, token_stem in enumerate(vocab)}
    return vocab_dict, label_x_dict,label_y_dict, label_text_new_index,label_texts_new

def get_scores(tf_idf_tests,tf_idf_label,label_text_index,label_code_dict,y_test_text):
    # print(np.max(np.dot(tf_idf_tests,tf_idf_label.T)))
    labels_index = np.argmax(np.dot(tf_idf_tests,tf_idf_label.T),axis=1)
    codes = [label_code_dict[label_text_index[label_index]] for label_index in labels_index]
    #print(codes)
    #print(y_test_text)
    acc = 0.0
    for index , code in enumerate(codes):
        if code == y_test_text[index]:
            acc+=1.0
    score = acc / len(y_test_text)
    return score

def get_bm_25_score(query_list, bm25_list, label_text_new_index, label_y_dict, y_test_text):

    bm25Model = bm25.BM25(bm25_list)
    average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
    codes = []
    for query in query_list:
        scores = bm25Model.get_scores(query, average_idf)
        idx = scores.index(max(scores))
        codes.append(label_y_dict[label_text_new_index[idx]])
    acc = 0.0
    for index , code in enumerate(codes):
        if code == y_test_text[index]:
            acc+=1.0
    score = acc / len(y_test_text)
    return score


def process(train_path, dev_path, test_path):
    x_text, label_text,  y_text = load_data(train_path,dev_path,test_path)
    vocab_dict, label_x_dict, label_y_dict, label_text_new_index, label_texts_new = get_vocab(x_text,label_text,y_text)
    text_matrix = np.zeros((len(label_texts_new)+len(x_text[1])+len(x_text[2]),len(vocab_dict)))
    index = 0
    bm25_list = []
    query_dev_list = []
    query_test_list = []
    for label_text in label_texts_new:
        texts = label_x_dict[label_text]
        texts.append(label_text)
        word_ids = np.zeros(len(vocab_dict))
        for text in texts:
            token_stem_list = []
            for _, token in enumerate(word_tokenize(text)):
                token_norm = stemmer.stem(token.lower())
                word_ids[vocab_dict[token_norm]]+=1.0
                token_stem_list.append(token_norm)
        text_matrix[index] = word_ids
        bm25_list.append(token_stem_list)
        index+=1
    for dev_text in x_text[1]:
        word_ids = np.zeros(len(vocab_dict))
        token_stem_list = []
        for _, token in enumerate(word_tokenize(dev_text)):
            token_norm = stemmer.stem(token.lower())
            if token_norm in vocab_dict:
                word_ids[vocab_dict[token_norm]] += 1.0
            token_stem_list.append(token_norm)
        text_matrix[index] = word_ids
        query_dev_list.append(token_stem_list)
        index += 1
    for test_text in x_text[2]:
        word_ids = np.zeros(len(vocab_dict))
        token_stem_list = []
        for _, token in enumerate(word_tokenize(test_text)):
            token_norm = stemmer.stem(token.lower())
            if token_norm in vocab_dict:
                word_ids[vocab_dict[token_norm]] += 1.0
            token_stem_list.append(token_norm)
        text_matrix[index] = word_ids
        query_test_list.append(token_stem_list)
        index += 1
    transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf = transformer.fit_transform(text_matrix)
    tf_idf_matrix = tfidf.toarray()
    tf_idf_label = tf_idf_matrix[:len(label_texts_new)]
    tf_idf_dev = tf_idf_matrix[len(label_texts_new):len(label_texts_new)+len(x_text[1])]
    tf_idf_test = tf_idf_matrix[len(label_texts_new)+len(x_text[1]):]

    return get_scores(tf_idf_dev,tf_idf_label,label_text_new_index,label_y_dict,y_text[1]),\
           get_scores(tf_idf_test,tf_idf_label,label_text_new_index,label_y_dict,y_text[2]), \
           get_bm_25_score(query_dev_list, bm25_list, label_text_new_index, label_y_dict, y_text[1]),\
           get_bm_25_score(query_test_list,bm25_list, label_text_new_index, label_y_dict, y_text[2])


def term_matching_baseline(dataset):
    scores_tf_idf_dev = 0.0
    scores_bm25_dev = 0.0
    scores_tf_idf_test = 0.0
    scores_bm25_test = 0.0
    print("Run Term Matching Baseline for %s: ..." %(dataset))

    for i in range(10):
        score_tf_idf_dev, score_tf_idf_test, score_bm25_dev, score_bm25_test = process("data/"+dataset + "/"+dataset+".fold-"+ str(i) +".train.txt",
                           "data/"+dataset + "/"+dataset+".fold-"+ str(i) +".validation.txt",
                           "data/"+dataset + "/"+dataset+".fold-"+ str(i) +".test.txt")
        #print("Folder "+str(i) +" Dev accuracy: TF-IDF  ", score_tf_idf_dev)
        #print("Folder " + str(i) + " Dev accuracy: BM25  ", score_bm25_dev)
        #print("Folder "+str(i) +" Test accuracy: TF-IDF  ", score_tf_idf_test)
        #print("Folder " + str(i) + " Test accuracy: BM25  ", score_bm25_test)

        scores_tf_idf_dev+=score_tf_idf_dev
        scores_bm25_dev +=score_bm25_dev
        scores_tf_idf_test +=score_tf_idf_test
        scores_bm25_test +=score_bm25_test

    print("Average Dev accuracy for %s: TF-IDF %s " %(dataset, scores_tf_idf_dev/10.0))
    print("Average Dev accuracy for %s: BM25 %s" %(dataset, scores_bm25_dev/10.0))
    print("Average Test accuracy for %s: TF-IDF %s " %(dataset, scores_tf_idf_test/10.0))
    print("Average Test accuracy for %s: BM25 %s" %(dataset, scores_bm25_test/10.0))

# term_matching_baseline("AskAPatient")