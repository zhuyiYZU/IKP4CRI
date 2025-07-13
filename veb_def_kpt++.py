# kpt++
# -*- coding:utf-8 -*-
import pandas as pd
from nltk import word_tokenize
from nltk import pos_tag
import nltk
import numpy as np
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
import spacy
import csv
import torch
from transformers import BertTokenizer, BertForMaskedLM,BertModel
from nltk.corpus import stopwords
from textblob import TextBlob
import time
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from gensim.models import KeyedVectors
import networkx as nx
from sklearn.cluster import SpectralClustering
# 聚类
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import statistics


class Extract_and_update_verbalizer:
    def __init__(self, bert_path, data_labels,output_file):
        self.bert_path = bert_path
        self.data_labels = data_labels
        self.output_file = output_file


    def optimize_verbalizer(self, kpt_center_words, kpt_ext_words):

        # 初始化 BERT tokenizer 和模型
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.model = BertModel.from_pretrained(self.bert_path)

        # 将类别词列表转换为字符串
        # label0 = self.data_labels[0]
        # label1 = self.data_labels[1]
        label0 = str(kpt_center_words[0])
        label1 = str(kpt_center_words[1])
        # print(label0)
        # print(label1)

        # 将扩展词集列表转换为字符串
        verbalizer = ','.join(kpt_ext_words)
        verbalizer = str(verbalizer)
        # print(verbalizer)
        verbalizer_list = verbalizer.split(',')

        devs = {}
        for word in verbalizer_list:
            # 使用 tokenizer 对两个字符串进行编码
            encoded_word = self.tokenizer(word, return_tensors='pt', padding=True, truncation=True)
            encoded_label0 = self.tokenizer(label0, return_tensors='pt', padding=True, truncation=True)
            encoded_label1 = self.tokenizer(label1, return_tensors='pt', padding=True, truncation=True)

            # 获取编码后的输入 ID
            input_ids_word = encoded_word['input_ids']
            input_ids_label0 = encoded_label0['input_ids']
            input_ids_label1 = encoded_label1['input_ids']

            # 使用模型获取词向量表示
            with torch.no_grad():
                outputs_word = self.model(input_ids_word)
                outputs_label0 = self.model(input_ids_label0)
                outputs_label1 = self.model(input_ids_label1)

            # 获取词向量表示的最后一层 hidden states
            word_embeddings_word = outputs_word.last_hidden_state
            word_embeddings_label0 = outputs_label0.last_hidden_state
            word_embeddings_label1 = outputs_label1.last_hidden_state

            # 计算平均向量
            average_vector_word = torch.mean(word_embeddings_word, dim=1)
            average_vector_label0 = torch.mean(word_embeddings_label0, dim=1)
            average_vector_label1 = torch.mean(word_embeddings_label1, dim=1)

            # 将 PyTorch 张量转换为 NumPy 数组
            average_vector_word = average_vector_word.detach().numpy()
            average_vector_label0 = average_vector_label0.detach().numpy()
            average_vector_label1 = average_vector_label1.detach().numpy()

            cos0 = cosine_similarity(average_vector_word, average_vector_label0)
            cos1 = cosine_similarity(average_vector_word, average_vector_label1)
            data = [float(cos0), float(cos1)]  # 两个数作为数据集的一部分
            std_dev = statistics.stdev(data)
            devs[word] = std_dev

        sorted_dict = dict(sorted(devs.items(), key=lambda x: x[1], reverse=True))

        keys_to_keep = list(sorted_dict.keys())[:30]  # 保留前30个键
        optimized_ext_words = ','.join(keys_to_keep)

        # 输出到文件
        with open(self.output_file, 'w') as f:
            f.write(optimized_ext_words)

        return optimized_ext_words


PROCESSORS = {
    'ver_pro' : Extract_and_update_verbalizer
}

#bert_path, Word2Vec_path, test_path, data_labels,output_file
if __name__ == "__main__":
    bert_path = '/home/yzu-zy/Model/bert-base-cased'
    # 加载预训练的Word2Vec模型
    #Word2Vec_model = KeyedVectors.load_word2vec_format('/home/yzu-zy/Model/crawl-300d-2M-subword/crawl-300d-2M-subword.vec')
    # Word2Vec_model = ''
    # test_path = '/home/yzu-zy/wangye/shot_prompt/datasets/TextClassification/newstitle/train.csv'
    # test_path = '/home/yzu-zy/wangye/shot_prompt/datasets/inspired/train.csv'
    # data_labels = ['politics', 'sports', 'business', 'technology']

    #data_labels = ['business','entertainment','health','technology','sport','us','world']
    data_labels = ['uninterested', 'interested']
    output_file = 'expend_verbalizer_spe.txt'

    verbalizer_proce = Extract_and_update_verbalizer(bert_path, data_labels, output_file)

    words = ['uninterested', 'inapathetic', 'indifferent', 'bored', 'ambivalent', 'unsatisfied',
    'dismissive', 'incurious', 'blase', 'benumbed', 'dulled', 'disinterested', 'unmoved',
    'unconcerned', 'oblivious', 'distrustful', 'reticent', 'infatuated', 'unimpressed',
    'befuddled', 'uninformed', 'unfazed', 'unsympathetic', 'ignorant', 'uninteresting',
    'unfocused', 'unperturbed', 'unimportant', 'uninvolved', 'clueless', 'timid',
    'unconvinced', 'uncommunicative', 'unappealing', 'underwhelmed', 'interested',
    'disagreeable', 'unmotivated', 'inattentive', 'irrelevant', 'fixated', 'impervious',
    'callous', 'enamored']

    verbalizer_proce.optimize_verbalizer(data_labels,words)
