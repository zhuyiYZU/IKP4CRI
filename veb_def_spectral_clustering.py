
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
from transformers import BertTokenizer, BertForMaskedLM
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


class Extract_and_update_verbalizer:
    def __init__(self, bert_path, data_labels,output_file):
        self.bert_path = bert_path

        self.data_labels = data_labels
        self.output_file = output_file

    def optimize_pu_ext_words(self, pu_center_words, pu_ext_words, cluster_num=1, knn_k=5):
        """
        优化扩展词集列表，包含聚类、距离矩阵计算、拉普拉斯矩阵构造等完整功能。

        :param self: 包含配置的参数类，需有 bert_path, Word2Vec_model, data_labels, output_file 属性
        :param pu_center_words: 分类列表，中心词集
        :param pu_ext_words: 扩展词集列表
        :param cluster_num: 聚类数量
        :param knn_k: KNN 参数
        :param max_words_per_cluster: 每个簇最多保留的词数
        :param visualize: 是否可视化聚类结果
        :return: 优化后的扩展词集列表
        """
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        import numpy as np
        import torch
        from transformers import BertTokenizer, BertModel
        from sklearn.cluster import KMeans
        # import matplotlib.pyplot as plt

        # 初始化 BERT tokenizer 和模型
        tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        model = BertModel.from_pretrained(self.bert_path)

        def embed_word(word):
            """获取单词的 BERT 嵌入向量"""
            encoded = tokenizer(word, return_tensors='pt', padding=True, truncation=True,max_length=512)
            input_ids = encoded['input_ids']
            with torch.no_grad():
                outputs = model(input_ids)
            word_embedding = outputs.last_hidden_state
            avg_vector = torch.mean(word_embedding, dim=1)
            return avg_vector.detach().numpy()

        def calculate_distance(x1, x2):
            """计算两个点之间的欧式距离"""
            return np.linalg.norm(x1 - x2)

        def calculate_distance_matrix(data):
            """计算距离矩阵"""
            n = len(data)
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    dist_matrix[i][j] = dist_matrix[j][i] = calculate_distance(data[i], data[j])
            return dist_matrix

        def get_adjacency_matrix(data, k):
            """构建邻接矩阵 W"""
            n = len(data)
            dist_matrix = calculate_distance_matrix(data)
            W = np.zeros((n, n))
            for idx, row in enumerate(dist_matrix):
                nearest_indices = np.argsort(row)[:k + 1]  # 取前 k 个最近邻
                W[idx, nearest_indices] = 1
            return (W + W.T) / 2

        def get_degree_matrix(W):
            """计算度矩阵 D"""
            return np.diag(W.sum(axis=1))

        def get_laplacian_matrix(D, W):
            """计算拉普拉斯矩阵 L"""
            return D - W

        def get_eigen_vectors(L, cluster_num):
            """计算特征向量"""
            eigvals, eigvecs = np.linalg.eigh(L)
            return eigvecs[:, :cluster_num]

        def cluster_and_trim_words(data, cluster_num=1):
            """基于聚类修剪扩展词集"""
            kmeans = KMeans(n_clusters=cluster_num,n_init=10)
            labels = kmeans.fit_predict(data)

            clusters = [[] for _ in range(cluster_num)]
            for idx, label in enumerate(labels):
                clusters[label].append(data[idx])

            return clusters
            # trimmed_clusters = []
            # for cluster in clusters:
            #     if len(cluster) > max_words_per_cluster:
            #         distances = euclidean_distances(cluster, [np.mean(cluster, axis=0)]).flatten()
            #         sorted_indices = np.argsort(distances)[:max_words_per_cluster]
            #         trimmed_clusters.extend([cluster[i] for i in sorted_indices])
            #     else:
            #         trimmed_clusters.extend(cluster)
            # return trimmed_clusters

        # 获取嵌入表示
        center_vectors = [embed_word(word)[0] for word in pu_center_words]
        ext_vectors = [embed_word(word)[0] for word in pu_ext_words]

        # 聚类修剪
        adjacency_matrix = get_adjacency_matrix(ext_vectors, knn_k)
        degree_matrix = get_degree_matrix(adjacency_matrix)
        laplacian_matrix = get_laplacian_matrix(degree_matrix, adjacency_matrix)
        eig_vectors = get_eigen_vectors(laplacian_matrix, cluster_num)

        # 修剪每个簇中的词
        # trimmed_vectors = cluster_and_trim_words(eig_vectors, cluster_num, max_words_per_cluster)
        cluster = cluster_and_trim_words(eig_vectors, cluster_num)

        # 筛选符合条件的扩展词
        # optimized_ext_words = []
        # for word, vector in zip(pu_ext_words, ext_vectors):
        #     if any(cosine_similarity(vector.reshape(1, -1), center.reshape(1, -1))[0][0] > 0.5 for center in
        #            center_vectors):
        #         optimized_ext_words.append(word)

        similarity_scores = []

        for word, vector in zip(pu_ext_words, ext_vectors):
            max_similarity =max(cosine_similarity(vector.reshape(1, -1),center_vectors)[0])
            similarity_scores.append((word,max_similarity))
        #按相似度降序排序
        similarity_scores.sort(key=lambda x:x[1],reverse=True)
        print(similarity_scores)
        optimized_ext_words = [word for word,_ in similarity_scores]
        # 取前50个扩展词
        # optimized_ext_words.append = [word for word, _ in similarity_scores[:50]]
        # 保存优化结果
        print(optimized_ext_words)
        print(len(optimized_ext_words))
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(','.join(optimized_ext_words))

        return optimized_ext_words


PROCESSORS = {
    'ver_pro' : Extract_and_update_verbalizer
}

#bert_path, Word2Vec_path, test_path, data_labels,output_file
if __name__ == "__main__":
    bert_path = '/home/yzu-zy/Model/bert-base-cased'
    # 加载预训练的Word2Vec模型
    ##Word2Vec_model = KeyedVectors.load_word2vec_format('/home/yzu-zy/Model/crawl-300d-2M-subword/crawl-300d-2M-subword.vec')
    # Word2Vec_model = ''
    # test_path = '/home/yzu-zy/wangye/shot_prompt/datasets/TextClassification/newstitle/train.csv'
    # test_path = '/home/yzu-zy/wangye/shot_prompt/datasets/inspired/train.csv'
    # data_labels = ['politics', 'sports', 'business', 'technology']
    print(1)
    #data_labels = ['business','entertainment','health','technology','sport','us','world']
    data_labels = ['uninterested']
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
    print(len(words))
    # words = ['uninterested', 'inapathetic', 'indifferent', 'bored', 'ambivalent', 'unsatisfied',
    #          'dismissive', 'incurious', 'blase', 'benumbed', 'dulled', 'disinterested', 'unmoved',
    #          'unconcerned']
    verbalizer_proce.optimize_pu_ext_words(data_labels,words)

