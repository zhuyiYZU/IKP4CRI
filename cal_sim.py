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
    def __init__(self, bert_path, Word2Vec_model,data_labels,output_file):
        self.bert_path = bert_path
        self.data_labels = data_labels
        self.output_file = output_file
        self.Word2Vec_model = Word2Vec_model



    # 使用标签传播算法进行分类
    def cluster_words(self, center_word, words):
        G = nx.Graph()
        words.append(center_word)
        for word in words:
            G.add_node(word, weight=np.random.rand())

        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                word1 = words[i]
                word2 = words[j]
                similarity = np.random.rand()
                G.add_edge(word1, word2, weight=similarity)

        clustering = SpectralClustering(n_clusters=1, affinity='precomputed', assign_labels='discretize')
        adjacency_matrix = np.array(nx.to_numpy_array(G, weight='weight'))
        labels = clustering.fit_predict(adjacency_matrix)

        clustered_words = {}
        for word, label in zip(words, labels):
            if label not in clustered_words:
                clustered_words[label] = []
            clustered_words[label].append((word, G.nodes[word]['weight']))

        center_cluster = None
        for cluster, words_and_similarities in clustered_words.items():
            if center_word in [word for word, _ in words_and_similarities]:
                center_cluster = cluster
                break

        if center_cluster is not None:
            center_cluster_words = clustered_words[center_cluster]
            sorted_words = sorted(center_cluster_words, key=lambda x: x[1], reverse=True)
            return sorted_words
        else:
            return []

    #这两个
    # 损失函数
    def compute_extension_word_losses(self, model, tokenizer, template, extension_words):
        # 分词化输入句子
        tokens = tokenizer.tokenize(template)
        masked_index = tokens.index("[MASK]")  # 找到[MASK]标记的位置

        extension_losses = {}

        def compute_loss(outputs, target_ids):
            # 使用logits计算损失
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, target_ids)
            return loss

        # 遍历每个扩展词汇
        for word in extension_words:
            # 复制原始句子并将[MASK]标记替换为扩展词汇
            masked_tokens = tokens.copy()
            masked_tokens[masked_index] = word

            # 将标记转换为输入ID
            input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

            # 创建输入张量
            input_tensor = torch.tensor([input_ids])

            # 创建目标张量，维度与词汇表大小相同，并将数据类型设置为Long
            target_id = tokenizer.convert_tokens_to_ids(tokens[masked_index])
            target_tensor = torch.zeros(1, len(tokenizer.get_vocab()), dtype=torch.long)
            target_tensor[0, target_id] = 1  # 将正确标签设置为1

            # 使用BERT模型计算损失
            with torch.no_grad():
                outputs = model(input_tensor)
                loss = compute_loss(outputs, target_tensor)

            # 存储扩展词汇的损失
            extension_losses[word] = loss.item()

        # 根据损失值对扩展词汇进行排序
        sorted_extension_words = sorted(extension_losses.items(), key=lambda x: x[1], reverse=True)

        return sorted_extension_words

    # mask
    def calculate_word_probabilities(self, model, tokenizer, template, extension_words):
        # 分词化模板
        tokenized_template = tokenizer.tokenize(template)

        # 找到模板中的mask位置
        mask_indices = [i for i, token in enumerate(tokenized_template) if token == '[MASK]']

        # 将模板分词转换为输入张量
        input_ids = tokenizer.convert_tokens_to_ids(tokenized_template)
        input_tensor = torch.tensor(input_ids).unsqueeze(0)  # 添加批次维度

        # 获取BERT的预测结果
        with torch.no_grad():
            predictions = model(input_tensor)

        # 获取[mask]位置上的预测概率分布
        mask_probs = predictions.logits[0, mask_indices, :]

        # 计算每个单词的概率
        word_probabilities = {}
        for word_to_fill in extension_words:
            word_id = tokenizer.convert_tokens_to_ids(word_to_fill)
            probability = torch.nn.functional.softmax(mask_probs, dim=-1)[:, word_id]
            word_probabilities[word_to_fill] = probability.item()
        sorted_extension_words = sorted(word_probabilities.items(), key=lambda x: x[1], reverse=True)

        return sorted_extension_words

    # jaccard
    def jaccard_similarity_sort(self, category_word, word_list):
        def jaccard_similarity(w1, w2):
            set1 = set(w1)
            set2 = set(w2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union != 0 else 0  # 避免除零错误

        similarity_scores = [(word, jaccard_similarity(category_word, word)) for word in word_list]

        # 按照相似度得分降序对词和得分进行排序
        sorted_words = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        return sorted_words
        #[("b", 20), ("c", 15), ("a", 10)]
        # # 打印每个词和其相似度得分
        # for word, score in sorted_words:
        #     print(f"Word: {word}, Jaccard Similarity Score: {score}")

    def find_intersection_of_top_words(self, categories_results):
        # 初始化一个列表，用于存储每个类别的前五个词
        top_5_words_by_category = []
        print(categories_results)
        # 提取每个类别的前五个词
        for category_results in categories_results:

            # print(f'debug:{category_results}')
            # 对排序结果按得分进行降序排序
            category_results.sort(key=lambda x: x[1], reverse=True)
            top_5_words = [word for word, score in category_results[:10]]

            # 存储每个类别的前五个词
            top_5_words_by_category.append(top_5_words)

        # 找到所有类别前五个词的并集并去重
        intersection_of_top_words = set(top_5_words_by_category[0]).union(*top_5_words_by_category)

        return intersection_of_top_words

    def read_existing_words_from_file(self, file_path, category_word):
        existing_words_line = None
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                words = line.strip().split(',')
                if words and words[0] == category_word:
                    existing_words_line = line.strip()
                    break
        return existing_words_line

    def calculate_similarity(self, category_word, extension_words):
        # 存储扩展词与类别标签词的相似度
        similarity_scores = {}

        # 遍历扩展词数组并计算相似度
        for word in extension_words:
            try:
                # 使用Gensim的similarity方法计算相似度
                similarity = self.Word2Vec_model.similarity(category_word, word)

                # 存储相似度分数
                similarity_scores[word] = similarity
            except KeyError:
                # 处理词不在词汇表中的情况
                similarity_scores[word] = 0  # 或其他适当的默认值

        # 按相似度分数降序排序结果
        sorted_results = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        # 提取前top_n个词
        top_words = [word for word, score in sorted_results]

        return top_words

# 谱聚类-按距离
    def optimize_pu_ext_words(self, pu_center_words, pu_ext_words, cluster_num=1, knn_k=5):
        """
        优化扩展词集列表，包含聚类、距离矩阵计算、拉普拉斯矩阵构造等完整功能。

        :param self: 包含配置的参数类，需有 bert_path, Word2Vec_model, data_labels, output_file 属性
        :param pu_center_words: 分类列表，中心词集
        :param pu_ext_words: 扩展词集列表
        :param cluster_num: 聚类数量
        :param knn_k: KNN 参数
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

        # 取前50个扩展词
        # optimized_ext_words.append = [word for word, _ in similarity_scores[:50]]
        # 保存优化结果


        return similarity_scores


# 计算标准差
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

        if isinstance(verbalizer_list, str):
            verbalizer_list = verbalizer_list.split(',')

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

        sorted_list = sorted(devs.items(), key=lambda x: x[1], reverse=True)

        # keys_to_keep = list(sorted_dict.keys()) # 保留前30个键
        # optimized_ext_words = ','.join(keys_to_keep)

        # 输出到文件
        # with open(self.output_file, 'w') as f:
        #     f.write(optimized_ext_words)

        return sorted_list

    def write_clusters_to_txt(self, words,words_in_cluster):
        # 初始化BERT模型和分词器

        model = BertForMaskedLM.from_pretrained(self.bert_path)
        tokenizer = BertTokenizer.from_pretrained(self.bert_path)


        # 读取原有的词汇数据
        existing_words_data = {}
        with open(self.output_file, 'r', encoding='utf-8') as existing_file:
            for line in existing_file:
                words = line.strip().split(',')
                if words:
                    category = words[0]
                    existing_words_data[category] = words[1:]

        with open(self.output_file, 'w', encoding='utf-8') as file:
            for custom_label in self.data_labels:
                # 找到该自定义标签下的单词


                # 标签传播
                cluster_res = self.cluster_words(custom_label, words_in_cluster)
                print(cluster_res)
                # 损失函数
                input_sentence = f"This is a word about [MASK], which is related to {custom_label}"
                bert_loss_res = self.compute_extension_word_losses(model, tokenizer, input_sentence, words_in_cluster)

                # mask预测
                bert_mask_res = self.calculate_word_probabilities(model, tokenizer, input_sentence, words_in_cluster)
                # jaccard
                word_list = self.jaccard_similarity_sort(custom_label, words_in_cluster)

                word_list_pu = self.optimize_pu_ext_words(custom_label, words_in_cluster)
                word_list_kpt = self.optimize_verbalizer(custom_label, words_in_cluster)
                # if isinstance(verbalizer_list, str):
                #     verbalizer_list = verbalizer_list.split(',')

                # print(f'bert_loss_res：{type(bert_loss_res)}')
                # print(f'bert_mask_res：{type(bert_mask_res)}')
                # print(f'cluster_res：{type(cluster_res)}')
                # print(f'word_list：{type(word_list)}')
                # print(f'word_list_kpt：{type(word_list_kpt)}')

                # categories_results = [cluster_res,bert_loss_res, bert_mask_res, cluster_res, word_list,word_list_pu,word_list_kpt]
                categories_results = [cluster_res,bert_loss_res,word_list_pu,word_list_kpt]
                # categories_results = [word_list_kpt]
                # print(len(categories_results))
                intersection_of_words = self.find_intersection_of_top_words(categories_results)

                words_in_cluster = [word for word in list(intersection_of_words) if word != custom_label]
                # print(words_in_cluster)
                # 获取已存在的词汇列表（如果存在）
                existing_words = existing_words_data.get(custom_label, [])
                # print(existing_words)
                # 合并新词汇和已存在的词汇
                final_word_list = words_in_cluster + existing_words
                cosine_result = self.calculate_similarity(custom_label, final_word_list)

                # 只保留前50个词
                final_word_list = [custom_label] + cosine_result[:50]
                print(final_word_list)
                # 将单词以逗号分隔的形式写入文件
                line = ','.join(final_word_list)
                file.write(line + '\n')
                print('写入success')

PROCESSORS = {
    'ver_pro' : Extract_and_update_verbalizer
}

#bert_path, Word2Vec_path, test_path, data_labels,output_file
if __name__ == "__main__":
    # bert_path = '/home/yzu-zy/Model/chinese-roberta-wwm-ext'
    bert_path = '/home/yzu-zy/Model/bert-base-cased'
    # 加载预训练的Word2Vec模型
    Word2Vec_model = KeyedVectors.load_word2vec_format('/home/yzu-zy/Model/crawl-300d-2M-subword/crawl-300d-2M-subword.vec',limit=100000)
    # Word2Vec_model = ''
    # test_path = '/home/yzu-zy/wangye/shot_prompt/datasets/TextClassification/newstitle/train.csv'
    # test_path = '/home/yzu-zy/wangye/shot_prompt/datasets/inspired/train.csv'
    # data_labels = ['politics', 'sports', 'business', 'technology']

    #data_labels = ['business','entertainment','health','technology','sport','us','world']
    data_labels = ['interested']
    output_file = 'expend_verbalizer_spe.txt'

    verbalizer_proce = Extract_and_update_verbalizer(bert_path, Word2Vec_model,data_labels, output_file)

    words = [
        'interested', 'concern', 'compound interest', 'usury', 'interest rate', 'involvement', 'occupy', 'fascinate', 'money', 'investment', 'vested interest', 'stake', 'curiosity', 'pursuit', 'fee', 'financial', 'loan', 'price', 'debt', 'raise', 'business', 'share', 'benefit', 'growth', 'worry', 'pastime', 'sake', 'credit', 'interestingness', 'grubstake', 'absorb', 'ownership', 'debtor', 'intrigue', 'matter to', 'interest group', 'lender', 'credit risk', 'equity', 'charisma', 'enthusiasm', 'uninteresting', 'profit', 'asset', 'hobby', 'engage', 'engross', 'color', 'principal sum', 'rate', 'future', 'speculation', 'income', 'balance', 'e', 'fund', 'entrepreneur', 'catholic church', 'money supply', 'fascination', 'attention', 'accounting', 'diversion', 'avocation', 'percentage', 'provoke', 'part', 'relate', 'portion', 'sideline', 'kindle', 'welfare', 'behalf', 'power', 'right', 'news', 'grip', 'law', 'jurisprudence', 'reversion', 'lobby', 'touch', 'colour', 'vividness', 'fire', 'recreation', 'shrillness', 'evoke', 'interesting', 'wonder', 'plural', 'elicit', 'refer', 'pertain', 'arouse', 'loanable funds', 'kangaroo', 'dowry', 'clergy', 'illegal', 'spellbind', 'topicality', 'enkindle', 'powerfulness', 'transfix', 'by-line', 'newsworthiness', 'higher', 'term', 'investor', 'investors', 'gain', 'value', 'rates', 'terms', 'time preference', 'concerns', 'expectations', 'partly', 'rise', 'market', 'increase', 'investments', 'significant', 'reflects', 'continuing', 'increasing', 'account', 'much', 'raised', 'further', 'borrowing', 'laity', 'demand', 'buying', 'current', 'domestic', 'gains', 'rising', 'moreover', 'substantial', 'raising', 'markets', 'despite', 'boost', 'bond', 'reason', 'likely', 'strong', 'economic', 'decline', 'growing', 'funds', 'its', 'benefits', 'reflected', 'corporate', 'increased', 'heresy', 'trend', 'reflect', 'giving', 'lending', 'reflecting', 'prices', 'stronger', 'spending', 'exchange', 'influence', 'focus', 'concerned', 'recent', 'currency', 'given', 'banks', 'stock', 'continue', 'considering', 'impact', 'change', 'analysts', 'returns', 'uncertainty', 'though', 'beyond', 'expect', 'deals', 'rather', 'raises', 'expense', 'suggests', 'consumer', 'opportunity cost', 'religious', 'renaissance', 'uninterested', 'rule', 'simple interest', 'fixed costs', 'fixed cost', 'fixed charge', 'touch on', 'come to', 'bear on', 'third house', 'special-interest group', 'personal magnetism', 'personal appeal', 'special interest group', 'special interest', 'spare-time activity', 'pressure group', 'lobby group', 'undivided right', 'advocacy group', 'social group', 'undivided interest', 'plural form', 'have to do with', 'terminable interest', 'security interest', 'insurable interest', 'controlling interest', 'inflation', 'paul johnson', 'love', 'middle east', 'laws of eshnunna', 'perpetuity', 'first council of nicaea', 'hate', 'annual percentage rate', 'unconcerned', 'usurer', 'ecumenical council', 'task', 'solicitous', 'thomas aquinas', 'medieval economy', 'bore', 'tire', 'desire', 'preference', 'admiration', 'affinity', 'disinterest', 'affection', 'appreciation', 'appetite', 'distaste', 'passion', 'inclination', 'respect', 'dislike', 'intention', 'dissatisfaction', 'sympathy', 'antipathy', 'keenness', 'piques', 'commitment', 'participation', 'royalty', 'fondness', 'excitement', 'aversion', 'devotion', 'disdain', 'opinion', 'confidence', 'eagerness', 'allegiance', 'expectation', 'ire', 'animosity', 'displeasure', 'ardor', 'shareholding', 'skepticism', 'satisfaction', 'unhappiness', 'consternation', 'dividends', 'belief', 'relevance', 'potential', 'significance', 'outrage', 'esteem', 'disenchantment', 'allure', 'infatuation', 'hostility', 'resentment', 'disquiet', 'antagonism', 'unease', 'sentiment', 'fandom', 'willingness', 'deposit', 'goodwill', 'conflict', 'support', 'adoration', 'wariness', 'consideration', 'disapproval', 'acquaintanceship', 'optimism', 'urgency', 'abhorrence', 'incentive', 'moneylender', 'muslim world', 'curio', 'concernment', 'unattractive', 'rainmaker', 'contractum trinius', 'game', 'rapt', 'solicitude', 'businesslike', 'grue', 'involve', 'expenditure', 'banque de france', 'carefree', 'extortion', 'glamour', 'enthral', 'islamic banking and finance', 'monochromatic', 'appropriation', 'inquisitive', 'teal', 'acculturation', 'planchette', 'vest interest', 'seigniorage', 'bankrupt', 'curious', 'fundraiser', 'cost of capital', 'trepidation', 'pamper', 'absorption', 'heedful', 'tinge', 'captivate', 'theft', 'bicolor', 'imbue', 'complexion', 'chartreuse', 'thrift', 'colorist', 'fiscal', 'free market', 'curiousness', 'lavender', 'inquisitiveness', 'mauve', 'bribe', 'expend', 'enterprise', 'banc', 'default', 'hobgoblin', 'carefulness', 'pallor', 'supply and demand', 'self interest', 'uninterestingness', 'interestedness', 'intension', 'dovishness', 'bullishness', 'acedia', 'legal', 'preoccupy', 'school of salamanca', 'colorphobia', 'martín de azpilcueta', 'legal interest', 'prepossess', 'enthrall', 'anne-robert-jacques turgot', 'baron de laune', 'scaremonger', 'misgive', 'rights', 'theory of fructification', 'profiteer', 'theorica', 'uncolorable', 'viridescent', 'purply', 'gluon', 'verdantly', 'coloristic', 'blee', 'colorize', 'bankroll', 'incarnadine', 'colorability', 'chromakey', 'colorous', 'colormap', 'rate of return', 'absorbable', 'uncolored', 'bluebill', 'polychromatic', 'fedzilla', 'recolor', 'colorization', 'colorism', 'argaman', 'trichromatic', 'pinchpenny', 'worrisome', 'tinct', 'antigreen', 'unworryingly', 'adam smith', 'carl menger', 'mortgage', 'liquidity', 'security', 'spreadsheet', 'property', 'creditworthiness', 'frédéric bastiat', 'knut wicksell', 'bertil ohlin', 'dennis robertson', 'save account', 'premium bond', 'irving fisher', 'affinity card', 'bank rate', 'john maynard keynes', 'robber baron', 'unearned income', 'finance and economy', 'future interest', 'dutch auction', 'guaranteed investment certificate', 'mortgage lender', 'united states', 'get inform', 'pay one due', 'see something new', 'zero coupon bond', 'chamber of commerce', 'time value of money', 'suck up', 'dead pledge', 'watch it', 'key money', 'post obit', 'operate expense', 'venture capital', 'banker lien', 'glassy eye', 'real versus nominal value', 'be curious', 'you get bore', 'turn on tv', 'fee simple', 'nervous nellie', 'you learn something', 'change channel', 'promissory note', 'learn news', 'buy magazine', 'mutual fund', 'disposable income', 'pay for', 'become inform', 'switch on tv', 'big business', 'open newspaper', 'turn on television', 'have eye', 'some money', 'flesh color', 'bank account', 'skin sign', 'open eye', 'prepare for exam', 'liquid asset', 'social network', 'red blue', 'cream color', 'olive color', 'straw color', 'turn tv on', 'it be funny', 'you be entertain', 'colour television', 'penny pinch', 'get knowledge', 'colour in', 'sit on couch', 'credit transfer', 'peace dividend', 'ability to see', 'hot money', 'rose color', 'medium of exchange', 'heather mixture', 'control person', 'deep pocket', 'roth ira', 'e cash', 'see particular program', 'sole proprietorship', 'color charge', 'pastel color', 'gain knowledge', 'merchant bank', 'risk premium', 'mortgage loan', 'credit rating agency', 'credit score', 'credit bureau', 'real interest rate', 'government debt', 'central bank', 'monetary policy', 'excess reserves', 'scale invariant', 'jacob bernoulli', 'quantity theory of money', 'darrell duffie', 'jarrow-turnbull model', 'geometric series', 'federal reserve bank of new york', 'open market operations', 'federal funds', 'federal funds rate', 'federal reserve', 'scale factor'
    ]


    verbalizer_proce.write_clusters_to_txt(data_labels,words)





