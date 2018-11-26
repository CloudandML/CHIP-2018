# -*- coding utf-8 -*-
"""
https://www.kaggle.com/c/quora-question-pairs/kernels  计算句子对
https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb   kernels
"""
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from gensim import corpora, models, similarities
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import KeyedVectors
import os

df_train = pd.read_csv('./data/train_prepare.csv', encoding='utf-8')
df_test = pd.read_csv('./data/test_prepare.csv', encoding='utf-8')
# df_train = pd.read_csv('./data/train_prepare_1w.csv', encoding='utf-8')#.iloc[:100]
# df_test = pd.read_csv('./data/train_prepare_h1w.csv', encoding='utf-8')#.iloc[:100]

with open('./data/word_embedding.json', encoding='utf-8') as f:
    words_em = json.load(f)
with open('./data/char_embedding.json', encoding='utf-8') as f:
    chars_em = json.load(f)


def get_weight(count, eps=10000, min_count=1):
    """
    # If a word appears only once, we ignore it completely (likely a typo)
    # Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

    :param count:
    :param eps:
    :param min_count:
    :return:
    """
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


# eps = 5000
train_qw = pd.Series(df_train['wid1'].tolist() + df_train['wid2'].tolist()).astype(str)
words = (" ".join(train_qw)).split()
counts = Counter(words)
word_weights = {word: get_weight(count) for word, count in counts.items()}

train_qc = pd.Series(df_train['cid1'].tolist() + df_train['cid2'].tolist()).astype(str)
chars = (" ".join(train_qc)).split()
char_counts = Counter(chars)
char_weights = {char: get_weight(count) for char, count in char_counts.items()}


# print('Most common words and weights: \n')
# print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
# print('\nLeast common words and weights: ')
# print(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])


def get_embed_weight(embedding, word):
    return np.sum(embedding[word])


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['wid1']).split():
        q1words[word] = 1
    for word in str(row['wid2']).split():
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


def char_match_share(row):
    q1chars = {}
    q2chars = {}
    for char in str(row['cid1']).split():
        q1chars[char] = 1
    for char in str(row['cid2']).split():
        q2chars[char] = 1
    if len(q1chars) == 0 or len(q2chars) == 0:
        return 0
    shared_chars_in_q1 = [w for w in q1chars.keys() if w in q2chars]
    shared_chars_in_q2 = [w for w in q2chars.keys() if w in q1chars]
    R = (len(shared_chars_in_q1) + len(shared_chars_in_q2)) / (len(q1chars) + len(q2chars))
    return R


def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['wid1']).split():
        q1words[word] = 1
    for word in str(row['wid2']).split():
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [word_weights.get(w, 0) for w in q1words.keys() if w in q2words] + [word_weights.get(w, 0) for w in
                                                                                         q2words.keys() if w in q1words]
    total_weights = [word_weights.get(w, 0) for w in q1words] + [word_weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def tfidf_char_match_share(row):
    q1chars = {}
    q2chars = {}
    for char in str(row['cid1']).split():
        q1chars[char] = 1
    for char in str(row['cid2']).split():
        q2chars[char] = 1
    if len(q1chars) == 0 or len(q2chars) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopchars
        return 0

    shared_weights = [char_weights.get(w, 0) for w in q1chars.keys() if w in q2chars] + [char_weights.get(w, 0) for w in
                                                                                         q2chars.keys() if w in q1chars]
    total_weights = [char_weights.get(w, 0) for w in q1chars] + [char_weights.get(w, 0) for w in q2chars]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def get_length(words):
    return len(words.split())


def get_word_embedding_mean(sentence):
    words = sentence.split()
    words_mean_list = []
    for word in words:
        # print('word: ', word, type(word), words_em[word])
        if words_em.get(word, 0):
            words_mean_list.append(np.sum(words_em[word]))
        else:
            words_mean_list.append(0)
    words_mean = np.mean(words_mean_list)
    return words_mean


def get_char_embedding_mean(sentence):
    chars = sentence.split()
    chars_mean_list = []
    for char in chars:
        if chars_em.get(char, 0):
            chars_mean_list.append(np.sum(chars_em[char]))
        else:
            chars_mean_list.append(0)
    chars_mean = np.mean(chars_mean_list)
    return chars_mean


def tfidf_word_embed_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['wid1']).split():
        q1words[word] = 1
    for word in str(row['wid2']).split():
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [np.sum(words_em.get(w, 0)) for w in q1words.keys() if w in q2words] + [np.sum(words_em.get(w, 0))
                                                                                             for w in
                                                                                             q2words.keys() if
                                                                                             w in q1words]
    total_weights = [np.sum(words_em.get(w, 0)) for w in q1words] + [np.sum(words_em.get(w, 0)) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def tfidf_char_embed_match_share(row):
    q1chars = {}
    q2chars = {}
    for char in str(row['cid1']).split():
        q1chars[char] = 1
    for char in str(row['cid2']).split():
        q2chars[char] = 1
    if len(q1chars) == 0 or len(q2chars) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopchars
        return 0

    shared_weights = [np.sum(chars_em.get(w, 0)) for w in q1chars.keys() if w in q2chars] + [np.sum(chars_em.get(w, 0))
                                                                                             for w in
                                                                                             q2chars.keys() if
                                                                                             w in q1chars]
    total_weights = [np.sum(chars_em.get(w, 0)) for w in q1chars] + [np.sum(chars_em.get(w, 0)) for w in q2chars]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


# 传统句子相似度计算过程
# 创建词袋，并对应词袋字词的index
train_qw = pd.Series(
    df_train['wid1'].tolist() + df_train['wid2'].tolist() + df_test['wid1'].tolist() + df_test['wid2'].tolist()).astype(
    str)
qids = pd.Series(
    df_train['qid1'].tolist() + df_train['qid2'].tolist() + df_test['qid1'].tolist() + df_test['qid2'].tolist()).astype(
    str)
counts = CountVectorizer()
word_bag = counts.fit_transform(train_qw).toarray()
bag_word_index_dict = counts.vocabulary_  # 如 {'W100840': 29} 即词对应bag中的第29个词

train_qc = pd.Series(
    df_train['cid1'].tolist() + df_train['cid2'].tolist() + df_test['cid1'].tolist() + df_test['cid2'].tolist()).astype(
    str)
counts = CountVectorizer()
char_bag = counts.fit_transform(train_qc).toarray()
bag_char_index_dict = counts.vocabulary_  # 如 {'W100840': 29} 即词对应bag中的第29个词

# print('31 bag_word_index_dict[W100914]: ', bag_word_index_dict)
# sklearn TFIDF 创建TFIDF
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
tfidf_weights = tfidf.fit_transform(word_bag).toarray()
char_tfidf_weights = tfidf.fit_transform(char_bag).toarray()
# 将question 用tfidf向量表示
q2vec = {}
for qid, vec in zip(qids, word_bag):
    # print('38 \t', qid, vec)
    if qid not in q2vec.keys():
        q2vec[qid] = vec
    # else:
    #     print('重复')
q2vec_char = {}
for qid, vec in zip(qids, char_bag):
    # print('38 \t', qid, vec)
    if qid not in q2vec_char.keys():
        q2vec_char[qid] = vec


def get_topic_model_topic(corpus_model, qids):
    qw_model = {}
    for doc, qid in zip(corpus_model, qids):
        # print('238 doc: ', doc)
        values = [d[1] for d in doc]
        if values:
            max_idx = np.array(values).argmax()
            topic = doc[max_idx][0]
        else:
            topic = np.random.randint(0, 300)  # 如果values为空则随机赋值
        qw_model[qid] = topic
    return qw_model


def get_topic_sim(sims_model):
    n, m = len(df_train), len(df_test)
    train_qw_topic_sim, test_qw_topic_sim = [], []
    for i, sim in enumerate(sims_model[:n]):
        if i >= n:
            break
        else:
            # print('i sim_i: ',i, i+n)
            train_qw_topic_sim.append(sim[i + n])
    for i, sim in enumerate(sims_model[2 * n:2 * n + m]):
        if i >= 2 * n + m:
            break
        else:
            # print('test i sim_i: ', i+2*n,i+2 * n + m)
            test_qw_topic_sim.append(sim[i + 2 * n + m])
    return train_qw_topic_sim, test_qw_topic_sim


def topic_model():
    # 使用lsi主题模型
    # word级别
    sents_w = [str(words).split() for words in train_qw.tolist()]
    dictionary = corpora.Dictionary(sents_w)
    corpus_w = [dictionary.doc2bow(text) for text in sents_w]
    tfidf_w = models.TfidfModel(corpus_w)
    corpus_tfidf = tfidf_w[corpus_w]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
    corpus_lsi = lsi[corpus_tfidf]

    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=300)
    corpus_lda = lda[corpus_tfidf]
    # 获取question所属的topic
    qw_lsi = get_topic_model_topic(corpus_lsi, qids)
    print('qw_lis 完成')
    qw_lda = get_topic_model_topic(corpus_lda, qids)
    print('qw_lda 完成')

    # MemoryError 矩阵太大
    # 使用主题模型计算相似性
    # lsi
    # words_lsi_sim_file = './model/words_lsi_similarity.sim'
    # if not os.path.exists(words_lsi_sim_file):
    #     index_lsi = similarities.MatrixSimilarity(corpus_lsi)  # 得到的是句子shape为 [len(train_qw), len(train_qw)]
    #     index_lsi.save(words_lsi_sim_file)
    # else:
    #     index_lsi = models.LsiModel.load(words_lsi_sim_file)
    # sims_lsi = index_lsi[corpus_lsi]
    # train_qw_lsi_sim, test_qw_lsi_sim = get_topic_sim(sims_lsi)

    # lda
    # words_lda_sim_file = './model/words_lda_similarity.lda'
    # if not os.path.exists(words_lda_sim_file):
    #    index_lda = similarities.MatrixSimilarity(corpus_lda)  # 得到的是句子shape为 [len(train_qw), len(train_qw)]
    #   index_lda.save(words_lda_sim_file)
    # else:
    #    index_lda = models.LdaModel.load(words_lda_sim_file)
    # sims_lda = index_lda[corpus_lda]
    # train_qw_lda_sim, test_qw_lda_sim = get_topic_sim(sims_lda)

    # char级别
    sents_c = [str(words).split() for words in train_qc.tolist()]
    dictionary_c = corpora.Dictionary(sents_c)
    corpus_c = [dictionary_c.doc2bow(text) for text in sents_c]
    tfidf_c = models.TfidfModel(corpus_c)
    corpus_tfidf_c = tfidf_c[corpus_c]
    lsi_c = models.LsiModel(corpus_tfidf_c, id2word=dictionary_c, num_topics=300)
    corpus_lsi_c = lsi_c[corpus_tfidf_c]
    lda_c = models.LdaModel(corpus_tfidf_c, id2word=dictionary_c, num_topics=300)
    corpus_lda_c = lda_c[corpus_tfidf_c]
    # 获取question所属的topic
    qc_lsi = get_topic_model_topic(corpus_lsi_c, qids)
    print('qc_lis 完成')
    qc_lda = get_topic_model_topic(corpus_lda_c, qids)
    print('qc_lda 完成')

    # 使用主题模型计算相似性
    # lsi
    # chars_lsi_sim_file = './model/chars_lsi_similarity.lda'
    # if not os.path.exists(chars_lsi_sim_file):
    #     index_lsi_c = similarities.MatrixSimilarity(corpus_lsi_c)  # 得到的是句子shape为 [len(train_qw), len(train_qw)]
    #     index_lsi_c.save(chars_lsi_sim_file)
    # else:
    #     index_lsi_c = models.LsiModel.load(chars_lsi_sim_file)
    # sims_lsi_c = index_lsi_c[corpus_lsi_c]
    # train_qc_lsi_sim, test_qc_lsi_sim = get_topic_sim(sims_lsi_c)
    # #lda
    # chars_lda_sim_file = './model/chars_lda_similarity.lda'
    # if not os.path.exists(chars_lda_sim_file):
    #     index_lda_c = similarities.MatrixSimilarity(corpus_lda_c)  # 得到的是句子shape为 [len(train_qw), len(train_qw)]
    #     index_lda_c.save(chars_lda_sim_file)
    # else:
    #     index_lda_c = models.LdaModel.load(chars_lda_sim_file)
    # sims_lda_c = index_lda_c[corpus_lda_c]
    # train_qc_lda_sim, test_qc_lda_sim = get_topic_sim(sims_lda_c)

    # return qw_lsi, qc_lsi, qw_lda, qc_lda, train_qw_lsi_sim, test_qw_lsi_sim, train_qw_lda_sim, test_qw_lda_sim, train_qc_lsi_sim, test_qc_lsi_sim,train_qc_lda_sim, test_qc_lda_sim
    return qw_lsi, qc_lsi, qw_lda, qc_lda


def cos_distance(tfidf_q1, tfidf_q2):
    num = float(np.sum(tfidf_q1 * tfidf_q2))
    denom = np.linalg.norm(tfidf_q1) * np.linalg.norm(tfidf_q2)
    return num / denom


word2vec_model = KeyedVectors.load_word2vec_format('./data/word_embedding2.txt', binary=False)
word2vec_model.init_sims(replace=True)  # normalizes vectors
char2vec_model = KeyedVectors.load_word2vec_format('./data/word_embedding2.txt', binary=False)
char2vec_model.init_sims(replace=True)  # normalizes vectors


def get_word_moving_distance(sent1, sent2, word2vec_model):
    # 词移距离
    distance = word2vec_model.wmdistance(sent1.split(), sent2.split())
    return distance


def get_word_weight(word, q2vec_list, word_em_dict, bag_word_dict):
    if (word.lower() in bag_word_dict) and (word in word_em_dict):
        # print('46 word:{} \n word_em_dict[word]:{} \n bag_word_index_dict[word]:{}'.format(word, word_em_dict[word], bag_word_dict[word.lower()]))

        index = bag_word_dict[word.lower()]
        word_weight = q2vec_list[index] * np.sum(word_em_dict[word])
    else:
        word_weight = 0
    return word_weight


def tfidf_word_em_weight_match(row):
    q2vec1 = q2vec[row['qid1']]
    q2vec2 = q2vec[row['qid2']]
    qw1_weight, qw2_weight = {}, {}
    for word1 in row['wid1'].split():
        qw1_weight[word1] = get_word_weight(word1, q2vec1, words_em, bag_word_index_dict)
    for word2 in row['wid2'].split():
        qw2_weight[word2] = get_word_weight(word2, q2vec2, words_em, bag_word_index_dict)
    # print('qw1_weigth:{} \n qw2_weigth:{}'.format(qw1_weight, qw2_weight))

    shared_weights = [qw1_weight.get(w, 0) for w in qw1_weight.keys() if w in qw2_weight] + [qw2_weight.get(w, 0) for w
                                                                                             in
                                                                                             qw2_weight.keys() if
                                                                                             w in qw1_weight]
    total_weights = [qw1_weight.get(w, 0) for w in qw1_weight] + [qw2_weight.get(w, 0) for w in qw2_weight]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def tfidf_char_em_weigth_match(row):
    q2vec1 = q2vec_char[row['qid1']]
    q2vec2 = q2vec_char[row['qid2']]
    qc1_weight, qc2_weight = {}, {}
    for char1 in row['cid1'].split():
        qc1_weight[char1] = get_word_weight(char1, q2vec1, chars_em, bag_char_index_dict)
    for char2 in row['cid2'].split():
        qc2_weight[char2] = get_word_weight(char2, q2vec2, chars_em, bag_char_index_dict)
    # print('90 qw1_weigth:{} \n qw2_weigth:{}'.format(qc1_weight, qc2_weight))
    shared_weights = [qc1_weight.get(c, 0) for c in qc1_weight.keys() if c in qc2_weight] + [qc2_weight.get(c, 0) for c
                                                                                             in
                                                                                             qc2_weight.keys() if
                                                                                             c in qc1_weight]
    total_weights = [qc1_weight.get(c, 0) for c in qc1_weight] + [qc2_weight.get(c, 0) for c in qc2_weight]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def getNumofCommonSubsent(sent1, sent2):
    """
    求两个字符串的最长公共子串
    思想：建立一个二维数组，保存连续位相同与否的状态
    :param sent1: 如：'W105587 W101644 W102193 W106548 W104416'
    :param sent2: 如：'W105587 W101644 W102193 W104454'
    :return: 如：(['W105587', 'W101644', 'W102193'], 3)
    """
    sent1 = sent1.split()
    sent2 = sent2.split()
    lsent1 = len(sent1)
    lsent2 = len(sent2)
    record = [[0 for i in range(lsent2 + 1)] for j in range(lsent1 + 1)]  # 多一位
    maxNum = 0  # 最长匹配长度
    p = 0  # 匹配的起始位

    for i in range(lsent1):
        for j in range(lsent2):
            if sent1[i] == sent2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i + 1][j + 1]
                    # 记录最大匹配长度的终止位置
                    p = i + 1
    return sent1[p - maxNum:p], maxNum


def edit_distance(sent1, sent2):
    """
    计算句子的编辑距离
    :param sent1:
    :param sent2:
    :return:
    """
    sent1 = sent1.split()
    sent2 = sent2.split()
    len1 = len(sent1)
    len2 = len(sent2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if sent1[i - 1] == sent2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


def get_data_Feature(df_train, df_test):
    # train_data_feature
    # First we create our training and testing data
    train_feat = df_train.copy()  # train_feat.shape = (20000, 7)
    train_feat['word_match'] = df_train.apply(word_match_share, axis=1, raw=True)
    train_feat['char_match'] = df_train.apply(char_match_share, axis=1, raw=True)
    train_feat['tfidf_word_match'] = df_train.apply(tfidf_word_match_share, axis=1, raw=True)
    train_feat['tfidf_char_match'] = df_train.apply(tfidf_char_match_share, axis=1, raw=True)
    train_feat['tfidf_word_em_match'] = df_train.apply(tfidf_word_embed_match_share, axis=1, raw=True)
    train_feat['tfidf_char_em_match'] = df_train.apply(tfidf_char_embed_match_share, axis=1, raw=True)
    train_feat['wid1_len'] = df_train['wid1'].apply(get_length)
    train_feat['wid2_len'] = df_train['wid2'].apply(get_length)
    train_feat['cid1_len'] = df_train['cid1'].apply(get_length)
    train_feat['cid2_len'] = df_train['cid2'].apply(get_length)
    train_feat['wid1_em_mean'] = df_train['wid1'].apply(get_word_embedding_mean)
    train_feat['wid2_em_mean'] = df_train['wid2'].apply(get_word_embedding_mean)
    train_feat['cid1_em_mean'] = df_train['cid1'].apply(get_char_embedding_mean)
    train_feat['cid2_em_mean'] = df_train['cid2'].apply(get_char_embedding_mean)

    train_feat['tfidf_word_weigth_match'] = df_train.apply(tfidf_word_em_weight_match, axis=1, raw=True)
    train_feat['tfidf_char_weigth_match'] = df_train.apply(tfidf_char_em_weigth_match, axis=1, raw=True)

    train_feat['word_longest_common_subseq'] = [
        getNumofCommonSubsent(df_train['wid1'].iloc[i], df_train['wid2'].iloc[i])[1] for i in range(len(df_train))]
    train_feat['char_longest_common_subseq'] = [
        getNumofCommonSubsent(df_train['cid1'].iloc[i], df_train['cid2'].iloc[i])[1] for
        i in range(len(df_train))]

    train_feat['words_edit_distance'] = [edit_distance(df_train['wid1'].iloc[i], df_train['wid2'].iloc[i]) for i in
                                         range(len(df_train))]
    train_feat['chars_edit_distance'] = [edit_distance(df_train['cid1'].iloc[i], df_train['cid2'].iloc[i]) for i
                                         in range(len(df_train))]
    train_feat['words_cos_distance'] = [cos_distance(q2vec[df_train['qid1'].iloc[i]], q2vec[df_train['qid2'].iloc[i]])
                                        for i in range(len(df_train))]
    train_feat['chars_cos_distance'] = [cos_distance(q2vec_char[df_train['qid1'].iloc[i]], q2vec_char[df_train['qid2'].iloc[i]]) for i
        in range(len(df_train))]

    train_feat['word_moving_distance'] = [
        get_word_moving_distance(df_train['wid1'].iloc[i], df_train['wid2'].iloc[i], word2vec_model) for i in
        range(len(df_train))]
    train_feat['char_moving_distance'] = [
        get_word_moving_distance(df_train['cid1'].iloc[i], df_train['cid2'].iloc[i], char2vec_model) for i in
        range(len(df_train))]

    # get topic model features
    #    qw_lsi, qc_lsi, qw_lda, qc_lda, train_qw_lsi_sim, test_qw_lsi_sim, train_qw_lda_sim, test_qw_lda_sim, train_qc_lsi_sim, test_qc_lsi_sim, train_qc_lda_sim, test_qc_lda_sim = topic_model()
    qw_lsi, qc_lsi, qw_lda, qc_lda = topic_model()

    train_feat['qid1_word_lsi_topic'] = [qw_lsi[df_train['qid1'].iloc[i]] for i in range(len(df_train))]
    train_feat['qid2_word_lsi_topic'] = [qw_lsi[df_train['qid2'].iloc[i]] for i in range(len(df_train))]
    train_feat['qid1_char_lsi_topic'] = [qc_lsi[df_train['qid1'].iloc[i]] for i in range(len(df_train))]
    train_feat['qid2_char_lsi_topic'] = [qc_lsi[df_train['qid2'].iloc[i]] for i in range(len(df_train))]

    train_feat['qid1_word_lda_topic'] = [qw_lda[df_train['qid1'].iloc[i]] for i in range(len(df_train))]
    train_feat['qid2_word_lda_topic'] = [qw_lda[df_train['qid2'].iloc[i]] for i in range(len(df_train))]
    train_feat['qid1_char_lda_topic'] = [qc_lda[df_train['qid1'].iloc[i]] for i in range(len(df_train))]
    train_feat['qid2_char_lda_topic'] = [qc_lda[df_train['qid2'].iloc[i]] for i in range(len(df_train))]

    # train_feat['words_lsi_sim'] = train_qw_lsi_sim
    # train_feat['chars_lsi_sim'] = train_qc_lsi_sim
    # train_feat['words_lda_sim'] = train_qw_lda_sim
    # train_feat['chars_lda_sim'] = train_qc_lda_sim
    train_feat['words_lsi_cha'] = abs(train_feat['qid1_word_lsi_topic'] - train_feat['qid2_word_lsi_topic'])
    train_feat['words_lda_cha'] = abs(train_feat['qid1_word_lda_topic'] - train_feat['qid2_word_lda_topic'])
    train_feat['chars_lsi_cha'] = abs(train_feat['qid1_char_lsi_topic'] - train_feat['qid2_char_lsi_topic'])
    train_feat['chars_lda_cha'] = abs(train_feat['qid1_char_lda_topic'] - train_feat['qid2_char_lda_topic'])

    # pd.drop(['',''], axis=1, inplace=True)
    train_feat.drop(['qid1_word_lsi_topic', 'qid2_word_lsi_topic', 'qid1_char_lsi_topic', 'qid2_char_lsi_topic',
                     'qid1_word_lda_topic', 'qid2_word_lda_topic', 'qid1_char_lda_topic', 'qid2_char_lda_topic'],
                    axis=1, inplace=True)

    # train_feat.to_csv('./data/train_F28.csv', index=False, encoding='utf-8')

    # 数据归一化，归一化后预测效果非常差
    # train_feat = train_feat.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)),  axis=0)
    # train_feat = train_feat.apply(lambda x: (x - np.mean(x)) / (np.std(x)),  axis=0)


    # test_data_feature
    test_feat = df_test.copy()  # train_feat.shape = (10000, 7)
    test_feat['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
    test_feat['char_match'] = df_test.apply(char_match_share, axis=1, raw=True)
    test_feat['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)
    test_feat['tfidf_char_match'] = df_test.apply(tfidf_char_match_share, axis=1, raw=True)
    test_feat['tfidf_word_em_match'] = df_test.apply(tfidf_word_embed_match_share, axis=1, raw=True)
    test_feat['tfidf_char_em_match'] = df_test.apply(tfidf_char_embed_match_share, axis=1, raw=True)
    test_feat['wid1_len'] = df_test['wid1'].apply(get_length)
    test_feat['wid2_len'] = df_test['wid2'].apply(get_length)
    test_feat['cid1_len'] = df_test['cid1'].apply(get_length)
    test_feat['cid2_len'] = df_test['cid2'].apply(get_length)
    test_feat['wid1_em_mean'] = df_test['wid1'].apply(get_word_embedding_mean)
    test_feat['wid2_em_mean'] = df_test['wid2'].apply(get_word_embedding_mean)
    test_feat['cid1_em_mean'] = df_test['cid1'].apply(get_char_embedding_mean)
    test_feat['cid2_em_mean'] = df_test['cid2'].apply(get_char_embedding_mean)

    test_feat['tfidf_word_weigth_match'] = df_test.apply(tfidf_word_em_weight_match, axis=1, raw=True)
    test_feat['tfidf_char_weigth_match'] = df_test.apply(tfidf_char_em_weigth_match, axis=1, raw=True)

    test_feat['word_longest_common_subseq'] = [
        getNumofCommonSubsent(df_test['wid1'].iloc[i], df_test['wid2'].iloc[i])[1] for i in range(len(df_test))]
    test_feat['char_longest_common_subseq'] = [
        getNumofCommonSubsent(df_test['cid1'].iloc[i], df_test['cid2'].iloc[i])[1] for
        i in range(len(df_test))]

    test_feat['words_edit_distance'] = [edit_distance(df_test['wid1'].iloc[i], df_test['wid2'].iloc[i]) for i in
                                        range(len(df_test))]
    test_feat['chars_edit_distance'] = [edit_distance(df_test['cid1'].iloc[i], df_test['cid2'].iloc[i]) for i
                                        in range(len(df_test))]
    test_feat['words_cos_distance'] = [cos_distance(q2vec[df_test['qid1'].iloc[i]], q2vec[df_test['qid2'].iloc[i]]) for
                                       i in range(len(df_test))]
    test_feat['chars_cos_distance'] = [cos_distance(q2vec_char[df_test['qid1'].iloc[i]], q2vec_char[df_test['qid2'].iloc[i]]) for i
        in range(len(df_test))]
    test_feat['word_moving_distance'] = [
        get_word_moving_distance(df_test['wid1'].iloc[i], df_test['wid2'].iloc[i], word2vec_model) for i in
        range(len(df_test))]
    test_feat['char_moving_distance'] = [
        get_word_moving_distance(df_test['cid1'].iloc[i], df_test['cid2'].iloc[i], char2vec_model) for i in
        range(len(df_test))]


    test_feat['qid1_word_lsi_topic'] = [qw_lsi[df_test['qid1'].iloc[i]] for i in range(len(df_test))]
    test_feat['qid2_word_lsi_topic'] = [qw_lsi[df_test['qid2'].iloc[i]] for i in range(len(df_test))]
    test_feat['qid1_char_lsi_topic'] = [qc_lsi[df_test['qid1'].iloc[i]] for i in range(len(df_test))]
    test_feat['qid2_char_lsi_topic'] = [qc_lsi[df_test['qid2'].iloc[i]] for i in range(len(df_test))]

    test_feat['qid1_word_lda_topic'] = [qw_lda[df_test['qid1'].iloc[i]] for i in range(len(df_test))]
    test_feat['qid2_word_lda_topic'] = [qw_lda[df_test['qid2'].iloc[i]] for i in range(len(df_test))]
    test_feat['qid1_char_lda_topic'] = [qc_lda[df_test['qid1'].iloc[i]] for i in range(len(df_test))]
    test_feat['qid2_char_lda_topic'] = [qc_lda[df_test['qid2'].iloc[i]] for i in range(len(df_test))]

    # test_feat['words_lsi_sim'] = test_qw_lsi_sim
    # test_feat['chars_lsi_sim'] = test_qc_lsi_sim
    # test_feat['words_lda_sim'] = test_qw_lda_sim
    # test_feat['chars_lda_sim'] = test_qc_lda_sim

    test_feat['words_lsi_cha'] = abs(test_feat['qid1_word_lsi_topic'] - test_feat['qid2_word_lsi_topic'])
    test_feat['words_lda_cha'] = abs(test_feat['qid1_word_lda_topic'] - test_feat['qid2_word_lda_topic'])
    test_feat['chars_lsi_cha'] = abs(test_feat['qid1_char_lsi_topic'] - test_feat['qid2_char_lsi_topic'])
    test_feat['chars_lda_cha'] = abs(test_feat['qid1_char_lda_topic'] - test_feat['qid2_char_lda_topic'])

    # delete topic values
    test_feat.drop(['qid1_word_lsi_topic', 'qid2_word_lsi_topic', 'qid1_char_lsi_topic', 'qid2_char_lsi_topic',
                    'qid1_word_lda_topic', 'qid2_word_lda_topic', 'qid1_char_lda_topic', 'qid2_char_lda_topic'], axis=1,
                   inplace=True)

    # test_feat.to_csv('./data/test_F28.csv', index=False, encoding='utf-8')
    # 归一化处理
    # test_feat = test_feat.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=0)

    return train_feat, test_feat



def main():
    train_data, test_data = get_data_Feature(df_train, df_test)
    train_data.to_csv('./data/train_1w_F28.csv', index=False, encoding='utf-8')
    test_data.to_csv('./data/test_h1w_F28.csv', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()
