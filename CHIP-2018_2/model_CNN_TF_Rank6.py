# -*- coding utf-8 -*-
"""
# 项目：CHIP_2项目，句子对匹配
# 模型原型：第六名模型中的SiameseTextCNN的tensorflow版  https://github.com/TianyuZhuuu/CHIP2018
# 运行结果：目前只使用4000的训练集，测试1000的测试集，得到的test result约69.+ ，train result 80~90+，存在过拟合嫌疑
# 本实验没能使用完整数据，因为数据处理使用了for循环有待使用更有效的方法，数据存储有大量的冗余数据有待改进，MemoryError
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import OneHotEncoder
import json
import os
import pickle
from sklearn.metrics import f1_score

tf_config = tf.ConfigProto()

# parameter
epochs = 61
batch_size = 64
filter_num = 64
fc_dim = 128
learning_rate = 0.001
dropout = 0.5  #The dropout rate, E.g. "rate=0.1" would drop out 10% of input units.

tf.set_random_seed(1)
np.random.seed(1)

vec_length = 300

data_path = './data/'
with open(data_path + 'word_embedding.json', 'r') as f:
    word_vec = json.load(f)

with open(data_path + 'char_embedding.json', 'r') as f:
    char_vec = json.load(f)

def data_prepare(train_data, test_data):
    # n = train_data.shape[0]
    data_all = pd.concat([train_data, test_data])
    print('data_all shape: ', data_all.shape)
    train_data_new = data_all.copy()
    train_data_new['wid1'] = train_data_new['wid1'].str.split()
    train_data_new['wid1_len'] = train_data_new['wid1'].apply(lambda x: len(x))
    train_data_new['wid2'] = train_data_new['wid2'].str.split()
    train_data_new['wid2_len'] = train_data_new['wid2'].apply(lambda x: len(x))

    train_data_new['cid1'] = train_data_new['cid1'].str.split()
    train_data_new['cid1_len'] = train_data_new['cid1'].apply(lambda x: len(x))
    train_data_new['cid2'] = train_data_new['cid2'].str.split()
    train_data_new['cid2_len'] = train_data_new['cid2'].apply(lambda x: len(x))

    global max_words_length, max_chars_length
    max_words_length = max(pd.concat([train_data_new['wid1_len'], train_data_new['wid2_len']], axis=0))
    max_chars_length = max(pd.concat([train_data_new['cid1_len'], train_data_new['cid2_len']], axis=0))
    max_length = dict()
    max_length['max_words_length'] = max_words_length
    max_length['max_chars_length'] = max_chars_length
    print('max_words_length: {} \t max_chars_length: {}'.format(max_words_length, max_chars_length))

    # data prepare
    # train_data_new = data_prepare(train_data, test_data)
    train_data_new['wid1_vec'] = train_data_new['wid1'].apply(get_word_vec)
    train_data_new['wid2_vec'] = train_data_new['wid2'].apply(get_word_vec)
    train_data_new['cid1_vec'] = train_data_new['cid1'].apply(get_char_vec)
    train_data_new['cid2_vec'] = train_data_new['cid2'].apply(get_char_vec)
    # print(train_data_new.head(2))


    m = len(train_data)
    data = train_data_new[['wid1_vec', 'wid2_vec', 'cid1_vec', 'cid2_vec']].iloc[:m]
    data_label = label_onehot(train_data_new['label'].iloc[:m])
    print('train data shape: ', data.tial(2))

    test_data = train_data_new[['wid1_vec', 'wid2_vec', 'cid1_vec', 'cid2_vec']].iloc[m:]
    # test_data_label = label_onehot(train_data_new['label'].iloc[m:])
    print('test data shape: ', test_data.tail(2))
    # vail data
    n = 1000
    vail_data = data_asarray(data.iloc[-n:], data_label[-n:])
    with open('./data/vail_data', 'wb') as f:
        pickle.dump(vail_data, f)
    # train data
    train_data_ = iter([data.iloc[:-n, :], data_label[:-n, :]])
    with open('./data/train_data', 'wb') as f:
        pickle.dump(train_data_, f)
    # test data
    test_data_ = data_asarray(test_data, [])
    with open('./data/test_data', 'wb') as f:
        pickle.dump(test_data_, f)
    with open('./data/words_max_length','w', encoding='utf-8') as f:
        json.dump(max_length, f)
    return train_data_, vail_data, test_data_, max_words_length, max_chars_length


def get_word_vec(row):
    result = np.zeros(max_words_length*vec_length)
    for i,w in enumerate(row):
        # print(i*vec_length,(i+1)*vec_length)
        if word_vec.get(w):
            result[i*vec_length:(i+1)*vec_length] = word_vec[w]
    return result

def get_char_vec(row):
    result = np.zeros(max_chars_length*vec_length)
    for i,w in enumerate(row):
        if char_vec.get(w):
            result[i*vec_length:(i+1)*vec_length] = char_vec[w]
    return result

def data_batch(data, d_label, batch_size):
    '''
    Return a total of `num` random samples and labels.
    '''
    # idx = np.arange(0 , len(data))
    # np.random.shuffle(idx)
    data_wid1_vec, data_wid2_vec, data_label = [], [], []
    data_cid1_vec, data_cid2_vec = [], []
    n = len(data)//batch_size
    for i in range(n):
        data_b = data[i*batch_size:(i+1)*batch_size]
        data_wid1_vec.append(data_b['wid1_vec'].tolist())
        data_wid2_vec.append(data_b['wid2_vec'].tolist())

        data_cid1_vec.append(data_b['cid1_vec'].tolist())
        data_cid2_vec.append(data_b['cid2_vec'].tolist())
        # data_label.append(data_b['label'].apply(lambda x: [x]).tolist())
        data_label.append(d_label[i*batch_size:(i+1)*batch_size])
    print('data_batch: ', len(data_wid1_vec), len(data_wid2_vec), len(data_label))
    return np.asarray(data_wid1_vec), np.asarray(data_wid2_vec), np.asarray(data_cid1_vec), np.asarray(data_cid2_vec), np.asarray(data_label)


def label_onehot(data_label):
    label_onehot = data_label.values.reshape(len(data_label), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    return onehot_encoder.fit_transform(label_onehot)


def data_asarray(data, data_label):
    word_x1 = np.asanyarray(data['wid1_vec'].tolist())
    word_x2 = np.asanyarray(data['wid2_vec'].tolist())
    char_x1 = np.asanyarray(data['cid1_vec'].tolist())
    char_x2 = np.asanyarray(data['cid2_vec'].tolist())
    if len(data_label)!=0:
        y = np.asanyarray(data_label)
        vail_data = iter([word_x1, word_x2, char_x1, char_x2, y])
    else:
        vail_data = iter([word_x1, word_x2, char_x1, char_x2])
    return vail_data

def interact(x1, x2, axis=-1):
    """
    :param x1: [batch, (len1), dim]
    :param x2: [batch, (len2), dim]
    :return:
    """
    diff = tf.abs(x1 - x2)
    prod = x1 * x2
    concat = tf.concat((x1, x2, diff, prod), axis=axis)
    return concat


def evluate(real, pred):
    from sklearn.metrics import classification_report
    result = classification_report(real, pred, digits=5)
    print('result: \n',result)
    return result


class Model():
    def model(self, max_words_length, max_chars_length):
        # ----------# model structure------------ #
        # placeholder
        # word
        self.tf_x1 = tf.placeholder(tf.float32, [None, max_words_length*vec_length])
        self.tf_xs1 = tf.reshape(self.tf_x1, [-1,max_words_length,vec_length,1])
        self.tf_x2 = tf.placeholder(tf.float32, [None, max_words_length*vec_length])
        self.tf_xs2 = tf.reshape(self.tf_x2, [-1,max_words_length,vec_length,1])
        # char
        self.tf_char_x1 = tf.placeholder(tf.float32, [None, max_chars_length*vec_length])
        self.tf_char_xs1 = tf.reshape(self.tf_char_x1, [-1,max_chars_length,vec_length,1])
        self.tf_char_x2 = tf.placeholder(tf.float32, [None, max_chars_length*vec_length])
        self.tf_char_xs2 = tf.reshape(self.tf_char_x2, [-1,max_chars_length,vec_length,1])

        self.tf_y = tf.placeholder(tf.int32, [None,2])
        # keep_prob = tf.placeholder(tf.float32)
        self.tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing


        def conv2d(tf_data, filter_num):
            # conv kernel_size=(1,300) out: shape=(?, 43, 1, filer_num)
            conv_branch_1 = tf.layers.conv2d(inputs=tf_data, filters=filter_num, kernel_size=(1,300), strides=1, padding='valid', activation=tf.nn.relu)  # -> (43, 1, 64)
            print('169 conv conv_branch_1: ', conv_branch_1)
            conv_branch_1 = tf.reduce_max(tf.squeeze(conv_branch_1, axis=2), axis=1)  # squeeze & max: [batch, num_filter]
            print('170 conv_branch_1: ', conv_branch_1)
            conv_branch_2 = tf.layers.conv2d(inputs=tf_data, filters=filter_num, kernel_size=(2,300), strides=1, padding='valid', activation=tf.nn.relu)  # -> (43, 1, 64)
            conv_branch_2 = tf.reduce_max(tf.squeeze(conv_branch_2, axis=2), axis=1)
            conv_branch_3 = tf.layers.conv2d(inputs=tf_data, filters=filter_num, kernel_size=(3,300), strides=1, padding='valid', activation=tf.nn.relu)  # -> (43, 1, 64)
            conv_branch_3 = tf.reduce_max(tf.squeeze(conv_branch_3, axis=2), axis=1)
            conv_data =  tf.concat([conv_branch_1, conv_branch_2, conv_branch_3], axis=1)  # -> shape=(batch_size, 3*filter_num)
            return conv_data

        # q1 words
        q1_word_conv = conv2d(self.tf_xs1, filter_num=filter_num)
        print('q1_word_conv: ', q1_word_conv)
        q2_word_conv = conv2d(self.tf_xs2, filter_num=filter_num)
        print('q2_word_conv: ', q2_word_conv)
        q_word = interact(q1_word_conv, q2_word_conv, axis=-1)   # -> shape=(batch_size, 4*3*filter_num)
        print('q_word: ', q_word)

        # q1 chars
        q1_char_conv = conv2d(self.tf_char_xs1, filter_num=filter_num)
        print('q1_char_conv: ', q1_char_conv)
        q2_char_conv = conv2d(self.tf_char_xs2, filter_num=filter_num)
        print('q2_char_conv: ', q2_char_conv)
        q_char = interact(q1_char_conv, q2_char_conv, axis=-1)    # -> shape=(batch_size, 4*3*filter_num)
        print('q_char: ', q_char)


        x_feat = tf.concat([q_word, q_char], axis=1)  # -> shape=(batch_size, 2*4*3*filter_num)
        print('x_feat: ', x_feat)
        # i,j,k = x_feat.shape[1], x_feat.shape[2], x_feat.shape[3]
        # print('ijk',i, j, k)
        # flat = tf.reshape(x_feat, [-1, i*j*k])
        fc1 = tf.layers.dense(x_feat, fc_dim, activation=tf.nn.relu)
        fc1_d = tf.layers.dropout(fc1, rate=dropout, training=self.tf_is_training)
        self.output = tf.layers.dense(fc1_d, 2, activation=tf.nn.sigmoid)
        print('output :', self.output)
        # loss
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.tf_y, logits=self.output)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
        # evaluate
        # self.acc = tf.metrics.accuracy(labels=tf.argmax(self.tf_y, axis=1), predictions=tf.argmax(self.output, axis=1),)[1]
        # compute loss for f1 score
        recall = tf.metrics.recall(labels=tf.argmax(self.tf_y, axis=1), predictions=tf.argmax(self.output, axis=1),)[1]
        precision = tf.metrics.precision(labels=tf.argmax(self.tf_y, axis=1), predictions=tf.argmax(self.output, axis=1), )[1]
        self.acc = 2 * (precision * recall) / (precision + recall)
        self.saver = tf.train.Saver()
# ---------- # end model structure------------ #


def train(train_data, vail_data, max_words_length, max_chars_length):
    vail_x1, vail_x2, vail_char_x1, vail_char_x2, vail_y = vail_data
    # print(vail_data.shape)  # (2000, 10)

    train_data_x, train_label = train_data
    train_x1_batch, train_x2_batch, train_char_x1_batch, train_char_x2_batch, train_y_batch = data_batch(train_data_x,train_label,
                                                                                                         batch_size=batch_size)
    # print('trian_y_batch: ',train_x1_batch.shape,train_x1_batch[0].shape)

    built = Model()
    built.model(max_words_length, max_chars_length)
    # initialization. global an local must be initializate
    sess = tf.Session(config=tf_config)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
    sess.run(init_op)     # initialize var in graph
    # train
    for step in range(epochs):
        for i in range(len(train_data_x)//batch_size):
            b_x1 = train_x1_batch[i]
            b_x2 = train_x2_batch[i]
            b_char_x1 = train_char_x1_batch[i]
            b_char_x2 = train_char_x2_batch[i]
            b_y = train_y_batch[i]
            # print('b_x1: ', b_x1.shape)
            # print('step:{} \t b_x1:{} '.format(step, b_x1))
            _, loss_, train_acc = sess.run([built.train_op, built.loss, built.acc],
                                           feed_dict={built.tf_x1: b_x1, built.tf_x2: b_x2,
                                                      built.tf_char_x1: b_char_x1, built.tf_char_x2: b_char_x2,
                                                      built.tf_y: b_y, built.tf_is_training:True})
            # print('i: {} \t loss_: {} \t train_acc: {}'.format(i, loss_, train_acc))

        if step%20==0:
            accuracy_, _ = sess.run([built.acc, built.output], feed_dict={built.tf_x1: vail_x1, built.tf_x2: vail_x2,
                                                         built.tf_char_x1: vail_char_x1, built.tf_char_x2: vail_char_x2,
                                                         built.tf_y: vail_y, built.tf_is_training:False})
            print('Step:', step, '| train loss: %.4f' % loss_, '| train accuracy: %.5f' % train_acc, '| vail accuracy: %.5f' % accuracy_)

            # # print 10 predictions from test data
            # test_output = sess.run(output, {tf_x1: test_x1[:10], tf_x2: test_x2[:10], tf_char_x1: test_char_x1[:10], tf_char_x2: test_char_x2[:10], tf_is_training:False})
            # pred_y = np.argmax(test_output, 1)
            # print('step: ', step)
            # print(pred_y, 'prediction number')
            # print(np.argmax(test_y[:10], 1), 'real number')
    # save model
    # saver = tf.train.Saver()
    built.saver.save(sess, './model/CNN_model_1206.ckpt')

def train_mix(train_data, vail_data,test_data, test_pred, max_words_length, max_chars_length):
    # vail_x1, vail_x2, vail_char_x1, vail_char_x2, vail_y = vail_data
    test_x1, test_x2, test_char_x1, test_char_x2 = test_data
    train_data_x, train_label = train_data
    # model
    built = Model()
    built.model(max_words_length, max_chars_length)
    columns = ['wid1_vec', 'wid2_vec', 'cid1_vec', 'cid2_vec']
    # K-fold cross validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    train_data_x = np.array(train_data_x)
    train_label = np.array(train_label)
    kf.get_n_splits(train_data_x)
    for train_index, vail_index in kf.split(train_data_x):
        print(train_index[:5], train_index[-5:], vail_index[:5], vail_index[-5:])
        train_x_, vail_x_ = train_data_x[train_index], train_data_x[vail_index]
        train_y_, vail_y_ = train_label[train_index], train_label[vail_index]

        train_x_ = pd.DataFrame(train_x_, columns=columns)
        vail_x_ = pd.DataFrame(vail_x_, columns=columns)
        train_x1_batch, train_x2_batch, train_char_x1_batch, train_char_x2_batch, train_y_batch = data_batch(train_x_,train_y_,
                                                                                                         batch_size=batch_size)
        vail_x1, vail_x2, vail_char_x1, vail_char_x2, vail_y = data_asarray(vail_x_, vail_y_)


        # initialization. global an local must be initializate
        sess = tf.Session(config=tf_config)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
        sess.run(init_op)     # initialize var in graph
        # train
        for step in range(epochs):
            for i in range(len(train_x_)//batch_size):
                b_x1 = train_x1_batch[i]
                b_x2 = train_x2_batch[i]
                b_char_x1 = train_char_x1_batch[i]
                b_char_x2 = train_char_x2_batch[i]
                b_y = train_y_batch[i]
                # print('b_x1: ', b_x1.shape)
                # print('step:{} \t b_x1:{} '.format(step, b_x1))
                _, loss_, train_acc = sess.run([built.train_op, built.loss, built.acc],
                                               feed_dict={built.tf_x1: b_x1, built.tf_x2: b_x2,
                                                          built.tf_char_x1: b_char_x1, built.tf_char_x2: b_char_x2,
                                                          built.tf_y: b_y, built.tf_is_training:True})
                # print('i: {} \t loss_: {} \t train_acc: {}'.format(i, loss_, train_acc))
            # print('318  step: {} \t loss_: {} \t train_acc: {}'.format(step, loss_, train_acc))

            if step%20==0:
                accuracy_, _ = sess.run([built.acc, built.output], feed_dict={built.tf_x1: vail_x1, built.tf_x2: vail_x2,
                                                             built.tf_char_x1: vail_char_x1, built.tf_char_x2: vail_char_x2,
                                                             built.tf_y: vail_y, built.tf_is_training:False})
                print('Step:', step, '| train loss: %.4f' % loss_, '| train accuracy: %.5f' % train_acc, '| vail accuracy: %.5f' % accuracy_)


                # testing

                test_output = sess.run(built.output, feed_dict={built.tf_x1: test_x1, built.tf_x2: test_x2,
                                                                              built.tf_char_x1: test_char_x1,
                                                                              built.tf_char_x2: test_char_x2,
                                                                              built.tf_is_training: False})
                pred_y = np.argmax(test_output, 1)
                test_pred['label_pred'] = pred_y
                if not os.path.exists('./result'):
                    os.makedirs('./result')
                test_pred.to_csv('./result/test_pred.csv', encoding='utf-8', index=False)
                # evaluate result
                data_real = pd.read_csv('./result/test_withlabel.csv', encoding='utf-8')
                data_pred = pd.read_csv('./result/test_pred.csv', encoding='utf-8')
                data = pd.merge(data_pred, data_real, how='inner', on=['qid1', 'qid2'])
                evluate(real=data['label'], pred=data['label_pred'])
    built.saver.save(sess, './model/CNN_model_1206_' + str(step) + '.ckpt')


def test(test_data, test_pred, max_words_length, max_chars_length):
    test_x1, test_x2, test_char_x1, test_char_x2 = test_data

    model_built = Model()
    model_built.model(max_words_length, max_chars_length)
    # saver = tf.train.Saver()

    with tf.Session(config=tf_config) as sess:
        # saver.restore(sess, './model/CNN_model_1206.ckpt')
        model_built.saver.restore(sess, './model/CNN_model_1206.ckpt')
        # predictions from test data
        test_output = sess.run(model_built.output, feed_dict={model_built.tf_x1: test_x1, model_built.tf_x2: test_x2,
                                                              model_built.tf_char_x1: test_char_x1, model_built.tf_char_x2: test_char_x2,
                                                              model_built.tf_is_training:False})
    pred_y = np.argmax(test_output, 1)
    test_pred['label_pred'] = pred_y
    if not os.path.exists('./result'):
        os.makedirs('./result')
    test_pred.to_csv('./result/test_pred.csv', encoding='utf-8', index=False)


def main():
    # first prepare data
    # train_data = pd.read_csv(data_path + 'train_prepare.csv', encoding='utf-8').iloc[:5000]
    # test_data = pd.read_csv(data_path + 'test_prepare.csv', encoding='utf-8').iloc[:1000]
    # train_data_, vail_data, test_data_, max_words_length, max_chars_length = data_prepare(train_data, test_data)

    # second training
    # with open('./data/train_data_4000', 'rb') as f:
    #     train_data_ = pickle.load(f)
    # with open('./data/vail_data_1000', 'rb') as f:
    #     vail_data_ = pickle.load(f)
    # with open('./data/words_max_length', 'r') as f:
    #     max_length = json.load(f)
    # train(train_data_, vail_data_, max_length['max_words_length'], max_length['max_chars_length'])

    # training and testing
    test_data_0 = pd.read_csv(data_path + 'test_prepare.csv', encoding='utf-8').iloc[:1000]
    test_pred = test_data_0[['qid1', 'qid2']].copy()
    with open('./data/train_data_4000', 'rb') as f:
        train_data_ = pickle.load(f)
    with open('./data/vail_data_1000', 'rb') as f:
        vail_data_ = pickle.load(f)
    with open('./data/test_data_1000', 'rb') as f:
        test_data_ = pickle.load(f)
    with open('./data/words_max_length', 'r') as f:
        max_length = json.load(f)
    train_mix(train_data_, vail_data_,test_data_, test_pred, max_words_length=max_length['max_words_length'], max_chars_length=max_length['max_chars_length'])


    # thrid testing  # test 和 train 不可同时运行。因为同时运行时，构建的model参数名称不同了，是直接重新构建了一个model数据流
    # test_data = pd.read_csv(data_path + 'test_prepare.csv', encoding='utf-8').iloc[:1000]
    # test_pred = test_data[['qid1', 'qid2']].copy()
    # with open('./data/words_max_length', 'r') as f:
    #     max_length = json.load(f)
    # with open('./data/test_data_1000', 'rb') as f:
    #     test_data_ = pickle.load(f)
    # test(test_data_, test_pred, max_length['max_words_length'], max_length['max_chars_length'])

    # evaluate result
    # data_real = pd.read_csv('./result/test_withlabel.csv', encoding='utf-8')
    # data_pred = pd.read_csv('./result/test_pred.csv', encoding='utf-8')
    # data = pd.merge(data_pred, data_real, how='inner', on=['qid1', 'qid2'])
    # evluate(real=data['label'], pred=data['label_pred'])

if __name__=='__main__':
    main()