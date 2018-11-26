# -*- coding utf-8 -*-
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb


"""
    该实验转换为2分类模型。将数据离散化，使用sklearn模型进行分类。数据特征已保存。
"""
def train(train_date, flag='xgb'):
    x_train = train_date.iloc[:, 7:]
    y_train = train_date['label'].values
    # print('341: ', x_train.shape, len(y_train))
    # Finally, we split some of the data off for validation
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

    if flag == 'xgb':
        # XGBoost
        # Set our parameters for xgboost
        params = {}
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
        # params['eta'] = 0.04
        # params['max_depth'] = 8
        params['eta'] = 0.04
        params['max_depth'] = 5
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_valid = xgb.DMatrix(x_valid, label=y_valid)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=62, verbose_eval=10)

        # sklearn自带模型
        # cls_watchlist = [(x_train, y_train), (x_valid, y_valid)]
        # bst = xgb.XGBClassifier(max_depth=max_depth, learning_rate=eta, n_estimators=400, early_stopping_rounds=62, verbose_eval=10).fit(x_train,y_train,eval_set=cls_watchlist)
        return bst

    if flag == 'GaussianNB':
        from sklearn.naive_bayes import GaussianNB
        # Gaussian Naive Bayes classification
        model_gnb = GaussianNB()
        model_gnb.fit(x_train, y_train)
        return model_gnb

    if flag == 'svm':
        from sklearn.svm import SVC
        # 支持向量机
        C = 0.04
        gamma = 5
        model_svc = SVC(kernel='rbf', C=C, gamma=gamma)
        model_svc.fit(x_train, y_train)
        return model_svc


def testing(test_data, model, flag='xgb'):
    x_test = test_data.iloc[:, 7:]
    if flag == 'xgb':
        d_test = xgb.DMatrix(x_test)
        p_test = model.predict(d_test)
    else:
        p_test = model.predict(x_test)

    print('p_test[:20]: ', p_test[:20])

    sub = pd.DataFrame()
    sub['qid1'] = test_data['qid1']
    sub['qid2'] = test_data['qid2']
    # sub['label'] = p_test
    sub['label'] = np.zeros_like(test_data['qid2'])
    sub['label'][p_test > 0.5] = 1
    sub.to_csv('./data/result_xgb_cls_1w_181126.csv', index=False, encoding='utf-8')

def evluate(real, pred):
    from sklearn.metrics import classification_report
    result = classification_report(real, pred, digits=5)
    print('result: \n',result)
    return result


def main():
    # train_data, test_data = get_data_Feature(df_train, df_test)
    # 已有特征数据
    train_data = pd.read_csv('./data/train_1w_F28.csv', encoding='utf-8')
    test_data = pd.read_csv('./data/train_h1w_F28.csv', encoding='utf-8')
    # train_data = pd.read_csv('./data/train_F28.csv', encoding='utf-8')
    # test_data = pd.read_csv('./data/test_F28.csv', encoding='utf-8')

    # f = open('./data/xgb_params_181126_1.csv', 'a', encoding='utf-8')
    # f_r = open('./data/xgb_evluate_181126_1.csv', 'a', encoding='utf-8')
    # xgb parameter
    # eta = [0.02, 0.03, 0.04, 0.042, 0.044, 0.045, 0.046, 0.048, 0.05, 0.055, 0.06, 0.07]
    # max_depth = [4, 5, 6, 7, 8, 9, 10]
    # eta = [0.04]
    # max_depth = [5]
    # for e in eta:
    #     for d in max_depth:
    #         # f.write(str(e)+','+str(d)+'\n')
    #         flag = 'xgb'   # 'svm' , 'GaussianNB', 'xgb'
    #         bst = train(train_data, flag=flag, eta=e, max_depth=d)
    #         testing(test_data, bst, flag)
    #         sub = pd.read_csv('./data/result_xgb_cls_1w_181126.csv', encoding='utf-8')
    #         test_data_real = pd.read_csv('./data/train_prepare_h1w.csv', encoding='utf-8')
    #         real_label = test_data_real['label']
    #         pred_label = sub['label']
    #         result = evluate(real_label, pred_label)
            # f_r.write(result+'\n')
    # f.close()
    # f_r.close()

    # 可以换为其他sklearn分类模型来进行分类
    flag = 'xgb'   # 'svm' , 'GaussianNB', 'xgb'
    bst = train(train_data, flag=flag)
    testing(test_data, bst, flag)
    sub = pd.read_csv('./data/result_xgb_cls_1w_181126.csv', encoding='utf-8')
    test_data = pd.read_csv('./data/train_prepare_h1w.csv', encoding='utf-8')
    real_label = test_data['label']
    pred_label = sub['label']
    result = evluate(real_label, pred_label)


if __name__ == '__main__':
    main()