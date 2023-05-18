import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, multilabel_confusion_matrix, \
    precision_score, recall_score, f1_score, r2_score, accuracy_score, balanced_accuracy_score, roc_curve, auc, \
    classification_report, matthews_corrcoef, fowlkes_mallows_score
    # , mean_absolute_percentage_error
plt.rc('font', family='Times New Roman')


def mean_absolute_percentage_error(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class Evaluate:
    def __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag=False, nor_std=None):
        self.data = copy.deepcopy(data)
        self.eval_type = eval_type
        self.task_name = task_name
        self.name_clf = name_clf
        self.nor_flag = nor_flag
        self.nor_std = nor_std
        self.true, self.pred = self.nor_inver(true, pred)

    def nor_inver(self, true, pred):
        if self.nor_flag:
            data_inver = []
            for labelorpred in [true, pred]:
                self.data.loc[:, 'label0'] = labelorpred
                data_inver.append(self.nor_std.inverse_transform(self.data))
            true = data_inver[0][:, -1].reshape(-1, 1)
            pred = data_inver[1][:, -1].reshape(-1, 1)
        return true, pred


class Eval_Class(Evaluate):
    def __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag=False, nor_std=None):
        Evaluate. __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag, nor_std)
        if pred.shape[1] > 1:
            self.pred = np.argmax(pred, axis=1)
        else:
            self.pred = pred
        if len(true.shape) < 2 or true.shape[1] > 1:
            self.true = np.argmax(true, axis=1)
        else:
            self.true = true

    def eval(self):
        if np.unique(self.true).shape[0] < 3:
            auc_ = 0
            if 'class' in self.task_name:
                fpr, tpr, thresholds = roc_curve(self.true, self.pred)
                auc_ = auc(fpr, tpr)
                youden = (1 - fpr) + tpr - 1
                index_max = np.argmax(youden)
                threshold_max = thresholds[index_max]
        else:
            auc_ = 0
            threshold_max = 0.5
        pred = np.array([(lambda x: 0 if x < threshold_max else 1)(i) for i in self.pred])
        acc = accuracy_score(self.true, pred)
        acc_balanced = balanced_accuracy_score(self.true, pred)
        pre_weighted = precision_score(self.true, pred, average='weighted')
        pre_macro = precision_score(self.true, pred, average='macro')
        recall_weighted = recall_score(self.true, pred, average='weighted')
        recall_macro = recall_score(self.true, pred, average='macro')
        f1_weighted = f1_score(self.true, pred, average='weighted')
        f1_macro = f1_score(self.true, pred, average='macro')
        mcc = matthews_corrcoef(self.true, pred)
        fms = fowlkes_mallows_score(self.true.reshape(-1, ), pred.reshape(-1, ))
        # print(classification_report(self.true, pred))

        con_mat = confusion_matrix(self.true, pred)
        metric_dict = dict(zip([
                                '{} acc'.format(self.eval_type),
                                '{} acc_balanced'.format(self.eval_type),
                                '{} pre_weighted'.format(self.eval_type),
                                '{} pre_macro'.format(self.eval_type),
                                '{} recall_weighted'.format(self.eval_type),
                                '{} recall_macro'.format(self.eval_type),
                                '{} f1_weighted'.format(self.eval_type), '{} f1_macro'.format(self.eval_type),
                                '{} auc_'.format(self.eval_type),
                                '{} mcc'.format(self.eval_type),
                                '{} fms'.format(self.eval_type),
                                ],
                               [acc, acc_balanced, pre_weighted, pre_macro, recall_weighted, recall_macro, f1_weighted,
                                f1_macro, auc_, mcc, fms]))
        return pd.DataFrame([metric_dict], index=[self.name_clf])


class Eval_Regre(Evaluate):
    def __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag=False, nor_std=None):
        Evaluate. __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag, nor_std)

    def eval(self):
        r2 = r2_score(self.true, self.pred)
        mse = mean_squared_error(self.true, self.pred)
        mae = mean_absolute_error(self.true, self.pred)
        mape = mean_absolute_percentage_error(self.true, self.pred)
        metric_dict = dict(zip(['{} r2'.format(self.eval_type), '{} mae'.format(self.eval_type),
                                '{} mse'.format(self.eval_type), '{} mape'.format(self.eval_type)],
                               [r2, mae, mse, mape]))
        return pd.DataFrame([metric_dict], index=[self.name_clf])


def mvi_rec_evalution(data_raw, index_manual_nan, data_filled, name_clf=0, name_metric='test'):
    data_filled = data_filled.filter(regex=r'dense')
    data_raw = data_raw[data_filled.columns]
    index_manual_nan = index_manual_nan[data_filled.columns]
    if data_filled.shape[0] > data_raw.shape[0]:
        index_u = set(data_raw.index) & set(data_filled.index)
        data_filled = data_filled.loc[index_u]
        data_raw = data_raw.loc[index_u]
    elif data_filled.shape[0] == data_raw.shape[0]:
        data_filled.index = data_raw.index
    else:
        data_raw = data_raw.iloc[0: data_filled.shape[0]]
        data_filled.index = data_raw.index
        data_raw = data_raw.loc[data_filled.index]
    index_notna_ = index_manual_nan.loc[data_filled.index]

    data_raw_ = data_raw.fillna(data_filled)
    mae_byall = mean_absolute_error(data_raw_, data_filled)
    mse_byall = mean_squared_error(data_raw_, data_filled)
    mape_byall = mean_absolute_percentage_error(data_raw_, data_filled)
    rmse_byall = np.sqrt(mean_squared_error(data_raw_, data_filled))
    index_notna = data_raw.notna()
    index_manul_nan = np.logical_xor(index_notna, index_notna_)
    data_raw_labeldrop = data_raw.drop(columns=data_raw.filter(regex=r'label').columns)
    data_filled_labeldrop = data_filled.drop(columns=data_filled.filter(regex=r'label').columns)
    mms = MinMaxScaler(feature_range=(0, 1))
    std = StandardScaler()
    nor = std
    data_raw_labeldrop = pd.DataFrame(nor.fit_transform(data_raw_labeldrop),
                                      columns=data_raw_labeldrop.columns,
                                      index=data_raw_labeldrop.index)
    data_filled_labeldrop = pd.DataFrame(nor.fit_transform(data_filled_labeldrop),
                                         columns=data_filled_labeldrop.columns,
                                         index=data_filled_labeldrop.index)
    mae_list = []
    mse_list = []
    rmse_list = []
    mape_list = []
    for index_, values in index_manul_nan.iteritems():
        data_raw_col = data_raw_labeldrop[[index_]][values == True]
        data_filled_col = data_filled_labeldrop[[index_]][values == True]
        if data_raw_col.empty:
            continue
        mae_bycol = mean_absolute_error(data_raw_col, data_filled_col)
        mse_bycol = mean_squared_error(data_raw_col, data_filled_col)
        mape_bycol = mean_absolute_percentage_error(data_raw_col, data_filled_col)
        rmse_bycol = np.sqrt(mean_squared_error(data_raw_col, data_filled_col))
        mae_list.append(mae_bycol)
        mse_list.append(mse_bycol)
        mape_list.append(mape_bycol)
        rmse_list.append(rmse_bycol)
    metric_dict = dict(zip([f'{name_metric} mae_bycol', f'{name_metric} mse_bycol', f'{name_metric} rmse_bycol', f'{name_metric} mae_byall', f'{name_metric} mse_byall', f'{name_metric} rmse_byall'],
                           [np.mean(mae_list), np.mean(mse_list), np.mean(rmse_list), mae_byall, mse_byall, rmse_byall]))
    metric_df = pd.DataFrame([metric_dict], index=[name_clf])
    print(f'metric{metric_df}')
    return metric_df
