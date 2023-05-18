import copy
import time
import re
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from preprocess import load_data
from preprocess.get_dataset import DataPreprocessing
from preprocess.missing_values_imputation import MVI
from preprocess.imbalance_process import IMB
from model.evaluate import Eval_Regre, Eval_Class, mvi_rec_evalution
from model.baseline import Baseline, GBM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def model_tra_eval(train_set, train_label, val_set, val_label, test_set, test_label,
                   target, co_col, ca_col, task_name, nor_std, seed):
    train_set: pd.DataFrame
    train_label: pd.DataFrame
    val_set: pd.DataFrame
    val_label: pd.DataFrame
    test_set: pd.DataFrame
    val_label: pd.DataFrame
    target: dict

    metric_all = pd.DataFrame([])
    imp_feat_ = pd.DataFrame([])
    name_model_list = [
        ['GBM', 'LGBMClassifier'],
    ]
    for name_model in name_model_list:
        model_method = eval(name_model[0])(name_model[1],
                   train_set, train_label,
                   val_set, val_label,
                   test_set, test_label, target, co_col, ca_col, task_name, seed, param_init={})
        pred_tra, pred_val, pred_test, model = model_method.grid_fit_pred()
        imp_feat_ = model_method.imp_feat()
        if 'class' in task_name:
            metric = Eval_Class
        else:
            metric = Eval_Regre
        for index_, values in pred_tra.iteritems():
            metric_tra = metric(train_label.loc[:, index_].values.reshape(-1, 1),
                                pred_tra.loc[:, index_].values.reshape(-1, 1), train_set,
                                'train', task_name, name_model[1], nor_flag=False, nor_std=nor_std).eval()
            metric_val = metric(val_label.loc[:, index_].values.reshape(-1, 1),
                                pred_val.loc[:, index_].values.reshape(-1, 1), val_set,
                                'val', task_name, name_model[1], nor_flag=False, nor_std=nor_std).eval()
            metric_test = metric(test_label.loc[:, index_].values.reshape(-1, 1),
                                 pred_test.loc[:, index_].values.reshape(-1, 1), test_set,
                                 'test', task_name, name_model[1], nor_flag=False, nor_std=nor_std).eval()
            metric_single = pd.concat([metric_test, metric_val, metric_tra], axis=1)
            metric_all = pd.concat([metric_all, metric_single], axis=0)
            print(metric_all)
    return metric_all, pd.DataFrame(pred_tra, columns=train_label.columns, index=train_label.index), \
               pd.DataFrame(pred_test, columns=test_label.columns, index=test_label.index), imp_feat_


def run(train_data, test_data, target, args, seed=2022) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    target: dict
    if args.test_ratio == 0 or not test_data.empty:
        train_set = train_data
        test_set = test_data
    else:
        train_set, test_set = train_test_split(train_data, test_size=args.test_ratio, random_state=seed)

    metric_df_all_run = pd.DataFrame([])
    imp_feat_df_all_run = pd.DataFrame([])
    pred_train_df_all = pd.DataFrame([])
    pred_test_df_all = pd.DataFrame([])
    test_label_all = pd.DataFrame([])
    kf = KFold(n_splits=args.n_splits)
    for k, (train_index, val_index) in enumerate(kf.split(train_set)):
        print(f'KFlod {k}')
        metric_all_ = pd.DataFrame([])
        train_set_cv = train_set.iloc[train_index]
        val_set_cv = train_set.iloc[val_index]
        test_set_cv = copy.deepcopy(test_set)

        dp = DataPreprocessing(train_set_cv, val_set_cv, test_set_cv, target, seed,
                               flag_label_onehot=False,
                               flag_ex_null=True, flag_ex_std_flag=False, flag_ex_occ=False,
                               flag_ca_co_sel=True, flag_ca_fac=True, flag_onehot=False, flag_nor=True,
                               flag_feat_emb=False, flag_RUS=False, flaq_save=False)
        if args.Flag_DataPreprocessing:
            train_set_cv, val_set_cv, test_set_cv, ca_col, co_col, nor = dp.process()
            train_set_cv.to_csv(f'./Train_KFlod[{k}]_{args.task_name}.csv', index_label=['index'])
            val_set_cv.to_csv(f'./Val_KFlod[{k}]_{args.task_name}.csv', index_label=['index'])
            test_set_cv.to_csv(f'./Test_KFlod[{k}]_{args.task_name}.csv', index_label=['index'])
        else:
            train_set_cv = pd.read_csv(f'../DataSet/processed_flod/data_kflod/Train_KFlod[{k}]_{args.task_name}.csv', index_col=['index'])
            val_set_cv = pd.read_csv(f'../DataSet/processed_flod/data_kflod/Val_KFlod[{k}]_{args.task_name}.csv', index_col=['index'])
            test_set_cv = pd.read_csv(f'../DataSet/processed_flod/data_kflod/Test_KFlod[{k}]_{args.task_name}.csv', index_col=['index'])
            nor = None
        col_drop = dp.features_ex(train_set_cv)
        train_set_cv.drop(columns=col_drop, inplace=True)
        col_ = train_set_cv.columns
        val_set_cv = val_set_cv[col_]
        test_set_cv = test_set_cv[col_]
        ca_col = train_set_cv.filter(regex=r'sparse').columns.tolist()
        co_col = train_set_cv.filter(regex=r'dense').columns.tolist()

        train_label = train_set_cv[[target['label1']]]
        val_label = val_set_cv[[target['label1']]]
        test_label = test_set_cv[[target['label1']]]
        train_x = train_set_cv.drop(columns=target.values())
        val_x = val_set_cv.drop(columns=target.values())
        test_x = test_set_cv.drop(columns=target.values())

        print(f'train_x shape {train_x.shape} | val_x shape {val_x.shape} | test_x shape {test_x.shape}')
        if args.Flag_MVI:
            mvi = MVI(train_x.shape[1], co_col, ca_col, args.task_name, target, seed, method=args.method_mvi)
            train_x_mn, train_label, index_ManualNan_train = mvi.manual_nan(train_x, train_label, args.ratio_manual, k, type_data='Train', flag_saving=args.Flag_Mask_Saving)
            train_x = train_x.loc[train_x_mn.index]
            val_x_mn, val_label, index_ManualNan_val = mvi.manual_nan(val_x, val_label, args.ratio_manual, k, type_data='Val', flag_saving=args.Flag_Mask_Saving)
            test_x_mn, test_label, index_ManualNan_test = mvi.manual_nan(test_x, test_label, args.ratio_manual, k, type_data='Test', flag_saving=args.Flag_Mask_Saving)
            train_x_filled, train_label = mvi.fit_transform(train_x_mn, train_x, train_label)
            val_x_filled = mvi.transform(val_x_mn)
            test_x_filled = mvi.transform(test_x_mn)

            pd.concat([train_x_filled, train_label], axis=1).to_csv(
                f'./TrainFilled_mvi[{args.method_mvi}]_KFlod[{k}]_{args.task_name}.csv', index_label=['index'])
            pd.concat([val_x_filled, val_label], axis=1).to_csv(
                f'./ValFilled_mvi[{args.method_mvi}]_KFlod[{k}]_{args.task_name}.csv', index_label=['index'])
            pd.concat([test_x_filled, test_label], axis=1).to_csv(
                f'./TestFilled_mvi[{args.method_mvi}]_KFlod[{k}]_{args.task_name}.csv', index_label=['index'])
        else:
            path_root = f'../DataSet/mimic/filled/mr{args.ratio_manual["mr"]}/'
            path_list_label = os.listdir(path_root)

            path_train = None
            for path in path_list_label:
                path_re = re.search(r'TrainFilled_mvi\[{}\]_KFlod\[{}\]'.format(args.method_mvi, k), path)
                if path_re:
                    path_train = path_root + path_re.string
                path_re = re.search(r'ValFilled_mvi\[{}\]_KFlod\[{}\]'.format(args.method_mvi, k), path)
                if path_re:
                    path_val = path_root + path_re.string
                path_re = re.search(r'TestFilled_mvi\[{}\]_KFlod\[{}\]'.format(args.method_mvi, k), path)
                if path_re:
                    path_test = path_root + path_re.string
            if path_train is None:
                continue
            train_x_filled = pd.read_csv(path_train, index_col=['index'])
            train_label = train_x_filled[[target['label1']]]
            train_x_filled.drop(columns=target['label1'], inplace=True)
            val_x_filled = pd.read_csv(path_val, index_col=['index'])
            val_label = val_x_filled[[target['label1']]]
            val_x_filled.drop(columns=target['label1'], inplace=True)
            test_x_filled = pd.read_csv(path_test, index_col=['index'])
            test_label = test_x_filled[[target['label1']]]
            test_x_filled.drop(columns=target['label1'], inplace=True)

            path_root_mask = f'../DataSet/mimic/processed_flod/mask_mr{args.ratio_manual["mr"]}/'
            index_ManualNan_test = pd.read_csv(path_root_mask + f'mask_test_ManualNan_KFlod[{k}].csv', index_col=['index'])
            index_ManualNan_train = pd.read_csv(path_root_mask + f'mask_train_ManualNan_KFlod[{k}].csv', index_col=['index'])

        MetricRec_mvi = mvi_rec_evalution(test_x, index_ManualNan_test, test_x_filled, args.method_mvi, 'test')
        metric_all_ = pd.concat([metric_all_, MetricRec_mvi], axis=1)
        MetricRec_mvi = mvi_rec_evalution(train_x, index_ManualNan_train, train_x_filled, args.method_mvi, 'train')
        metric_all_ = pd.concat([metric_all_, MetricRec_mvi], axis=1)

        if args.Deal_Imbalance_Method == 'False' or args.Deal_Imbalance_Method == 'ours':
            train_x_filled_ger, train_label_ger = train_x_filled, train_label
        else:
            imb = IMB(args.Deal_Imbalance_Method)
            train_x_filled_ger, train_label_ger = imb.fit_transform(train_x_filled, train_label)

        metric = pd.DataFrame([])
        if args.Flag_downstream:
            metric, pred_train_df, pred_test_df, imp_feat = model_tra_eval(train_x_filled_ger, train_label_ger,
                                                                           val_x_filled, val_label,
                                                                           test_x_filled, test_label,
                                                                           target, co_col, ca_col, args.task_name, nor, seed)

            pred_train_df_save = pred_train_df
            imp_feat_df_all_run = pd.concat([imp_feat_df_all_run, imp_feat], axis=1)
            pred_train_df_all = pd.concat([pred_train_df_all, pred_train_df_save], axis=1)
            pred_test_df_all = pd.concat([pred_test_df_all, pred_test_df], axis=1)

            print(f'test auc_{args.method_mvi} {metric["test auc_"].values}, mcc {metric["test mcc"].values}, val auc_ {metric["val auc_"].values}')

        metric.index = [args.method_mvi]
        metric_all_ = pd.concat([metric_all_, metric], axis=1)
        metric_df_all_run = pd.concat([metric_df_all_run, metric_all_], axis=0)
        test_label_all = pd.concat([test_label_all, test_label], axis=1)
    return metric_df_all_run, pred_train_df_all, pred_test_df_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='mimic_preprocessed_class',
                        help='mimic_preprocessed_class')
    parser.add_argument('--nrun', type=int, default=1,
                        help='total number of runs[default: 1]')
    parser.add_argument('--n_splits', type=int, default=10,
                        help='cross-validation fold, 1 refer not CV [default: 1]')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='proportion of test sets divided from training set, '
                             '0 refer dataset has its own test set [default: 0.2]')
    parser.add_argument('--val_ratio', type=float, default=0.0, # 0 if cross validation is used
                        help='proportion of test sets divided from training set [default: 0.2]')
    parser.add_argument('--method_mvi', type=str, default='ours',
                        help='missing values imputation method [default: "ours"]')
    parser.add_argument('--ratio_manual', type=dict, default={'a': 0.7, 'b': 0.6, 'c': 0.8, 'mr': 0.75},
                        help='ratio of manual missing values [default: ""]')
    parser.add_argument('--Flag_LoadMetric', type=bool, default=False, metavar='N',
                        help='overload metric training before[default: False]')
    parser.add_argument('--Flag_DataPreprocessing', type=bool, default=True, metavar='N',
                        help='[default: True]')
    parser.add_argument('--Flag_MVI', type=bool, default=True, metavar='N',
                        help='[default: True]')
    parser.add_argument('--Deal_Imbalance_Method', type=str, default='False', metavar='N',
                        help='[default: False]')
    parser.add_argument('--Flag_downstream', type=bool, default=True, metavar='N',
                        help='[default: True]')
    parser.add_argument('--Flag_Mask_Saving', type=bool, default=False, metavar='N',
                        help='[default: True]')
    parser.add_argument('--test', type=int, default=0, metavar='N',
                        help='[default: 0]')
    args = parser.parse_args()

    if args.Flag_LoadMetric:
        metric_df_all = pd.read_csv('./metric_raw_inver.csv', index_col=0)
    else:
        metric_df_all = pd.DataFrame([])
    test_prediction_all = pd.DataFrame([])
    train_prediction_all = pd.DataFrame([])
    history_df_all = pd.DataFrame([])

    for trial in range(args.nrun):
        test_prediction_all = pd.DataFrame([])
        train_prediction_all = pd.DataFrame([])
        metric_df_all = pd.DataFrame([])
        print('rnum : {}'.format(trial))
        seed = (trial * 5503) % 2022 # a different random seed for each run
        # data fetch
        # input: file path
        # output: data with DataFrame
        train_data, test_data, target = load_data.data_load(args.task_name)

        # run model
        # input: train_data
        # output: metric, test_prediction, loss history with DataFrame
        metric_df, train_prediction, test_prediction = run(train_data, test_data, target, args, seed)

        metric_df_all = pd.concat([metric_df_all, metric_df], axis=0)
        test_prediction_all = pd.concat([test_prediction_all, test_prediction], axis=1)
        train_prediction_all = pd.concat([train_prediction_all, train_prediction], axis=1)

        local_time = time.strftime("%m_%d_%H_%M", time.localtime())
        metric_df_all.to_csv(f'./metric_mvi[{args.method_mvi}]_im[{args.Deal_Imbalance_Method}]_mr[{mr}]_{args.task_name}_{local_time}.csv', index_label=['index'])
        if args.Flag_downstream:
            test_prediction_all.to_csv(f'./TestPred_mvi[{args.method_mvi}]_im[{args.Deal_Imbalance_Method}]_mr[{mr}]_{args.task_name}_{local_time}.csv', index_label=['index'])
            train_prediction_all.to_csv(f'./TrainPred_mvi[{args.method_mvi}]_im[{args.Deal_Imbalance_Method}]_mr[{mr}]_{args.task_name}_{local_time}.csv', index_label=['index'])

    # print metric
    metric_df_all['model'] = metric_df_all.index
    metric_mean = metric_df_all.groupby('model').mean()
    metric_mean_test = metric_mean.filter(regex=r'test')
    metric_mean_val = metric_mean.filter(regex=r'val')
    metric_std = metric_df_all.groupby('model').std()
    print(metric_mean)
    print('mean test auc_: ', metric_mean_test['test auc_'])
    print('mean val auc_: ', metric_mean_val['val auc_'])
    print(metric_std)
    pass
pass
