import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier


class Baseline():
    def __init__(self, model_name, train_set, train_label, val_set, val_label, test_set, test_label, target,
                 co_col, ca_col, task_name, seed, param_init={}):
        self.train_set = train_set
        self.train_label = train_label
        self.val_set = val_set
        self.val_label = val_label
        self.test_set = test_set
        self.test_label = test_label
        self.target = target
        self.co_col = co_col
        self.ca_col = ca_col
        self.task_name = task_name
        self.seed = seed
        if 'class' in task_name:
            self.output_dim = len(np.unique(self.train_label.values))
        else:
            self.output_dim = 1
        self.model = self.model_bulid(model_name, param_init)
        self.param_grid = {}
        self.param_fit = {}

    @staticmethod
    def input_process(input_x, input_y):
        return input_x, input_y

    def model_bulid(self, model_name, param_init):
        model = eval(model_name)(**param_init)
        return model

    def grid_fit_pred(self):
        train_x, train_y = self.input_process(self.train_set, self.train_label)
        val_x, val_y = self.input_process(self.val_set, self.val_label)
        test_x, test_y = self.input_process(self.test_set, self.test_label)
        clf = GridSearchCV(self.model, self.param_grid)
        clf.fit(train_x, train_y[self.target['label1']], **self.param_fit)
        print('Best parameters found by grid search are:', clf.best_params_)
        self.model = clf.best_estimator_
        self.model.fit(train_x, train_y[self.target['label1']], **self.param_fit)
        pred_tra = self.model.predict(train_x).reshape(-1, 1)
        pred_val = self.model.predict(val_x).reshape(-1, 1)
        pred_test = self.model.predict(test_x).reshape(-1, 1)
        pred_tra_df = pd.DataFrame(pred_tra, index=self.train_label.index, columns=[self.target['label1']])
        pred_val_df = pd.DataFrame(pred_val, index=self.val_label.index, columns=[self.target['label1']])
        pred_test_df = pd.DataFrame(pred_test, index=self.test_label.index, columns=[self.target['label1']])
        return pred_tra_df, pred_val_df, pred_test_df, self.model

    def imp_feat(self):
        try:
            feat_dict = dict(zip(self.train_set.columns, self.model.coef_))
        except:
            feat_dict = dict(zip(self.train_set.columns, self.model.feature_importances_))
        else:
            pass
        return pd.DataFrame.from_dict([feat_dict], orient='columns').T.sort_values(by=0, ascending=False)


class GBM(Baseline):
    def __init__(self, model, train_set, train_label, val_set, val_label, test_set, test_label, target, co_col, ca_col, task_name, seed, param_init={}):
        Baseline.__init__(self, model, train_set, train_label, val_set, val_label, test_set, test_label, target, co_col, ca_col, task_name, seed,
                          param_init)
        self.param_grid = {
            'learning_rate':  np.arange(0.2, 0.3, 0.1),
            'max_depth': range(3, 4, 1),
            'min_child_samples': range(32, 33, 1),
            'feature_fraction': np.arange(0.3, 0.4, 0.1),
            'bagging_fraction': np.arange(0.2, 0.3, 0.1),
            'bagging_freq': np.arange(1, 2, 1),
            'reg_alpha': np.arange(2e-1, 3e-1, 1e-1),
            'reg_lambda': np.arange(1e-1, 2e-1, 1e-1),
            'n_estimators': range(150, 250, 100),
            'max_bin': range(2, 3, 1),
            'min_data_in_leaf': range(20, 21, 1)
        }
        self.param_fit = {
                        # 'categorical_feature': self.ca_col,
                        'eval_metric': 'auc',
                        'verbose': 2
                        }

    def model_bulid(self, model_name, param_init):
        param_init = {
                'boosting_type': 'gbdt',
                'objective': 'cross_entropy',
                # 'metric': 'l1',
                'is_unbalance': False,
                'early_stopping_rounds': 20,
                'random_state': self.seed,
                'n_jobs': -1,
                }
        model = eval(model_name)(**param_init)
        return model

    def input_process(self, input_x, input_y, state=None):
        input_y_list = []
        for col, col_value in input_y.iteritems():
            input_y_list.append(col_value)
        return input_x, input_y_list

    def grid_fit_pred(self):
        train_x, train_y = self.input_process(self.train_set, self.train_label.iloc[:, 0])
        val_x, val_y = self.input_process(self.val_set, self.val_label.iloc[:, 0])
        test_x, test_y = self.input_process(self.test_set, self.test_label.iloc[:, 0])
        clf = GridSearchCV(self.model, self.param_grid)
        clf.fit(train_x, train_y, eval_set=(val_x, val_y), **self.param_fit)
        print('Best parameters found by grid search are:', clf.best_params_)
        self.model = clf.best_estimator_
        self.model.fit(train_x, train_y, eval_set=(val_x, val_y), **self.param_fit)
        pred_tra = self.model.predict_proba(train_x)[:, 1]
        pred_val = self.model.predict_proba(val_x)[:, 1]
        pred_test = self.model.predict_proba(test_x)[:, 1]
        pred_tra_df = pd.DataFrame(pred_tra, index=self.train_label.index, columns=[self.target['label1']])
        pred_val_df = pd.DataFrame(pred_val, index=self.val_label.index, columns=[self.target['label1']])
        pred_test_df = pd.DataFrame(pred_test, index=self.test_label.index, columns=[self.target['label1']])
        imp_feat = self.imp_feat()
        return pred_tra_df, pred_val_df, pred_test_df, self.model

    def imp_feat(self):
        return pd.DataFrame.from_dict([dict(zip(self.train_set.columns, self.model.feature_importances_))],
                                      orient='columns').T.sort_values(by=0)
