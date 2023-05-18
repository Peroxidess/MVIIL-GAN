from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np


class DataPreprocessing():
    def __init__(self, train_data, val_data=pd.DataFrame([]), test_data=pd.DataFrame([]),
                 target=None, ca_feat_th=8, task_name='reg', seed=2022,
                 flag_label_onehot=False,
                 flag_ex_null=False, flag_ex_std_flag=False, flag_ex_occ=False,
                 flag_ca_co_sel=True, flag_ca_fac=True, flag_onehot=True, flag_nor=True,
                 flag_feat_emb=False, flag_RUS=False, flaq_save=False):

        self.target = target
        if not target:
            self.col_label = train_data.filter(regex=r'label').columns.tolist()
        else:
            self.col_label = list(target.values())

        self.train_data = self.drop_labelnan(train_data)
        self.val_data = self.drop_labelnan(val_data)
        self.test_data = self.drop_labelnan(test_data)
        self.data_all = None

        self.ca_feat_th = ca_feat_th
        self.task_name = task_name
        self.seed = seed
        self.flag_label_onehot = flag_label_onehot
        self.flag_ex_null = flag_ex_null
        self.flag_ex_std_flag = flag_ex_std_flag
        self.flag_ex_occ = flag_ex_occ
        self.flag_ca_co_sel = flag_ca_co_sel
        self.flag_ca_fac = flag_ca_fac
        self.flag_onehot = flag_onehot
        self.flag_nor = flag_nor
        self.flag_feat_emb = flag_feat_emb
        self.flag_RUS = flag_RUS
        self.flaq_save = flaq_save

    def drop_labelnan(self, data):
        data_ = pd.DataFrame([])
        if not data.empty:
            data_ = data.dropna(subset=list(self.target.values()))
        return data_

    def features_ex(self, data, drop_NanRatio=0.95):
        col_drop = []
        # Exclude features with too many null values
        col_null = {}
        if self.flag_ex_null:
            data_null_count = \
                data.isnull().sum() / data.shape[0]
            col_null = data_null_count[data_null_count > drop_NanRatio].index

        # Exclude features whose standard deviation is too small
        col_std = {}
        if self.flag_ex_std_flag:
            # std = StandardScaler()
            # data_ = pd.DataFrame(std.fit_transform(data), index=data.index, columns=data.columns)
            data_des = data.describe()
            col_std = data_des.columns[data_des.loc['std'] < 0.1]
            for col_ in self.col_label:
                try:
                    col_std.remove(col_)
                except:
                    pass

        # Exclude features that have too much of a certain value
        col_occ_r = {}
        data_temp = data.fillna(-1)
        if self.flag_ex_occ:
            dict_occ_r = {}
            for col_ in data_temp.columns:
                val_count = data_temp[col_].value_counts()
                occ_r = val_count.iloc[0] / data_temp.shape[0]
                dict_occ_r[col_] = occ_r
            df_occ_r = pd.DataFrame.from_dict(dict_occ_r, orient='index').sort_values(by=0)
            col_occ_r = df_occ_r[df_occ_r[0] > 0.95].index
        if self.flag_ex_null or self.flag_ex_std_flag or self.flag_ex_occ:
            col_drop = set(col_occ_r) | set(col_null) | set(col_std)
            col_drop = col_drop - set(self.col_label) - set('tab')
        return col_drop

    def ca_co_sel(self, data):
        if self.flag_ca_co_sel:
            dict_col_rename = {}
            ca_col = []
            co_col = []
            col_to2excluded = self.col_label + ['tab']
            data_columns = data.drop(columns=col_to2excluded)
            for col in data_columns.columns:
                data_col = data[col]
                col_feat_num = len(set(data_col))
                if col_feat_num > self.ca_feat_th and (data[col].dtypes == 'float64' or data[col].dtypes == 'int64'):
                    col_ = col + '_dense'
                    co_col.append(col_)
                elif self.ca_feat_th >= col_feat_num > 1 or data[col].dtypes == 'object':
                    col_ = col + '_sparse'
                    ca_col.append(col_)
                else:
                    col_ = col
                    print('ca co can`t handle', data[col].dtype, col)
                dict_col_rename[col] = col_
        else:
            ca_col = data.filter(regex=r'sparse').columns.tolist()
            co_col = data.filter(regex=r'dense').columns.tolist()
        return ca_col, co_col, dict_col_rename

    def feat_emb(self, ca_col, co_col):
        emb_col = ca_col
        feat_dict = {}
        tc = 0
        for col in emb_col:
            if col in co_col:
                feat_dict[col] = {0: tc}
                tc += 1
            elif col in ca_col:
                us = self.data_all[col].unique()
            feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
            tc += len(us)
        feat_dim = tc
        return feat_dict, feat_dim

    def process(self):
        if not self.test_data.empty:
            self.test_data['tab'] = 2
        if not self.val_data.empty:
            self.val_data['tab'] = 1
        self.train_data['tab'] = 0
        self.data_all = pd.concat([self.train_data, self.val_data, self.test_data], axis=0)

        col_drop = self.features_ex(self.data_all[self.data_all['tab'] == 0], drop_NanRatio=0.95)
        self.data_all.drop(columns=col_drop, inplace=True)
        if self.flaq_save:
            self.data_all.drop(columns=['tab']).to_csv('./data_preprocessed_row.csv', encoding='gb18030', index_label='subject_id')
        if self.data_all[self.col_label].dtypes[0] == 'object':
            self.data_all[self.col_label] = self.data_all[self.col_label].apply(LabelEncoder().fit_transform)

        ca_col = self.data_all.columns
        co_col = self.data_all.columns
        if self.flag_ca_co_sel:
            ca_col, co_col, dict_col_rename = self.ca_co_sel(self.data_all)
            self.data_all.rename(columns=dict_col_rename, inplace=True)

        if self.flag_ca_fac:
            col_fac = self.data_all.select_dtypes('object').columns
            self.data_all[col_fac] = pd.concat([self.data_all[col_fac].apply(lambda ser: pd.factorize(ser, sort=True)[0])])
            self.data_all[col_fac] = self.data_all[col_fac].replace({-1: np.nan})

        if self.flag_onehot:
            self.data_all = pd.get_dummies(self.data_all, columns=ca_col, dummy_na=False)
            for ca_col_ in ca_col:
                ca_col_dum = self.data_all.filter(regex=f'{ca_col_}').columns
                data_temp = self.data_all[ca_col_dum].sum(axis=1)
                index2nan = self.data_all[data_temp == 0].index
                if not index2nan.empty:
                    self.data_all.loc[index2nan, ca_col_dum] = np.nan
            ca_col = self.data_all.filter(regex=r'sparse').columns.tolist()

        self.train_data = self.data_all[self.data_all['tab'] == 0]
        self.val_data = self.data_all[self.data_all['tab'] == 1]
        self.test_data = self.data_all[self.data_all['tab'] == 2]
        nor = None
        if self.flag_nor:
            mms = MinMaxScaler(feature_range=(0, 1))
            std = StandardScaler()
            nor = std
            if len(co_col) > 0:
                self.train_data[co_col] = pd.DataFrame(nor.fit_transform(self.train_data[co_col]), columns=co_col, index=self.train_data.index)
                self.val_data[co_col] = pd.DataFrame(nor.transform(self.val_data[co_col]), columns=co_col, index=self.val_data.index)
                self.test_data[co_col] = pd.DataFrame(nor.transform(self.test_data[co_col]), columns=co_col, index=self.test_data.index)
        if self.flag_RUS:
            set_label = self.train_data[self.target['label1']].unique()
            data_less_num = self.train_data.shape[0]
            list_data_sample = []
            for label_ in set_label:
                data_ = self.train_data.loc[self.train_data[self.target['label1']] == label_]
                print(f'class {label_}: {data_.shape[0]}')
                if data_.shape[0] < data_less_num:
                    data_less_num = data_.shape[0]
            for label_ in set_label:
                data_ = self.train_data.loc[self.train_data[self.target['label1']] == label_]
                data_sample = data_.sample(n=data_less_num, random_state=self.seed)
                list_data_sample.append(data_sample)
            self.train_data = pd.concat(list_data_sample).sample(frac=1, random_state=self.seed)
        if self.flaq_save:
            self.data_all = pd.concat([self.train_data, self.val_data, self.test_data], axis=0)
            self.data_all.to_csv('./data_preprocessed.csv', index=False, encoding='gb18030', index_label='subject_id')
        self.train_data.drop(columns=['tab'], inplace=True)
        self.val_data.drop(columns=['tab'], inplace=True)
        self.test_data.drop(columns=['tab'], inplace=True)
        col_all = ca_col + co_col + self.col_label
        return self.train_data[col_all], self.val_data[col_all], self.test_data[col_all], ca_col, co_col, nor
    pass
