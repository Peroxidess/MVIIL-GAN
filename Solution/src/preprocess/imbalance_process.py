from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE, RandomOverSampler
from imblearn.under_sampling import OneSidedSelection, RandomUnderSampler, ClusterCentroids
from imblearn.combine import SMOTETomek, SMOTEENN


class IMB():
    def __init__(self, name_method, seed=2022):
        self.seed = seed
        self.name_method = name_method
        self.model = self.model_build(name_method)

    def model_build(self, name_method):
        if name_method == 'SMOTE':
            model = SMOTE(random_state=self.seed)
        elif name_method == 'BorderlineSMOTE':
            model = BorderlineSMOTE(random_state=self.seed)
        elif name_method == 'ADASYN':
            model = ADASYN(random_state=self.seed)
        elif name_method == 'KMeansSMOTE':
            model = KMeansSMOTE(random_state=self.seed, cluster_balance_threshold=0.1)
        elif name_method == 'RandomOverSampler':
            model = RandomOverSampler(random_state=self.seed)
        elif name_method == 'OneSidedSelection':
            model = OneSidedSelection(random_state=self.seed)
        elif name_method == 'RandomUnderSampler':
            model = RandomUnderSampler(random_state=self.seed)
        elif name_method == 'ClusterCentroids':
            model = ClusterCentroids(random_state=self.seed)
        elif name_method == 'SMOTETomek':
            model = SMOTETomek(random_state=self.seed)
        elif name_method == 'SMOTEENN':
            model = SMOTEENN(random_state=self.seed)
        return model

    def fit_transform(self, data_x, data_y):
        X_new, Y_new = self.model.fit_resample(data_x, data_y)
        return X_new, Y_new
