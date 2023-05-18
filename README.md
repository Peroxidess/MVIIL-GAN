# Codebase for "A Method for Missing Values Imputation and Imbalanced Data Learning in Electronic Health Record Based on Generative Adversarial Networks"

This directory contains implementations of MVIIL-GAN for missing values imputation and imbalanced learning using MIMIC-IV dataset.

Please note that you need to download the MIMIC-IV dataset to the /DataSet/mimic/ directory or modify the directory where the file is read in /preprocess/load_data.py

Simply run python3 -m main.py


## Code explanation

(1) preprocess/load_data.py

- Load data

(2) preprocess/get_dataset.py

- Data preprocessing

(3) preprocess/missing_values_imputation.py

- Imputate missing values in dataset

(4) model/mviilgan.py

- Define MVIIL-GAN

(5) model/baseline.py

- Define supervised classification models

(6) model/evaluate.py

- Performance of computation in imputation tasks and prediction tasks

(7) main.py

- Setting parameter and Report the prediction performances of entire frameworks

Note that hyper-parameters should be optimized for different datasets.


## Main Dependency Library

fancyimpute==0.7.0

imbalanced-learn==0.9.0

imblearn==0.0

keras==2.7.0

Keras-Preprocessing==1.1.2

lightgbm==3.3.2

missingno==0.5.1

scikit-learn==1.0.2

tensorflow-gpu==2.7.0

