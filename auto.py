import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from explore2 import compute_idf_transform

np.random.seed(0)

data_path = 'data/'

data_train=np.load(os.path.join(data_path, 'data_train.npy'), allow_pickle=True)
vocab_map=np.load(os.path.join(data_path, 'vocab_map.npy'), allow_pickle=True)
data_test=np.load(os.path.join(data_path, 'data_test.npy'), allow_pickle=True)
label_train=np.loadtxt(os.path.join(data_path, 'label_train.csv'), delimiter=',', skiprows=1).astype(int)

idf_train,idf_t_train=compute_idf_transform(data_train)
idf_test,idf_t_test=compute_idf_transform(data_test)

import pandas as pd
from supervised.automl import AutoML
import psutil
import os
jobs = psutil.cpu_count(logical=False)

final_data_train=idf_train
final_data_test=idf_test

X_train=pd.DataFrame(final_data_train, columns=vocab_map)
X_test=pd.DataFrame( final_data_test, columns=vocab_map)
y_train=pd.DataFrame(label_train[:,1], columns=['target'])

models=["Baseline",
"Linear",
"Decision Tree",
"Random Forest",
"Extra Trees",
"LightGBM",
"Xgboost",
"CatBoost",
"Neural Network",
"Nearest Neighbors",]

val={
    "validation_type": "split",
    "train_ratio": 0.75,
    "shuffle": True,
    "stratify": True
}
init=dict(results_path=None,
        total_time_limit=3600*5,#3600,
        mode='Compete',#'Explain',
        ml_task="binary_classification",#'auto',
        model_time_limit=3600,
        algorithms=models,#'auto',
        train_ensemble=True,
        stack_models=True,#'auto',
        eval_metric='f1',#'auto',
        validation_strategy=val,#'auto',
        explain_level=0,#'auto',
        golden_features=True,#'auto',
        features_selection=True,#'auto',
        start_random_models=5,#'auto',
        hill_climbing_steps=2,#'auto',
        top_models_to_improve=3,#'auto',
        boost_on_errors=True,#'auto',
        kmeans_features=True,#'auto',
        mix_encoding=True,#'auto',
        max_single_prediction_time=None,#,
        optuna_time_budget=None,
        optuna_init_params={},
        optuna_verbose=True,
        fairness_metric='auto',
        fairness_threshold='auto',
        privileged_groups='auto',
        underprivileged_groups='auto',
        n_jobs=jobs,
        verbose=1,
        random_state=1234)
automl = AutoML(**init)
#automl=AutoML()
automl.fit(X_train, y_train)

predictions = automl.predict(X_test)

# Save the predictions to a CSV file
output = pd.DataFrame({'ID': range(len(predictions)), 'label': predictions})
output.to_csv(os.path.join(automl._get_results_path(),'submission.csv'), index=False)