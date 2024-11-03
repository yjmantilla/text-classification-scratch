nsel=r'Y:\code\text-classification-scratch\v22oct\binary_counts_class_sorted_nsel.csv'
sel=r'Y:\code\text-classification-scratch\v22oct\binary_counts_class_sorted_sel.csv'
all_=r'Y:\code\text-classification-scratch\v22oct\binary_counts_class_sorted.csv'
unsorted=r'Y:\code\text-classification-scratch\v22oct\binary_counts_class.csv'

import pandas as pd

df_nsel = pd.read_csv(nsel)
df_sel = pd.read_csv(sel)
df_sorted = pd.read_csv(all_)

df_sel['count1@False-fix']=df_sel['count1@False']
df_sel['count1@False-fix']=df_sel['count1@False-fix'].apply(lambda x: x if x>0 else 0.1)
df_sel['ratio1TF-fix']=df_sel['count1@True']/df_sel['count1@False-fix']
df_sel.sort_values(by='ratio1TF-fix',ascending=False,inplace=True)
df_sel.to_csv(r'Y:\code\text-classification-scratch\v22oct\binary_counts_class_sorted_sel_fixed.csv',index=False)


df = pd.read_csv(unsorted,na_filter=False)

df['feature']=df['feature'].astype(str)
presence_certain_true = df[df['count1@False']==0].sort_values(by='count1@True',ascending=False)
presence_certain_false = df[df['count1@True']==0].sort_values(by='count1@False',ascending=False)
presence_certain_true.to_csv(r'Y:\code\text-classification-scratch\v22oct\binary_counts_class_sorted_presence_certain_true.csv',index=False)
presence_certain_false.to_csv(r'Y:\code\text-classification-scratch\v22oct\binary_counts_class_sorted_presence_certain_false.csv',index=False)
ZERO=0.001

df['count0@False'] =df['count0@False'].apply(lambda x: x if x>0 else ZERO)
df['count1@False'] =df['count1@False'].apply(lambda x: x if x>0 else ZERO)
df['count0@True']  =df['count0@True'].apply (lambda x: x if x>0 else ZERO)
df['count1@True']  =df['count1@True'].apply (lambda x: x if x>0 else ZERO)
df['PresenceTrue'] =df['count1@True'] /df['count1@False']
df['PresenceFalse']=df['count1@False']/df['count1@True']
df['AbsenceTrue']  =df['count0@True'] /df['count0@False']
df['AbsenceFalse'] =df['count0@False']/df['count0@True']

THR_TRUE=2
THR_FALSE=4
presence_uncertain_true = df[(df['PresenceTrue']<100) & (df['PresenceTrue']>THR_TRUE)].sort_values(by='PresenceTrue',ascending=False)
presence_uncertain_false = df[(df['PresenceFalse']<100) & (df['PresenceFalse']>THR_FALSE)].sort_values(by='PresenceFalse',ascending=False)
presence_uncertain_true.to_csv(r'Y:\code\text-classification-scratch\v22oct\binary_counts_class_sorted_presence_uncertain_true.csv',index=False)
presence_uncertain_false.to_csv(r'Y:\code\text-classification-scratch\v22oct\binary_counts_class_sorted_presence_uncertain_false.csv',index=False)

NTRUE=2298
NFALSE=7124

NTRUE/NFALSE
NFALSE/NTRUE

useful=presence_uncertain_true['feature'].tolist()+presence_uncertain_false['feature'].tolist()+presence_certain_true['feature'].tolist()+presence_certain_false['feature'].tolist()
df_useful=df[df['feature'].isin(useful)].sort_values(by=['PresenceTrue','PresenceFalse'],ascending=[False,True])
df_useless=df[~df['feature'].isin(useful)]
df_useful.to_csv(r'Y:\code\text-classification-scratch\v22oct\binary_counts_class_sorted_useful.csv',index=False)
df_useless.to_csv(r'Y:\code\text-classification-scratch\v22oct\binary_counts_class_sorted_useless.csv',index=False)
df.to_csv(r'Y:\code\text-classification-scratch\v22oct\binary_counts_class_sorted_fixed.csv',index=False)

import os
import numpy as np
data_path = 'data/'
data_train=np.load(os.path.join(data_path, 'data_train.npy'), allow_pickle=True)
vocab_map=np.load(os.path.join(data_path, 'vocab_map.npy'), allow_pickle=True)
data_test=np.load(os.path.join(data_path, 'data_test.npy'), allow_pickle=True)
label_train=np.loadtxt(os.path.join(data_path, 'label_train.csv'), delimiter=',', skiprows=1).astype(int)

presence_certain_true_idx=[vocab_map.tolist().index(x) for x in presence_certain_true['feature'].tolist()]
presence_certain_false_idx=[vocab_map.tolist().index(x) for x in presence_certain_false['feature'].tolist()]
presence_uncertain_true_idx=[vocab_map.tolist().index(x) for x in presence_uncertain_true['feature'].tolist()]
presence_uncertain_false_idx=[vocab_map.tolist().index(x) for x in presence_uncertain_false['feature'].tolist()]

useful_idx=[vocab_map.tolist().index(x) for x in df_useful['feature'].tolist()]
useless_idx=[vocab_map.tolist().index(x) for x in df_useless['feature'].tolist()]

data_train_presence_certain_true=data_train[:,presence_certain_true_idx].sum(axis=1)/len(presence_certain_true_idx)
data_train_presence_certain_false=data_train[:,presence_certain_false_idx].sum(axis=1)/len(presence_certain_false_idx)
data_train_presence_uncertain_true=data_train[:,presence_uncertain_true_idx].sum(axis=1)/len(presence_uncertain_true_idx)
data_train_presence_uncertain_false=data_train[:,presence_uncertain_false_idx].sum(axis=1)/len(presence_uncertain_false_idx)

data_test_presence_certain_true=data_test[:,presence_certain_true_idx].sum(axis=1)/len(presence_certain_true_idx)
data_test_presence_certain_false=data_test[:,presence_certain_false_idx].sum(axis=1)/len(presence_certain_false_idx)
data_test_presence_uncertain_true=data_test[:,presence_uncertain_true_idx].sum(axis=1)/len(presence_uncertain_true_idx)
data_test_presence_uncertain_false=data_test[:,presence_uncertain_false_idx].sum(axis=1)/len(presence_uncertain_false_idx)


new_data_train=np.column_stack((data_train_presence_certain_true,data_train_presence_certain_false,data_train_presence_uncertain_true,data_train_presence_uncertain_false))
new_data_test=np.column_stack((data_test_presence_certain_true,data_test_presence_certain_false,data_test_presence_uncertain_true,data_test_presence_uncertain_false))
df_train=pd.DataFrame(new_data_train,columns=['presence_certain_true','presence_certain_false','presence_uncertain_true','presence_uncertain_false'])
df_submit=pd.DataFrame(new_data_test,columns=['presence_certain_true','presence_certain_false','presence_uncertain_true','presence_uncertain_false'])

#normalize each column

df_all=pd.concat([df_train,df_submit])
# df_train=(df_train-df_all.min())/(df_all.max()-df_all.min())
# df_submit=(df_submit-df_all.min())/(df_all.max()-df_all.min())

# Or standardize each column
df_train=(df_train-df_all.mean())/df_all.std()
df_submit=(df_submit-df_all.mean())/df_all.std()

df_train['label']=label_train[:,1]
# drop=['presence_uncertain_true','presence_uncertain_false']
# df_train.drop(drop,axis=1,inplace=True)
# df_submit.drop(drop,axis=1,inplace=True)

# make boxplot of the features against the label
import matplotlib.pyplot as plt
import seaborn as sns

from autogluon.tabular import TabularDataset, TabularPredictor,FeatureMetadata
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

from explore2 import balanced_split,stratified_split

trainset, testset,valset = balanced_split(None,label_train[:,1],0.7)
falseset=set(range(len(label_train)))-set(trainset)-set(testset)-set(valset)
falseset=list(falseset)
X_train=df_train.iloc[trainset]
X_test=df_train.iloc[testset]
X_val=df_train.iloc[valset]
X_false=df_train.iloc[falseset]
X_submit=df_submit

scracth=True

if scracth:
    os.makedirs('scratch',exist_ok=True)
    import datetime
    datestr=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outputpath=f'scratch/{datestr}'
    os.makedirs(outputpath,exist_ok=True)
    from explore2 import LogisticRegression,LinearRegression,Perceptron,SVM,opt,test_model
    import joblib
    ensemble=[]
    instances_per_model=5
    test_errors_per_instance=[]
    X_train_=X_train.drop(['label'],axis=1).to_numpy()
    #add bias
    X_train_=np.column_stack((X_train_,np.ones(X_train_.shape[0])))
    labels_train_sign=(X_train['label']*2-1).to_numpy()

    X_test_=X_test.drop(['label'],axis=1).to_numpy()
    X_test_=np.column_stack((X_test_,np.ones(X_test_.shape[0])))
    labels_test_sign=(X_test['label']*2-1).to_numpy()

    X_val_=X_val.drop(['label'],axis=1).to_numpy()
    X_val_=np.column_stack((X_val_,np.ones(X_val_.shape[0])))
    labels_val_sign=(X_val['label']*2-1).to_numpy()

    X_valtest_=np.concatenate((X_val_,X_test_))
    labels_valtest_sign=np.concatenate((labels_val_sign,labels_test_sign))

    X_submit_=X_submit.to_numpy()
    X_submit_=np.column_stack((X_submit_,np.ones(X_submit_.shape[0])))

    X_false_=X_false.drop(['label'],axis=1).to_numpy()
    X_false_=np.column_stack((X_false_,np.ones(X_false_.shape[0])))
    labels_false_sign=(X_false['label']*2-1).to_numpy()

    for model_,mod in zip([LogisticRegression,LinearRegression,Perceptron,SVM],['LogR','LR','Per','SVM']):
        regs = [0.001,0.01, 0.1, 1, 10]
        stepsizes = [0.01, 0.1, 1, 10,100]
        iters = [30,50, 100,500]

        if not os.path.exists(f'best_args_{mod}.npy'):
            best_args, best_error,results,combs = opt(model_, regs, stepsizes, iters,X_val_,labels_val_sign,X_train_,labels_train_sign)
            reg=best_args[0]
            stepsize=best_args[1]
            iters=best_args[2]
            np.save(os.path.join('scratch',f'best_args_{mod}.npy'), best_args)
            np.save(os.path.join('scratch',f'best_error_{mod}.npy'), best_error)
            np.save(os.path.join('scratch',f'results_{mod}.npy'), [r for x,r in results])
            np.save(os.path.join('scratch',f'combs_{mod}.npy'), combs)

        best_error=1#np.load(f'best_error_{mod}.npy')
        #best_args=np.load(f'best_args_{mod}.npy')
        results=np.load(os.path.join('scratch',f'results_{mod}.npy'),allow_pickle=True)
        combs=np.load(os.path.join('scratch',f'combs_{mod}.npy'),allow_pickle=True)

        for args, error in zip(combs, results):
            if error < best_error and error!=0:
                best_args = args
                best_error = error

        reg=best_args[0]
        stepsize=best_args[1]
        iters=best_args[2].astype(int)
        print('Best error:', best_error, 'Best args:', best_args)

        args=dict(verbose=True,reg=reg,stepsize=stepsize,iters=iters,trainset=X_train_,trainlabel=labels_train_sign,testset=X_valtest_,testlabel=labels_valtest_sign)


        print(mod)
        def foof(i):
            model_2,error_2=test_model(model_, **args)
            return model_2,error_2
        
        # parallelize
        results = joblib.Parallel(n_jobs=-1)(joblib.delayed(foof)(i) for i in range(instances_per_model))
        for i,r in enumerate(results):
            model_2,error_2=r
            ensemble.append(model_2)
            test_errors_per_instance.append(error_2)
            preds=model_2.predict(X_submit_)
            preds[preds>=0]=1
            preds[preds<0]=0

            # Question 2

            df = pd.DataFrame({'ID': np.arange(len(preds)), 'label': preds.astype(int)})
            df.to_csv(os.path.join(outputpath,f'submission_{mod}_{i}.csv'), index=False)

            preds=model_2.predict(X_valtest_)
            preds[preds>=0]=1
            preds[preds<0]=0
            error_rate=f'{np.mean(preds!=(labels_valtest_sign>0).astype(int)).round(2)}'
            print('Error rate:', error_rate)
            df = pd.DataFrame({'ID': np.arange(len(preds)), 'label': preds.astype(int)})
            df.to_csv(os.path.join(outputpath,f'known_{mod}_{i}_{error_rate}.csv'), index=False)
            df = pd.DataFrame({'ID': testset+valset, 'label': (labels_valtest_sign>0).astype(int)})
            df.to_csv(os.path.join(outputpath,f'known_{mod}_{i}_true.csv'), index=False)

            preds_false=model_2.predict(X_false_)
            preds_false[preds_false>=0]=1
            preds_false[preds_false<0]=0
            error_rate=f'{np.mean(preds_false!=(labels_false_sign>0).astype(int)).round(2)}'
            print('Error rate:', error_rate)
            df = pd.DataFrame({'ID': falseset, 'label': (labels_false_sign>0).astype(int)})
            df.to_csv(os.path.join(outputpath,f'known_{mod}_{i}_false.csv'), index=False)

    # load predictions from the models
    import numpy as np
    import pandas as pd
    import glob
    results=glob.glob(os.path.join(outputpath,'submission_*_*.csv'))
    answers=glob.glob(os.path.join(outputpath,'known*true.csv'))
    checks=glob.glob (os.path.join(outputpath,'known*'))
    checks=[x for x in checks if 'true' not in x]

    models=[x.split('_')[1] for x in results]
    answers=[pd.read_csv(x).drop(['ID'],axis=1) for x in answers]
    checks=[pd.read_csv(x).drop(['ID'],axis=1) for x in checks]
    results=[pd.read_csv(x) for x in results]

    # filter out Per

    answers=[x for x,y in zip(answers,models) if y!='Per']
    checks=[x for x,y in zip(checks,models) if y!='Per']
    results=[x for x,y in zip(results,models) if y!='Per']

    #ALL answers should be the same

    answer_array=(pd.concat(answers,axis=1).to_numpy().sum(axis=1)/len(answers)).astype(int)

    # vote on checks predictions

    check_array=(pd.concat(checks,axis=1).to_numpy().sum(axis=1)/len(checks))
    check_array[check_array>=0.5]=1
    check_array[check_array<0.5]=0

    error_rate=np.mean(check_array!=answer_array)

    # check if the predictions are correct


    preds_submission=np.array([x['label'].values for x in results])

    # vote for submission

    preds=np.mean(preds_submission,axis=0)
    preds[preds>=0.5]=1
    preds[preds<0.5]=0
    preds=preds.astype(int)

    df = pd.DataFrame({'ID': np.arange(len(preds)), 'label': preds})
    df.to_csv(os.path.join(outputpath,f'submission_ensemble.csv'), index=False)



gluon=False
if gluon:
    idf=False
    dataset=TabularDataset(X_train)
    if idf:
        t='float'
    else:
        t='int'
    type_map={col: t for col in X_train.columns}
    eval_metric='f1_macro'
    metadata=FeatureMetadata(type_map_raw=type_map)
    generator=AutoMLPipelineFeatureGenerator(pre_enforce_types=False,enable_categorical_features = False,feature_metadata_in=metadata)
    predictor=TabularPredictor(label='label', eval_metric=eval_metric,problem_type='binary',learner_kwargs={'positive_class':1})

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i, col in enumerate(df_train.columns[:-1]):
        sns.boxplot(x='label', y=col, data=df_train, ax=ax[i//2, i%2])
        ax[i//2, i%2].set_title(col)

    plt.savefig(os.path.join(predictor.path,'boxplot.png'))
    extra_metrics=['balanced_accuracy','accuracy','f1', 'f1_macro', 'f1_micro', 'f1_weighted']
    predictor.fit(train_data=dataset,presets='medium_quality',feature_generator=generator,calibrate_decision_threshold=True)#,feature_metadata=metadata)
    predictor.leaderboard(X_train,extra_metrics=extra_metrics).to_csv(os.path.join(predictor.path,'leaderboard_train.csv'))

    predictor.calibrate_decision_threshold(X_val,metric=eval_metric)
    predictor.evaluate(X_test)
    predictor.leaderboard(X_test,extra_metrics=extra_metrics).to_csv(os.path.join(predictor.path,'leaderboard_test.csv'))

    predictor.calibrate_decision_threshold(X_test,metric=eval_metric)
    predictor.evaluate(X_val)
    predictor.leaderboard(X_val,extra_metrics=extra_metrics).to_csv(os.path.join(predictor.path,'leaderboard_val.csv'))

    predictor.calibrate_decision_threshold(pd.concat([X_val,X_test]),metric=eval_metric)
    predictor.evaluate(X_false)
    predictor.leaderboard(X_false,extra_metrics=extra_metrics).to_csv(os.path.join(predictor.path,'leaderboard_false.csv'))

    predictor.calibrate_decision_threshold(X_false,metric=eval_metric)
    predictor.evaluate(pd.concat([X_test,X_val]))
    predictor.leaderboard(pd.concat([X_test,X_val]),extra_metrics=extra_metrics).to_csv(os.path.join(predictor.path,'leaderboard_testval.csv'))

    predictor.calibrate_decision_threshold(X_train,metric=eval_metric)
    X_non_train=pd.concat([X_test,X_val,X_false])
    predictor.leaderboard(X_non_train,extra_metrics=extra_metrics).to_csv(os.path.join(predictor.path,'leaderboard_non_train.csv'))

    X_all=pd.concat([X_train,X_non_train])
    predictor.calibrate_decision_threshold(X_all,metric=eval_metric)
    predictor.evaluate(X_all)
    predictor.leaderboard(X_all,extra_metrics=extra_metrics).to_csv(os.path.join(predictor.path,'leaderboard_all.csv'))

    predictions = predictor.predict(X_submit).values
    probas=predictor.predict_proba(X_submit).values
    X_false['label'].value_counts()
    np.unique(predictions,return_counts=True)

    # Save the predictions to a CSV file
    output = pd.DataFrame({'ID': range(len(predictions)), 'label': predictions})

    path=predictor.path
    output.to_csv(os.path.join(path,'submission_autogluon.csv'), index=False)
    # Save the probabilities to a CSV file
    output = pd.DataFrame(probas, columns=['proba0', 'proba1'])
    output.to_csv(os.path.join(path,'submission_autogluon_proba.csv'), index=False)

automl=False
if automl:
    from supervised.automl import AutoML
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split,KFold
    folder=StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    cv=list(folder.split(X_train.drop(['label'],axis=1), X_train['label']))

    models=["Baseline",
    "Linear",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "LightGBM",
    "Xgboost",
    "CatBoost",
    "Neural Network",
    "Nearest Neighbors",
    ]

    val={'validation_type': 'custom'}



    init=dict(results_path=None,
            total_time_limit=3600*5,#3600,
            mode='Explain',#'Explain',
            ml_task="binary_classification",#'auto',
            model_time_limit=3600,
            algorithms=models,#'auto',
            train_ensemble=True,
            stack_models=True,#'auto',
            eval_metric='average_precision',#'auto',
            validation_strategy=val,#'auto',
            explain_level=2,#'auto',
            golden_features=False,#'auto',
            features_selection=False,#'auto',
            start_random_models='auto',
            hill_climbing_steps='auto',
            top_models_to_improve='auto',
            boost_on_errors=False,#'auto',
            kmeans_features=False,#'auto',
            mix_encoding=False,#'auto',
            max_single_prediction_time=None,#,
            optuna_time_budget=None,
            optuna_init_params={},
            optuna_verbose=True,
            fairness_metric='auto',
            fairness_threshold='auto',
            privileged_groups='auto',
            underprivileged_groups='auto',
            n_jobs=-1,
            verbose=1,
            random_state=1234)
    automl = AutoML(**init)
    #automl=AutoML()
    automl.fit(X_train.drop(['label'],axis=1),X_train['label'],cv=cv)

    predictions = automl.predict(X_submit)

    # Use another model
    # https://github.com/mljar/mljar-supervised/issues/423
    # from supervised.model_framework import ModelFramework
    # model = ModelFramework.load("AUTOML_FOLDER_NAME_HERE", model_subpath="MODEL_FOLDER_NAME_HERE")
    # y_pred = automl._base_predict(qdqd_X_test, model=model)["prediction"].to_numpy()
    # y_pred = automl._base_predict(qdqd_X_test, model=model)["label"].to_numpy()

    path=automl._get_results_path()
    output = pd.DataFrame({'ID': range(len(predictions)), 'label': predictions})
    output.to_csv(os.path.join(path,'submission_mljar.csv'), index=False)