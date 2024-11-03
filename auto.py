import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from explore2 import compute_idf_transform
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split,KFold

import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Initialize SpaCy's English model outside the function for efficiency
nlp = spacy.load("en_core_web_sm")
stop_words = set(ENGLISH_STOP_WORDS)

def create_word_mapping(vocab_df):
    word_map = {}
    for word in vocab_df['word']:
        # Convert numbers to a single token
        if word.isdigit():
            word_map[word] = 'NUM'
        # Remove alphanumeric or non-informative tokens
        elif not word.isalpha() or word.lower() in stop_words:
            word_map[word] = None
        else:
            # Lemmatize using SpaCy
            doc = nlp(word.lower())
            lemmatized_word = doc[0].lemma_
            word_map[word] = lemmatized_word
    return word_map

def transform_matrix_with_vocab_map(X, vocab_df, new_vocab=None):
    """
    Transforms a document-term matrix by merging and filtering features based on a vocabulary map.
    
    Parameters:
    - X (np.ndarray): Original document-term matrix with shape (n_docs, n_features).
    - vocab_df (pd.DataFrame): DataFrame containing the vocabulary with a single column 'word'.
    - new_vocab (list, optional): List of vocabulary terms to use for the transformation.
                                  If provided, skips vocabulary generation and uses this directly.
    
    Returns:
    - new_X (np.ndarray): Transformed document-term matrix with merged features.
    - new_vocab (list): List of new vocabulary terms corresponding to the columns in new_X.
    """
    # Step 1: Generate or use the provided vocabulary mapping
    if new_vocab is None:
        # Generate the mapping based on the vocabulary
        word_map = create_word_mapping(vocab_df)

        # Map each column in X to the transformed vocabulary
        new_vocab_map = {}
        for i, word in enumerate(vocab_df['word']):
            transformed_word = word_map[word]
            if transformed_word:
                if transformed_word in new_vocab_map:
                    new_vocab_map[transformed_word].append(i)
                else:
                    new_vocab_map[transformed_word] = [i]
    else:
        # Directly use provided new_vocab to map columns
        new_vocab_map = {word: [i] for i, word in enumerate(new_vocab)}

    # Step 2: Construct the new document-term matrix by merging columns
    n_docs = X.shape[0]
    new_X = np.zeros((n_docs, len(new_vocab_map)))

    # Populate new_X by summing columns that map to the same new word
    new_vocab = list(new_vocab_map.keys())
    for new_feature_index, (new_word, old_indices) in enumerate(new_vocab_map.items()):
        new_X[:, new_feature_index] = X[:, old_indices].sum(axis=1)

    return new_X, new_vocab

# Example usage
# X = np.load('X.npy')  # Document-term matrix
# vocab_df = pd.read_csv('vocab_map.csv', header=None, names=['word'])  # Vocabulary map
# new_X, new_vocab = transform_matrix_with_vocab_map(X, vocab_df)

# Convert new_X to a DataFrame and display the first few rows
# new_X_df = pd.DataFrame(new_X, columns=new_vocab)
# print(new_X_df.head())



folder=StratifiedKFold(n_splits=5, random_state=42, shuffle=True)




np.random.seed(0)

data_path = 'data/'

data_train=np.load(os.path.join(data_path, 'data_train.npy'), allow_pickle=True)
vocab_map=np.load(os.path.join(data_path, 'vocab_map.npy'), allow_pickle=True)
data_test=np.load(os.path.join(data_path, 'data_test.npy'), allow_pickle=True)
label_train=np.loadtxt(os.path.join(data_path, 'label_train.csv'), delimiter=',', skiprows=1).astype(int)


import pandas as pd
from supervised.automl import AutoML
import psutil
import os
jobs = psutil.cpu_count(logical=False)
from explore2 import balanced_split,stratified_split


# prep_data_train = np.hstack([prep_data_train, np.ones((prep_data_train.shape[0], 1))])
# prep_data_test = np.hstack([prep_data_test, np.ones((prep_data_test.shape[0], 1))])

# labels_train_sign=label_train[:,1]
# labels_train_sign[np.where(labels_train_sign==1)[0]]=1
# labels_train_sign[np.where(labels_train_sign==0)[0]]=-1

#trainset, testset = stratified_split(prep_data_train, labels_train_sign, 0.7)
balanced=False
if balanced:
    trainset, testset,valset = balanced_split(None,label_train[:,1],0.7)
else:
    trainset, testset = train_test_split(
    range(len(label_train[:,1])),  # pass indices instead of the data
    stratify=label_train[:,1],     # stratify by target to maintain distribution
    test_size=0.3,  # or your desired test split size
    random_state=42 # for reproducibility
    )
    testset, valset = train_test_split(
    testset,  # pass indices instead of the data
    stratify=label_train[testset,1],     # stratify by target to maintain distribution
    test_size=0.5,  # or your desired test split size
    random_state=42 # for reproducibility
    )

feature_path=False#r'v22oct\binary_counts_class.csv'
if feature_path:
        df_binary=pd.read_csv(feature_path)

        df_binary.sort_values(by=['count0@False'],ascending=[True]).to_csv('binary_counts_class_sorted.csv',index=False)

        df_sel=df_binary[df_binary['count1@True']!=0].copy()
        df_sel['ratio10False']=df_sel['count1@False']/df_sel['count0@False']
        df_sel['ratio1TF']=df_sel['count1@True']/df_sel['count1@False']
        df_sel.sort_values(by=['ratio1TF'],ascending=[False]).to_csv('binary_counts_class_sorted_sel.csv',index=False)

        Pinteresting_features=df_sel[df_sel['ratio1TF']>=1].copy()
        Pinteresting_indexes=Pinteresting_features['feature'].apply(lambda x: vocab_map.tolist().index(x)).values

        df_nsel=df_binary.copy()#[df_binary['count1@True']==0].copy()
        df_nsel['ratio10FT']=df_nsel['count1@False']/df_nsel['count1@True']
        df_nsel.sort_values(by=['ratio10FT'],ascending=[False]).to_csv('binary_counts_class_sorted_nsel.csv',index=False)
        Ninteresting_features=df_nsel[df_nsel['ratio10FT']>=100].copy()
        Ninteresting_indexes=Ninteresting_features['feature'].apply(lambda x: vocab_map.tolist().index(x)).values
        
        # drop anything that has numbers
        Pinteresting_features=Pinteresting_features[~Pinteresting_features['feature'].str.contains(r'\d')]
        Ninteresting_features=Ninteresting_features[~Ninteresting_features['feature'].str.contains(r'\d')]
        # update indexes
        Pinteresting_indexes=Pinteresting_features['feature'].apply(lambda x: vocab_map.tolist().index(x)).values
        Ninteresting_indexes=Ninteresting_features['feature'].apply(lambda x: vocab_map.tolist().index(x)).values

        interesting_features=set(Pinteresting_indexes.tolist())#+Ninteresting_indexes.tolist())
        interesting_features=list(interesting_features)
        interesting_indexes=[vocab_map.tolist().index(vocab_map[i]) for i in interesting_features]

        len(interesting_features),len(Pinteresting_features),len(Ninteresting_features)
        #save the indexes and feature list
        np.save('interesting_indexes.npy',interesting_indexes)
        np.save('interesting_features.npy',interesting_features)
        #save positive and negative interesting features separately
        np.save('Pinteresting_indexes.npy',Pinteresting_indexes)
        np.save('Ninteresting_indexes.npy',Ninteresting_indexes)
        np.save('Pinteresting_features.npy',Pinteresting_features)
        np.save('Ninteresting_features.npy',Ninteresting_features)
else:
        # use everything
        interesting_indexes=np.arange(data_train.shape[1])
vocab_interesting=vocab_map[interesting_indexes]

# Example usage
# X = np.load('X.npy')  # Document-term matrix
# vocab_df = pd.read_csv('vocab_map.csv', header=None, names=['word'])  # Vocabulary map
# new_X, new_vocab = transform_matrix_with_vocab_map(X, vocab_df)

# Convert new_X to a DataFrame and display the first few rows
# new_X_df = pd.DataFrame(new_X, columns=new_vocab)
# print(new_X_df.head())


final_data_train_orig=data_train.copy()
final_data_test_orig=data_test.copy()

nlp_preprocess=False
if nlp_preprocess:
    final_data_train,new_vocab=transform_matrix_with_vocab_map(data_train, pd.DataFrame(vocab_map, columns=['word']))
    final_data_test,_=transform_matrix_with_vocab_map(data_test, pd.DataFrame(vocab_map, columns=['word']),new_vocab=new_vocab)
else:
    final_data_train,new_vocab=data_train,vocab_map
    final_data_test,_=data_test,vocab_map
len(set(vocab_map.tolist()))==len(vocab_map.tolist())
len(set(new_vocab))==len(new_vocab)

idf=False
if idf:
    idf_train,idf_t_train=compute_idf_transform(final_data_train)
    idf_test,idf_t_test=compute_idf_transform(final_data_test)

    final_data_train=idf_train
    final_data_test=idf_test



interesting_indexes=np.arange(final_data_train.shape[1])


vocab_interesting=new_vocab
#TODO use nltk or anything language to combine words that are similar and remove stopwords, and other language preprocessing
X_train=pd.DataFrame(final_data_train[trainset,:][:,interesting_indexes], columns=vocab_interesting)
y_train=pd.DataFrame(label_train[trainset,1], columns=['target'])
X_test=pd.DataFrame(final_data_train[testset,:][:,interesting_indexes], columns=vocab_interesting)
y_test=pd.DataFrame(label_train[testset,1], columns=['target'])
X_val=pd.DataFrame(final_data_train[valset,:][:,interesting_indexes], columns=vocab_interesting)
y_val=pd.DataFrame(label_train[valset,1], columns=['target'])
X_submit=pd.DataFrame( final_data_test[:,interesting_indexes], columns=vocab_interesting)


feat_sel=False
if feat_sel:
        from featurewiz import FeatureWiz
        fwiz = FeatureWiz(feature_engg = '', nrows=None, #transform_target=True, scalers="std",
                                category_encoders="auto",verbose=2, #add_missing=False, verbose=0, imbalanced=False, 
                        #ae_options={})
        )
        fwiz.fit(X_train, y_train)
        fwiz.features  

automl=False
if automl:
    cv=list(folder.split(X_train, y_train))

    models=[#"Baseline",
    #"Linear",
    #"Decision Tree",
    #"Random Forest",
    #"Extra Trees",
    "LightGBM",
    "Xgboost",
    "CatBoost",
    "Neural Network",
    #"Nearest Neighbors",
    ]

    val={'validation_type': 'custom'}



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
            golden_features=False,#'auto',
            features_selection=False,#'auto',
            start_random_models=5,#'auto',
            hill_climbing_steps=2,#'auto',
            top_models_to_improve=3,#'auto',
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
    automl.fit(X_train, y_train,cv=cv)

    predictions = automl.predict(X_submit)

    # Use another model
    # https://github.com/mljar/mljar-supervised/issues/423
    # from supervised.model_framework import ModelFramework
    # model = ModelFramework.load("AUTOML_FOLDER_NAME_HERE", model_subpath="MODEL_FOLDER_NAME_HERE")
    # y_pred = automl._base_predict(qdqd_X_test, model=model)["prediction"].to_numpy()
    # y_pred = automl._base_predict(qdqd_X_test, model=model)["label"].to_numpy()

    path=automl._get_results_path()

gluon=True
if gluon:
    from autogluon.tabular import TabularDataset, TabularPredictor,FeatureMetadata
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator


    X_train['label']=y_train['target']
    X_test['label']=y_test['target']
    X_val['label']=y_val['target']

    dataset=TabularDataset(X_train)
    if idf:
        t='float'
    else:
        t='int'
    type_map={col: t for col in X_train.columns}
    metadata=FeatureMetadata(type_map_raw=type_map)
    generator=AutoMLPipelineFeatureGenerator(pre_enforce_types=False,enable_categorical_features = False,feature_metadata_in=metadata)
    predictor=TabularPredictor(label='label', eval_metric='f1',problem_type='binary',learner_kwargs={'positive_class':1})
    predictor.fit(train_data=dataset,presets='medium_quality',feature_generator=generator)#,feature_metadata=metadata)
    predictor.leaderboard(X_train).to_csv(os.path.join(predictor.path,'leaderboard_train.csv'))
    predictor.evaluate(X_test)
    predictor.leaderboard(X_test).to_csv(os.path.join(predictor.path,'leaderboard_test.csv'))
    predictor.evaluate(X_val)
    predictor.leaderboard(X_val).to_csv(os.path.join(predictor.path,'leaderboard_val.csv'))
    predictions = predictor.predict(X_submit).values

    # Save the predictions to a CSV file
    output = pd.DataFrame({'ID': range(len(predictions)), 'label': predictions})

    path=predictor.path
    output.to_csv(os.path.join(path,'submission.csv'), index=False)

