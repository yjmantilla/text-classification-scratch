import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def counts(x):
    labels,counts=np.unique(x,return_counts=True) # unbalanced data
    label_dict={}
    for l,c in zip(labels,counts):
        label_dict[l]=c
    print(label_dict)
    return label_dict

def f1_score(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    return 2 * tp / (2 * tp + fp + fn)

def error_counter(z, y):
  return (z * y < 0).astype(int)
  
def linearloss(z, y):
  return .5 * (y - z) ** 2

def perceptronloss(z, y):
  return np.maximum(0, -y*z)

def svmloss(z, y):
  return np.maximum(0, 1 - y*z)

def logisticloss(z, y):
  return np.log(1 + np.exp(-y*z))


class LinearModel:
    """"Abstract class for all linear models.
    """

    def __init__(self, w0, reg):
        """The linear weights and bias are in w, a matrix of size n_features x 1 (i.e. an array of size n_features).
        The regularization hyperparameter is reg.
        """
        self.w = np.array(w0, dtype=float)
        self.reg = reg

    def predict(self, X):
        """Return f(x) for a batch X.
        X is a matrix of size n_examples x n_features.
        Returns an array of size n_examples.
        """
        return np.dot(X, self.w)

    def error_rate(self, X, y,verbose=False): 
        """Return the error rate for a batch X.
        X is a matrix of size n_examples x n_features.
        y is an array of size n_examples.
        Returns a scalar.
        """

        errors=[]
        # actually, implement balanced error rate
        for class_label in np.unique(y):
            idx = np.where(y == class_label)[0]
            if verbose:
                print('Class:', class_label, 'Count:', len(idx))
            predictions = self.predict(X[idx])
            errors.append(np.nanmean(self.predict(X[idx]) * y[idx] <= 0))
            if verbose:
                print('Error rate:', errors[-1])
        return np.nanmean(errors)

    # DUMMY; will be redefined in child classes
    def loss(self, X, y):
        """Computes the mean loss for batch X.
        Takes as input a matrix X and a vector y.
        Returns a scalar.
        """
        return 0

    # DUMMY; will be redefined in child classes
    def gradient(self, X, y):
        """Computes gradient of the loss with respect to w for a batch X.
        Takes as input a matrix X and a vector y
        Returns a vector with the same shape as w.
        """
        return self.w

    def train(self, trainset,trainlabel, base_stepsize, n_steps, plot=False,verbose=False):
        """Train with full batch gradient descent with a fixed step size 
        for n_steps iterations. Return the training losses and error rates
        seen along the trajectory.
        """

        X = trainset
        y = trainlabel
        losses = []
        errors = []
        # would be better to make this stochastic with mini-batches
        batch_percent=0.1
        stepsize=base_stepsize
        for i in range(n_steps):
            # Gradient Descent
            random_idx=np.random.choice(X.shape[0],int(batch_percent*X.shape[0]),replace=False)
            X_b = X[random_idx]
            y_b = y[random_idx]
            self.w -= stepsize * self.gradient(X_b, y_b)

            # exponential decrease of step size
            if i % 1 == 0:
                stepsize = stepsize * 0.99
            # if i % (n_steps//5) == 0:
            #     stepsize = base_stepsize
            # Update losses
            losses += [self.loss(X_b, y_b)]

            # Update errors
            errors += [self.error_rate(X_b, y_b)]

            # Plot
            if i % 1 == 0:
                if verbose:
                    print(i,flush=True,end=' ')
                #print("Step", i, "Loss", losses[-1], "Error", errors[-1])
        #final error
        if verbose:
            print('final step size', stepsize)
        errors += [self.error_rate(X, y)]
        if verbose:
            print("Training completed: the train error is {:.2f}%".format(errors[-1]*100))
        return np.array(losses), np.array(errors)
    

def test_model(modelclass,trainset,trainlabel,testset,testlabel,w0=None, reg=0.1, stepsize=.2, plot=False,iters=100,verbose=False):
    """Create instance of modelclass, train it, compute test error,
    plot learning curves and decision boundary.
    """
    if w0 is None:
        #w0 = np.zeros(trainset.shape[1]) # should i subtract 1? why? 
        #w0[-1] = -1
        #np.random.seed(0)
        w0 = np.random.randn(trainset.shape[1])
    model = modelclass(w0, reg)
    # initial error
    if verbose:
        print("The initial error is {:.2f}%".format(
        model.error_rate(testset, testlabel)*100))
    training_loss, training_error = model.train(trainset,trainlabel, stepsize, iters, plot=plot,verbose=verbose)
    if verbose:
        print("The test error is {:.2f}%".format(
        model.error_rate(testset, testlabel)*100))
    #print('Initial weights: ', w0)
    #print('Final weights: ', model.w)

    # learning curves
    if plot:
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8,2))
        ax0.plot(training_loss)
        ax0.set_title('loss')
        ax1.plot(training_error)
        ax1.set_title('error rate')

        # data plot
        plt.show()
    return model,model.error_rate(testset, testlabel)*100



def compute_idf_transform(X):
    """
    Compute the IDF transform for a given bag-of-words matrix.
    
    Parameters:
    X (numpy.ndarray): A matrix of shape [num_docs, num_words] where each entry is the count of a word in a document.
    
    Returns:
    numpy.ndarray: The transformed matrix where each entry is weighted by its corresponding IDF value.
    numpy.ndarray: The array of computed IDF values.
    """
    # Step 1: Number of documents and words
    n_docs, n_words = X.shape

    # Step 2: Compute document frequency (DF)
    df = np.sum(X > 0, axis=0)  # Counts the non-zero entries for each word

    # Step 3: Compute IDF
    idf = np.log(n_docs / (1 + df))  # Adding 1 to avoid division by zero

    # Step 4: Apply IDF transform
    X_idf = X * idf

    return X_idf, idf


#stratified split based on percentage of classes

def stratified_split(data, label, percentage):
    label_dict=counts(label)
    trainset=[]
    testset=[]
    # percentage based on the minority class

    for key in label_dict:
        idx=np.where(label==key)[0]
        train_count=int(len(idx)*percentage)
        np.random.shuffle(idx)
        train_idx=idx[:train_count]
        test_idx=idx[train_count:]
        trainset+=train_idx.tolist()
        testset+=test_idx.tolist()
    print('Stratified split Train')
    counts(label[trainset])
    print('Stratified split Test')
    counts(label[testset])
    return trainset, testset

def balanced_split(data, label, percentage):
    label_dict=counts(label)
    trainset=[]
    testset=[]
    valset=[]
    # percentage based on the minority class
    min_class=np.min(list(label_dict.values()))
    train_count=int(min_class*percentage)
    other=(1-percentage)/2
    for key in label_dict:
        idx=np.where(label==key)[0]
        np.random.shuffle(idx)
        train_idx=idx[:train_count]
        left=len(idx)-train_count
        test_idx=idx[train_count:train_count+int(min_class*other)]
        val_idx=idx[train_count+int(min_class*other):train_count+2*int(min_class*other)]
        trainset+=train_idx.tolist()
        testset+=test_idx.tolist()
        valset+=val_idx.tolist()
    print('Balanced split Train')
    counts(label[trainset])
    print('Balanced split Test')
    counts(label[testset])
    print('Balanced split Val')
    counts(label[valset])

    return trainset, testset,valset

class LogisticRegression(LinearModel):

    def __init__(self, w0, reg):
        super().__init__(w0, reg)

    def loss(self, X, y):
        """Computes the mean loss for batch X.
        Takes as input a matrix X and a vector y.
        Returns a scalar.
        """
        return np.nanmean(np.log(1 + np.exp(-y * self.predict(X)))) + .5 * self.reg * np.sum(self.w ** 2)

    def gradient(self, X, y):
        """Computes gradient of the loss with respect to w for a batch X.
        Takes as input a matrix X and a vector y
        Returns a vector with the same shape as w.
        """
        probas = 1 / (1 + np.exp(y * self.predict(X)))
        return ((probas * -y)[:, np.newaxis] * X).mean(axis=0) + self.reg * self.w

class Perceptron(LinearModel):

    def __init__(self, w0, reg):
        super().__init__(w0, reg)

    def loss(self, X, y):
        """Computes the mean loss for batch X.
        Takes as input a matrix X and a vector y.
        Returns a scalar.
        """
        return np.mean(np.maximum(0, -y * self.predict(X))) + 0.5 * self.reg * np.sum(self.w ** 2)

    def gradient(self, X, y):
        """Computes gradient of the loss with respect to w for a batch X.
        Takes as input a matrix X and a vector y
        Returns a vector with the same shape as w.
        """
        active = (-y * self.predict(X) > 0).astype(float)
        return - ((y * active)[:, np.newaxis] * X).mean(axis=0) + self.reg * self.w

class LinearRegression(LinearModel):

    def __init__(self, w0, reg):
        super().__init__(w0, reg)

    def loss(self, X, y):
        """Computes the mean loss for batch X.
        Takes as input a matrix X and a vector y.
        Returns a scalar.
        """
        return 0.5 * np.mean((self.predict(X) - y) ** 2) + self.reg * 0.5 * np.sum(self.w ** 2)

    def gradient(self, X, y):
        """Computes loss gradient with respect to w for a batch X.
        Takes as input a matrix X and a vector y.
        Returns a vector with the same shape as w.
        """
        return ((self.predict(X) - y)[:, np.newaxis] * X).mean(axis=0) + self.reg * self.w

class SVM(LinearModel):

    def __init__(self, w0, reg):
        super().__init__(w0, reg)

    def loss(self, X, y):
        """Computes the mean loss for batch X.
        Takes as input a matrix X and a vector y.
        Returns a scalar.
        """
        return np.mean(np.maximum(0, 1 - y * self.predict(X))) + 0.5 * self.reg * np.sum(self.w ** 2)

    def gradient(self, X, y):
        """Computes gradient of the loss with respect to w for a batch X.
        Takes as input a matrix X and a vector y
        Returns a vector with the same shape as w.
        """
        active = (1 - y * self.predict(X) > 0).astype(float)
        return - ((y * active)[:, np.newaxis] * X).mean(axis=0) + self.reg * self.w


# hyperparameters optimization through grid search

import itertools as it

# hyperparameters

import joblib
# grid search

def opt(class_,regs,stepsizes,iters,X_train,y_train,X_val,y_val):
    import itertools as it
    best_error = 1
    best_args = None
    combs=list(it.product(regs, stepsizes, iters))
    print('Number of combinations:', len(combs))
    def train_model(i,args):
        import numpy as np

        model = class_(np.random.randn(X_train.shape[1]), args[0])
        model.train(X_train, y_train, args[1], args[2])
        error = model.error_rate(X_val, y_val) #not touching the test set
        print(i)
        return model, error
    
    results = joblib.Parallel(n_jobs=-1)(joblib.delayed(train_model)(i,args) for i,args in enumerate(combs))
    for args, error in zip(combs, results):
        if error[1] < best_error:
            best_args = args
            best_error = error[1]
    return best_args, best_error,results,combs

if __name__ == '__main__':
    np.random.seed(0)

    data_path = 'data/'

    data_train=np.load(os.path.join(data_path, 'data_train.npy'), allow_pickle=True)
    vocab_map=np.load(os.path.join(data_path, 'vocab_map.npy'), allow_pickle=True)
    data_test=np.load(os.path.join(data_path, 'data_test.npy'), allow_pickle=True)
    label_train=np.loadtxt(os.path.join(data_path, 'label_train.csv'), delimiter=',', skiprows=1).astype(int)


    label_train[:,1]

    label_dict=counts(label_train[:,1])


    data_train.shape, vocab_map.shape, data_test.shape, label_train.shape

    data_dict=counts(data_train)

    # to get an idea of what words are in the documents, create a list of unique words per document based on the counts

    unique_words_per_doc=[]

    for i in range(data_train.shape[0]):
        unique_words= np.where(data_train[i,:]>0)[0]
        # map to vocab
        unique_words=[vocab_map[j] for j in unique_words]
        unique_words_per_doc.append(unique_words)

    # print the first 10 documents with their labels

    for i in range(10):
        print(label_train[i,1], unique_words_per_doc[i])
        print()

    # find the most common words per class
    # maybe by doing correlation with the labels of each feature
    data_train_binary=(data_train>0).astype(int)
    word_power={}
    index_power=np.zeros(data_train_binary.shape[1])

    #make dataframe
    if not os.path.exists('binary_counts.csv'):
        co={'feature':[],'count0':[],'count1':[]}
        for feat in range(data_train_binary.shape[1]):
            cc=counts(data_train_binary[:,feat])
            co['feature']+=[vocab_map[feat]]
            co['count0']+=[cc[0]]
            co['count1']+=[cc[1]]
            print('Feature:', vocab_map[feat],'Count 0:', cc[0], 'Count 1:', cc[1])

        df_binary=pd.DataFrame(co)
        df_binary.sort_values(by='count1',ascending=False)
        df_binary.sort_values(by='count1',ascending=False).to_csv('binary_counts.csv',index=False)

    if not os.path.exists('binary_counts_class.csv'):
        index_true=np.where(label_train[:,1]==1)[0]
        index_false=np.where(label_train[:,1]==0)[0]
        data_train_binary_true=data_train_binary[index_true,:]
        data_train_binary_false=data_train_binary[index_false,:]

        co={'feature':[],'count0@False':[],'count1@False':[],'count0@True':[],'count1@True':[]}
        for feat in range(data_train_binary_true.shape[1]):
            ccTrue=counts(data_train_binary_true[:,feat])
            ccFalse=counts(data_train_binary_false[:,feat])
            co['feature']+=[vocab_map[feat]]
            co['count0@False']+=[ccFalse.get(0,0)]
            co['count1@False']+=[ccFalse.get(1,0)]
            co['count0@True']+=[ccTrue.get(0,0)]
            co['count1@True']+=[ccTrue.get(1,0)]
            print('Feature:', vocab_map[feat],'Count 0@False:', ccFalse.get(0,0), 'Count 1@False:', ccFalse.get(1,0), 'Count 0@True:', ccTrue.get(0,0), 'Count 1@True:', ccTrue.get(1,0))

        df_binary=pd.DataFrame(co)
        df_binary.sort_values(by=['count0@False'],ascending=[False]).to_csv('binary_counts_class.csv',index=False)

    df_binary=pd.read_csv('binary_counts_class.csv')

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

    interesting_features=set(Pinteresting_indexes.tolist()+Ninteresting_indexes.tolist())
    interesting_features=list(interesting_features)
    interesting_indexes=[vocab_map.tolist().index(vocab_map[i]) for i in interesting_features]

    len(interesting_features),len(Pinteresting_features),len(Ninteresting_features)
    #check linear independence of features
    #np.linalg.matrix_rank(data_train_binary.T)


    if not os.path.exists('word_power.npy'):
        for feature in range(data_train_binary.shape[1]):
            #feature=vocab_map.tolist().index('convex')
            print('Feature:', vocab_map[feature])
            idx=len(np.where(data_train_binary[:,feature]==label_train[:,1])[0])/data_train_binary.shape[0]
            idx2=len(np.where(data_train_binary[:,feature]==np.bitwise_not(label_train[:,1].astype(bool)).astype(int))[0])/data_train_binary.shape[0]
            power=np.max([idx,idx2])
            word_power[vocab_map[feature]]=power
            index_power[feature]=power
        np.save('word_power.npy', word_power)
        np.save('index_power.npy', index_power)
    else:
        word_power=np.load('word_power.npy', allow_pickle=True).item()
        index_power=np.load('index_power.npy', allow_pickle=True)
    # sort the words by power
    sorted_word_power = sorted(word_power.items(), key=lambda x: x[1], reverse=True)
    less_powerful_words = sorted(word_power.items(), key=lambda x: x[1], reverse=False)

    # convex=vocab_map.tolist().index('convex')

    # data_train_convex=data_train_binary[:,convex].astype(int)
    # pred_convex=data_train_convex.copy()
    # # check if the convex feature is a good predictor of the label
    # accuracy_rate=np.mean(pred_convex==label_train[:,1])
    # df = pd.DataFrame({'ID': np.arange(len(pred_convex)), 'label': pred_convex})
    # df.to_csv('convex.csv', index=False)

    # label_train[:,1]
    # data_test[:,convex]



    N=100
    for i in range(N):
        print(sorted_word_power[i])

    for i in range(N):
        print(less_powerful_words[i])

    # see power distribution

    index_mean,index_std=np.mean(index_power), np.std(index_power)

    if False:
        plt.hist(index_power, bins=1000)
        # cut off from 7000
        plt.xlim(index_mean-index_std,index_mean+index_std)
        plt.show()

    # see how many words are in the top 10% of powerful words

    top_words_positive=[word for word in word_power if word_power[word]>0.7]
    top_words_negative=[word for word in word_power if 1-word_power[word]>0.7]
    top_words=top_words_positive+top_words_negative
    print(len(top_words),len(top_words_positive),len(top_words_negative))

    # get indices of top words

    top_word_indices=[i for i in range(data_train_binary.shape[1]) if vocab_map[i] in top_words]



    # save vocab_map to a csv file

    if not os.path.exists('vocab_map.csv'):
        df = pd.DataFrame(vocab_map)
        df.to_csv('vocab_map.csv', index=False)

    print(data_train)

    # handmade

    data_train_binary=(data_train>0).astype(int)

    P=data_train_binary[:,Pinteresting_indexes].sum(axis=1)
    N=data_train_binary[:,Ninteresting_indexes].sum(axis=1)



    preds=(P>1).astype(int)
    np.unique(preds,return_counts=True)
    err=[]
    f1s=[]
    for cl in np.unique(label_train[:,1]):
        idx=np.where(label_train[:,1]==cl)[0]
        print('Class:', cl, 'Count:', len(idx))
        print('Error rate:', np.mean(preds[idx]!=cl))
        print('F1 score:', f1_score(label_train[idx,1], preds[idx]))
        f1s.append(f1_score(label_train[idx,1], preds[idx]))
        err.append(np.mean(preds[idx]!=cl))

    print('Macro F1 score:', np.mean(f1s))
    print('Macro Error rate:', np.mean(err))
    print('Full error rate:', np.mean(preds!=label_train[:,1]))
    print('F1 score:', f1_score(label_train[:,1], preds))

    preds=(N<1).astype(int)
    err=[]
    f1s=[]

    np.unique(preds,return_counts=True)
    for cl in np.unique(label_train[:,1]):
        idx=np.where(label_train[:,1]==cl)[0]
        print('Class:', cl, 'Count:', len(idx))
        print('Error rate:', np.mean(preds[idx]!=cl))
        print('F1 score:', f1_score(label_train[idx,1], preds[idx]))
        f1s.append(f1_score(label_train[idx,1], preds[idx]))
        err.append(np.mean(preds[idx]!=cl))

    print('Macro F1 score:', np.mean(f1s))
    print('Macro Error rate:', np.mean(err))

    print('Full error rate:', np.mean(preds!=label_train[:,1]))
    print('F1 score:', f1_score(label_train[:,1], preds))

    # make predictions on the test set

    P_test=data_test[:,Pinteresting_indexes].sum(axis=1)
    N_test=data_test[:,Ninteresting_indexes].sum(axis=1)

    preds_test=(P_test>1).astype(int)
    df = pd.DataFrame({'ID': np.arange(len(preds_test)), 'label': preds_test})
    df.to_csv('submission_P.csv', index=False)
    preds_test=(N_test<1).astype(int)
    df = pd.DataFrame({'ID': np.arange(len(preds_test)), 'label': preds_test})
    df.to_csv('submission_N.csv', index=False)
    # make a logistic regression from scratch


    prep_data_train = data_train[:,interesting_indexes]#[:,top_word_indices]
    prep_data_test = data_test[:,interesting_indexes]#[:,top_word_indices]

    idf_train,idf_t_train=compute_idf_transform(data_train)
    idf_test,idf_t_test=compute_idf_transform(data_test)

    # normalize data by row sum
    # prep_data_train = prep_data_train#/ np.sum(prep_data_train, axis=1)[:, np.newaxis]
    # prep_data_test = prep_data_test #/ np.sum(prep_data_test, axis=1)[:, np.newaxis]

    prep_data_train=idf_train[:,interesting_indexes]
    prep_data_test=idf_test[:,interesting_indexes]

    #normalize data by z-score by row
    # prep_data_train = (prep_data_train - np.mean(prep_data_train, axis=1)[:, np.newaxis]) / np.std(prep_data_train, axis=1)[:, np.newaxis]
    # prep_data_test = (prep_data_test - np.mean(prep_data_test, axis=1)[:, np.newaxis]) / np.std(prep_data_test, axis=1)[:, np.newaxis]

    assert np.isnan(prep_data_train).sum()==0
    assert np.isnan(prep_data_test).sum()==0
    #add bias term

    # def random_projections(X, A):
    #     return X @ A /np.sqrt(2)

    # AP = np.random.randn(Pinteresting_indexes.shape[0],10)
    # AN = np.random.randn(Ninteresting_indexes.shape[0],10)
    # prep_data_train=np.hstack([random_projections(data_train[:,Pinteresting_indexes], AP),random_projections(data_train[:,Ninteresting_indexes], AN)])
    # prep_data_test=np.hstack([random_projections(data_test[:,Pinteresting_indexes], AP),random_projections(data_test[:,Ninteresting_indexes], AN)])
    #prep_data_test=random_projections(prep_data_test, A)

    prep_data_train = np.hstack([prep_data_train, np.ones((prep_data_train.shape[0], 1))])
    prep_data_test = np.hstack([prep_data_test, np.ones((prep_data_test.shape[0], 1))])

    labels_train_sign=label_train[:,1]
    labels_train_sign[np.where(labels_train_sign==1)[0]]=1
    labels_train_sign[np.where(labels_train_sign==0)[0]]=-1

    #trainset, testset = stratified_split(prep_data_train, labels_train_sign, 0.7)
    trainset, testset,valset = balanced_split(prep_data_train, labels_train_sign,0.7)


    ensemble=[]
    instances_per_model=5
    test_errors_per_instance=[]
    for model_,mod in zip([LogisticRegression,LinearRegression,Perceptron,SVM],['LogR','LR','Per','SVM']):
        regs = [0.001,0.01, 0.1, 1, 10]
        stepsizes = [0.01, 0.1, 1, 10,100]
        iters = [30,50, 100,500]

        if not os.path.exists(f'best_args_{mod}.npy'):
            best_args, best_error,results,combs = opt(model_, regs, stepsizes, iters,prep_data_train[valset], labels_train_sign[valset],prep_data_train[trainset], labels_train_sign[trainset])
            reg=best_args[0]
            stepsize=best_args[1]
            iters=best_args[2]
            np.save(f'best_args_{mod}.npy', best_args)
            np.save(f'best_error_{mod}.npy', best_error)
            np.save(f'results_{mod}.npy', [r for x,r in results])
            np.save(f'combs_{mod}.npy', combs)

        best_error=1#np.load(f'best_error_{mod}.npy')
        #best_args=np.load(f'best_args_{mod}.npy')
        results=np.load(f'results_{mod}.npy',allow_pickle=True)
        combs=np.load(f'combs_{mod}.npy',allow_pickle=True)

        for args, error in zip(combs, results):
            if error < best_error and error!=0:
                best_args = args
                best_error = error

        reg=best_args[0]
        stepsize=best_args[1]
        iters=best_args[2].astype(int)
        print('Best error:', best_error, 'Best args:', best_args)
        args=dict(verbose=True,reg=reg,stepsize=stepsize,iters=iters,trainset=prep_data_train[trainset],trainlabel=labels_train_sign[trainset],testset=prep_data_train[testset+valset],testlabel=labels_train_sign[testset+valset])
        prep_data_train[trainset].shape
        prep_data_train[testset].shape


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
            preds=model_2.predict(prep_data_test)
            preds[preds>=0]=1
            preds[preds<0]=0

            # Question 2

            df = pd.DataFrame({'ID': np.arange(len(preds)), 'label': preds.astype(int)})
            df.to_csv(f'submission_{mod}_{i}.csv', index=False)

            preds=model_2.predict(prep_data_train[testset+valset])
            preds[preds>=0]=1
            preds[preds<0]=0
            error_rate=f'{np.mean(preds!=(labels_train_sign[testset+valset]>0).astype(int)).round(2)}'
            print('Error rate:', error_rate)
            df = pd.DataFrame({'ID': np.arange(len(preds)), 'label': preds.astype(int)})
            df.to_csv(f'known_{mod}_{i}_{error_rate}.csv', index=False)
            df = pd.DataFrame({'ID': testset+valset, 'label': (labels_train_sign[testset+valset]>0).astype(int)})
            df.to_csv(f'known_{mod}_{i}_true.csv', index=False)

    # load predictions from the models
    import numpy as np
    import pandas as pd
    import glob
    results=glob.glob('submission_*_*.csv')
    answers=glob.glob('known*true.csv')
    checks=glob.glob('known*')
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
    df.to_csv(f'submission_ensemble.csv', index=False)



    #print('Linear Model')
    #test_model(LinearModel, **args)

    # from sklearn.linear_model import LogisticRegression as LR

    # logreg = LR(max_iter=1000)
    # logreg.fit(prep_data_train[trainset], labels_train_sign[trainset])
    # print('Logistic Regression')
    # print('Train error:', np.mean(logreg.predict(prep_data_train[trainset]) * labels_train_sign[trainset] <= 0))
    # print('Test error:', np.mean(logreg.predict(prep_data_train[testset]) * labels_train_sign[testset] <= 0))
    # from sklearn.metrics import f1_score
    # f1_score(labels_train_sign[testset], logreg.predict(prep_data_train[testset]))

    # from supervised.automl import AutoML
    # automl = AutoML(eval_metric='accuracy')
    # automl.fit(prep_data_train[trainset], labels_train_sign[trainset])

    # predictions = automl.predict(X_test)
