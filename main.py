import numpy as np
import pandas as pd
import random



global sample_share, random_seed, threshold, bias

random_seed, threshold = None,0.5


binary_class = {}


def binarize_X(X:pd.DataFrame,partitions = 5):
    """Scale values from X into pandas.DataFrame of binary values
    partitions: Each feature in proportionally divided into 5 partitions and added as new feature.
    Try changing the value of partitions and compare the results.
    """
    for column in X.columns:
        col_values = list(X[column].unique())

        if len(col_values) == 1 and (1 in col_values or 0 in col_values):
            continue
        elif len(col_values) == 2 and 0 in col_values and 1 in col_values:
            continue
        elif len(col_values)<=2 or X[column].dtypes==np.dtype('O'): #checking for categorical features
            col_values = sorted(list(X[column].unique()))
            for i in col_values:
                X['{}_{}'.format(column, i)]= (X[column] == i).astype(int)
        elif X[column].dtype== np.dtype('int64'): #checking for numerical features
            min = X[column].min()
            maxx = X[column].max()
            interval = maxx - min
            start = min+interval/partitions
            end = maxx - interval/partitions
            flag = 0
            for i in np.linspace(start,end,partitions):
                X['{}_{}'.format(column, flag)]=(X[column]>=i).astype(int)
                flag+=1
        X.drop([column],axis=1,inplace=True) #dropping the initial feature after scaling
    return X

def binarize_y(y):
    target = sorted(y.unique())
    if len(target)!=2:
        raise Exception('Missing binary target feature')
    binary_class[False] = target[0]
    binary_class[True] = target[1]
    return (y==target[1]).astype(int)

def make_extent(example):
    """create a list of extent for the given example."""
    return example[example==1].index.tolist()

def intersection(example,context):
    """Calculate the intersection between example and the context"""
    return [result for result in example if result in context]


def binarize_X1(X: pd.DataFrame) -> 'pd.DataFrame[bool]':
    """Scale values from X into pandas.DataFrame of binary values"""
    dummies = [pd.get_dummies(X[f], prefix=f, prefix_sep=': ') for f in X.columns]
    X_bin = pd.concat(dummies, axis=1).astype(bool)
    return X_bin

def fitClassifier(X,y,sample_share=0.5,random=True,threshold=0.000000000001):
    
    X = binarize_X1(X)
    y = binarize_y(y)
    global plus,minus,plus_obj,minus_obj

    plus = X[y==1]
    minus = X[y==0]
    if random:
        sample_size = int(sample_share * plus.shape[0])
        plus = plus.sample(n=sample_size, random_state=random_seed)
        minus = minus.sample(n=sample_size, random_state=random_seed)
    plus_obj = {}
    minus_obj = {}
    pos = plus
    neg = minus
    for col in X.columns:
        plus_obj[col] = pos[col][pos[col] == 1].index
        minus_obj[col] = neg[col][neg[col] == 1].index

def predict(X,bias='random'):
    random.seed(random_seed)
    X = binarize_X1(X)
    predictions = []
    for i in range(X.shape[0]):
        extent = make_extent(X.iloc[i])
        pos = check_hypothesis(extent,'plus')
        neg = check_hypothesis(extent,'minus')

        if pos == neg:
            if bias == 'random':
                prediction = random.choice([True, False])
            elif bias == 'plus':
                prediction = True
                
            else:
                prediction = False
        else:
            prediction = pos>=neg
        predictions.append(binary_class[prediction])
    return predictions




def check_hypothesis(extent,base_context):
    sample = (plus if base_context =='plus' else minus)
    review_sample = (minus if base_context =='plus' else plus)
    object = (minus_obj if base_context == 'plus' else plus_obj)

    hyp_score = 0
    for _, i in sample.iterrows():
        intersect = intersection(extent,make_extent(i))
        k = 0
        if intersect:
            hypothesis = object[intersect[0]]
            for column in intersect:
                hypothesis = intersection(hypothesis,object[column])
                if not hypothesis:
                    break
                k = len(hypothesis)/review_sample.shape[0]
                if k < threshold:
                    hyp_score+=len(intersect)/len(extent)
    hyp_score = hyp_score/sample.shape[0]
    return hyp_score


