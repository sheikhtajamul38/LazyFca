import numpy as np
import pandas as pd

def scaling_X(X:pd.DataFrame,partitions=5):
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


def create_intent(example):
    attrib = []
    for j in range(len(example)):
        attrib.append(j)  
    return set([str(i)+':'+str(k) for i,k in zip(attrib,example)])


def intersection(example, context):
    intent = create_intent(example)
    intersection_res = []
    count = 0
    for i in context:
        intersection_x = intent&i
        if intersection_x:
            count = count + 1
            intersection_res.append(intersection_x)
    return [intersection_res, count]


def aggregate1(plus_massiv,  plus_count, minus_massiv, minus_count):
    psum = sum([len(p) for p in plus_massiv])
    size_plus = len(plus_massiv)
    msum = sum([len(p) for p in minus_massiv])
    size_minus = len(minus_massiv)
    k = plus_count/psum
    z = minus_count/msum
    return k<=z

def classifier2(X_train, X_test, y_train, y_test):
    """Converting training and testing data to lists
    Creating positive and negative intents for the loaded data
    """
    X_tr = X_train.values.tolist()
    X_ts = X_test.values.tolist()
    y_tr = y_train.values.tolist()
    y_ts = y_test.values.tolist()

    plus_intent = [create_intent(O) for i, O in enumerate(X_tr) if y_tr[i]] 
    
    minus_intent = [create_intent(O) for i, O in enumerate(X_tr) if not y_tr[i]]
    
    results = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'contr': 0} 
    
    for j, i in enumerate(X_ts):
        label = {'plus': False, 'minus': False}
        
        plus = intersection(i, plus_intent)
        minus = intersection(i, minus_intent)
        plus_massiv = plus[0]
        plus_count = plus[1]
        minus_massiv = minus[0]
        minus_count = minus[1]
        if aggregate1(plus_massiv,  plus_count, minus_massiv, minus_count):
            label['plus'] = True
        else:
            label['minus'] = True
        if label['plus'] and not label['minus']:
            if y_ts[j]:
                results['tp']+=1
            else: 
                results['fp']+=1
        elif label['minus'] and not label['plus']:
            if y_ts[j]:
                results['fn']+=1
            else: 
                results['tn']+=1
        else:
            results['contr']+=1 #updating contradictory classes
    return results