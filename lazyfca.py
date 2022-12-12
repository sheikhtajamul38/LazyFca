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


def calculate(plus_con,  plus_count, minus_con, minus_count):
    k = plus_count/sum([len(p) for p in plus_con])
    z = minus_count/sum([len(p) for p in minus_con])
    return k<=z


def predict(X_train, X_test, y_train, y_test):
    """Converting training and testing data to lists
    Creating positive and negative intents for the loaded data
    """

    positive = [create_intent(O) for i, O in enumerate(X_train.values.tolist()) if y_train.values.tolist()[i]] 
    
    negative = [create_intent(O) for i, O in enumerate(X_train.values.tolist()) if not y_train.values.tolist()[i]]
    
    results = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'CTR': 0} 
    
    for j, i in enumerate(X_test.values.tolist()):
        label = {'plus': False, 'minus': False}
        
        plus = intersection(i, positive)
        minus = intersection(i, negative)

        #pos = plus[1]/sum([len(p) for p in plus[0]])
        #neg = minus[1]/sum([len(m) for m in minus[0]])

        if calculate(plus[0],  plus[1], minus[0], minus[1]):
            label['plus'] = True

        else:
            label['minus'] = True

        if label['plus'] and not label['minus']:
            if y_test.values.tolist()[j]:
                results['TP']+=1
            else: 
                results['FP']+=1

        elif label['minus'] and not label['plus']:
            if y_test.values.tolist()[j]:
                results['FN']+=1
            else: 
                results['TN']+=1
        else:
            results['CTR']+=1 #updating contradictory classes
    return results


def classificationReport(results):
    acc = (results['TP'] + results['TN']) / (results['TP'] + results['TN'] + results['FP'] + results['FN'])
    precision = results['TP'] / (results['TP'] + results['FP'])

    recall = results['TP'] / (results['TP'] + results['FN'])
    data = [{"Accuracy":acc*100,"Precision":precision*100,'Recall':recall*100}]
    #print("Classification Report")
    df = pd.DataFrame(data)
    return df
    