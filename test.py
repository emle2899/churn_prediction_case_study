import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence

def clean_data(data):
    df=data
    x=pd.to_datetime(('2014-06-1'))
    df.signup_date=pd.to_datetime(df.signup_date)
    df['Churn']=(pd.to_datetime(df['last_trip_date'])<=x).astype(int)
    df.pop('last_trip_date')
    df.pop('signup_date')
    df.avg_rating_of_driver=df.avg_rating_of_driver.fillna(2.5)
    df.dropna(inplace=True)
    df = df.replace({'phone' : {'Android':0, 'iPhone':1}})
    df.luxury_car_user=df.luxury_car_user.astype('int')
    df=pd.get_dummies(df,'city')
    df.to_csv('clean_train.csv', index=False)
    return df


if __name__=='__main__':
    df=pd.read_csv('data/churn_train.csv')
    dft=pd.read_csv('data/churn_test.csv')
    # df=pd.read_csv('clean_train.csv')
    # df.pop('signup_date')
    df=clean_data(df)
    dft=clean_data(dft)
    model=LogisticRegressionCV()
    gbc=GradientBoostingClassifier()
    y=df.pop('Churn')
    X=df
    y_test=dft.pop('Churn')
    X_test=dft
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
    # random_state=57)

    model.fit(X,y)
    probs=model.predict_proba(X_test)[:,1]
    pred=model.predict(X_test)
    acc=model.score(X_test,y_test)
    pres=precision_score(y_test, pred)
    rec=recall_score(y_test, pred)
    print(acc, pres, rec)
    gbc.fit(X,y)

    fig, axs = plot_partial_dependence(gbc, X, [0,1,2,3,4,5,6,7,8,9,10,11],
                                       feature_names=df.columns,
                                       n_jobs=-1, grid_resolution=50)
    fig.suptitle('Partial Dependence Plot for Gradient Boost Classifier')
    fig.set_figwidth(15)
    fig.set_figheight(15)
    fig.tightlayout=True
    fig.tight_layout(h_pad=10, w_pad=1)
    plt.savefig('stuff.png')
    plt.show()
    # fpr,tpr,thresholds=roc_curve(y_test,probs)
    # plt.plot(fpr, tpr)
    # plt.show()
