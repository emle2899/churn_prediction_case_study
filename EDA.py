import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
import pydot

df = pd.read_csv('./data/churn_train.csv')
df2 = pd.read_csv('./data/churn_test.csv')

def clean_data(df):
    x=pd.to_datetime(('2014-06-1'))
    df.signup_date=pd.to_datetime(df.signup_date)
    df['Churn']=(pd.to_datetime(df['last_trip_date'])<=x).astype(int)
    df.pop('last_trip_date')
    df.pop('signup_date')
    df.avg_rating_of_driver=df.avg_rating_of_driver.fillna(2.5)
    df.dropna(inplace=True)
    df = df.replace({'phone' : {'Android':0, 'iPhone':1}})
    df.luxury_car_user=df.luxury_car_user.astype('int')
    df = df.replace({'city' : {"Astapor":0, "King's Landing":1, "Winterfell":2}})
    return df

def plot_roc(X_test,y_test,model,title, **kwargs):
    '''
    Plot ROC curve
    INPUT:
    fpr, tpr = array
    title = string
    OUTPUT:
    plt.plot
    '''
    clf = model(**kwargs)
    clf.fit(X_train,y_train)
    probabilities = clf.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probabilities)

    plt.plot(fpr, tpr,label='auc = {0:.2f}'.format(metrics.auc(fpr,tpr)))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return model.score(X_test, y_test), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict)

# start analysis
df = clean_data(df)
df2 = clean_data(df2)
col1 = df.columns.tolist()

X_train = df.iloc[0:1000]
y_train = X_train.pop('Churn')
X_test = df2.iloc[0:1000]
y_test = X_test.pop('Churn')

rf = RandomForestClassifier(n_estimators=100,
                            oob_score=True)
rf.fit(X_train, y_train)


# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
score = rf.score(X_test,y_test)
errors = abs(predictions - y_test)

#importances = forest_fit.feature_importances_[:n]
n = 10
importances = rf.feature_importances_[:n]
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
features = list(df.columns[indices])


# Pull out one tree from the forest
tree = rf.estimators_[3]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree2.dot', feature_names = col1[:-1], rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree2.dot')
# Write graph to a png file
graph.write_png('tree2.png')


# print all the info
plt.figure()
plt.subplot(2, 2, 1)
plot_roc(X_test,y_test,model=RandomForestClassifier,title='RandomForest ROC')
plt.subplot(2, 2, 2)
plot_roc(X_test,y_test,model=LogisticRegression,title='Logistic Regression ROC')
plt.show()

print("\n13. Feature ranking:")
for f in range(n):
    print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))
print("\n9. confusion matrix:")
print(confusion_matrix(y_test, y_train))
print("\n16. Accuracy, Precision, Recall")
print(get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=25, max_features=5))
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print('Score:', score)
