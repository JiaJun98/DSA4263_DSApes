from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, mean_squared_error, auc, roc_curve
import Preprocessing
import pickle


df = pd.read_csv('reviews.csv')
df = Dataset(df)
df.create_tfidf(root_words = "lemmatize", stop_words = True)
tfidf = pd.DataFrame(df.tfidf[1].toarray())
tfidf['Time'] = df.date
tfidf['Sentiment'] = df.sentiments
tfidf = tfidf.replace({'positive': 1, 'negative':0})


def split(df):
    train, test = train_test_split(df, test_size = 0.2, random_state = 4263, stratify = df['Sentiment'])
    return [train, test]


train, test = split(tfidf)
X_train = train.drop(['Time', 'Sentiment'], axis = 1)
y_train = train['Sentiment'].to_numpy()
X_test = test.drop(['Time', 'Sentiment'], axis = 1)
y_test = test['Sentiment'].to_numpy()


# ## **Logistic Regression Model**

#Logistic Regression Model
logreg = LogisticRegression(random_state = 4263, multi_class = 'multinomial', solver = 'saga', max_iter = 2000)
logreg.fit(X_train, y_train)
lr_pred = logreg.predict(X_test)
lr_pred_prob = logreg.predict_proba(X_test)

#Evaluation Metrics
metrics = ['ROC', 'F1', 'Accuracy', 'Precision', 'Recall']
roc = roc_auc_score(y_test, lr_pred_prob[:,1], average = 'weighted')
f1 = f1_score(y_test, lr_pred, average = 'weighted')
acc = accuracy_score(y_test, lr_pred)
prec = precision_score(y_test, lr_pred, average = 'weighted')
recall = recall_score(y_test, lr_pred, average = 'weighted')
lr_scores = [roc, f1, acc, prec, recall]
for i in range(0, 5):
    print(metrics[i] + ': ' + str(round(lr_scores[i], 5)))


# ## **Random Forest Classifier Model**

#Random Forest Classifier Model
rf_filename = 'rf_model.sav'
rf = pickle.load(open(rf_filename, 'rb'))

#fit trained params
rf_pred = rf.predict(X_test)
rf_pred_prob = rf.predict_proba(X_test)

roc = roc_auc_score(y_test, rf_pred_prob[:,1], average = 'weighted')
f1 = f1_score(y_test, rf_pred, average = 'weighted')
acc = accuracy_score(y_test, rf_pred)
prec = precision_score(y_test, rf_pred, average = 'weighted')
recall = recall_score(y_test, rf_pred, average = 'weighted')
rf_scores = [roc, f1, acc, prec, recall]
for i in range(0, 5):
    print(metrics[i] + ':' + str(round(rf_scores[i], 5)))


# ## **XGBoost Classifier Model**

#XGBoost Classifier Model
xgb_filename = 'xgb_model.sav'
xgb = pickle.load(open(xgb_filename, 'rb'))

#fit trained params
xgb_pred_prob = xgb.predict_proba(X_test)
xgb_pred = xgb.predict(X_test)

#Evaluation Metrics
metrics = ['roc', 'f1', 'acc', 'prec', 'recall']
roc = roc_auc_score(y_test, xgb_pred_prob[:,1], average = 'weighted')
f1 = f1_score(y_test, xgb_pred, average = 'weighted')
acc = accuracy_score(y_test, xgb_pred)
prec = precision_score(y_test, xgb_pred, average = 'weighted')
recall = recall_score(y_test, xgb_pred, average = 'weighted')
xgb_scores = [roc, f1, acc, prec, recall]
for i in range(0, 5):
    print(metrics[i] + ':' + str(round(xgb_scores[i], 5)))


# ## **Overall Evaluation Metrics**
data = {'Logistic Regression': lr_scores, 
        'Random Forest': rf_scores, 
        'XGBoost': xgb_scores}
pd.DataFrame(data, index = ['roc', 'f1', 'acc', 'prec', 'recall'])