#!/usr/bin/env python
# coding: utf-8

# ## **Logistic Regression Model**

# In[ ]:
<<<<<<< Updated upstream


#Logistic Regression Model
logreg = LogisticRegression(random_state = 562, multi_class = 'multinomial', solver = 'saga', max_iter = 2000)
=======
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, mean_squared_error, auc, roc_curve

#Logistic Regression Model
logreg = LogisticRegression(random_state = 4263, multi_class = 'multinomial', solver = 'saga', max_iter = 2000)
>>>>>>> Stashed changes
logreg.fit(X_train, y_train)
lr_pred = logreg.predict(X_test)
lr_pred_prob = logreg.predict_proba(X_test)

#Evaluation Metrics
roc = roc_auc_score(y_test, lr_pred_prob, multi_class = 'ovo', average = 'weighted')
f1 = f1_score(y_test, lr_pred, average = 'weighted')
acc = accuracy_score(y_test, lr_pred)
prec = precision_score(y_test, lr_pred, average = 'weighted')
recall = recall_score(y_test, lr_pred, average = 'weighted')
lr_scores = [roc, f1, acc, prec, recall]
for i in range(0, 5):
    print(metrics[i] + ':' + str(round(lr_scores[i], 5)))


# ## **Random Forest Classifier Model**

# In[ ]:


#Parameter Tuning
n_est = np.arange(100,251,50)
max_d = np.arange(3,6,1)
rf_grid = {'max_depth':max_d, 'n_estimators':n_est}


# In[ ]:


#GridSearchCV
rf = RandomForestClassifier(random_state = 4263, criterion = 'entropy')
rf_gscv = GridSearchCV(rf, rf_grid)
rf_gscv.fit(X_train, y_train)
print(rf_gscv.best_params_)
rf_para = rf_gscv.best_params_


# In[ ]:


#Random Forest Classifier Model
rf = RandomForestClassifier(n_estimators = rf_para.get('n_estimators'), max_depth = rf_para.get('max_depth'), criterion = 'entropy', random_state = 4263)

#fit trained params
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_pred_prob = rf.predict_proba(X_test)

roc = roc_auc_score(y_test, rf_pred_prob, multi_class = 'ovo', average = 'weighted')
f1 = f1_score(y_test, rf_pred, average = 'weighted')
acc = accuracy_score(y_test, rf_pred)
prec = precision_score(y_test, rf_pred, average = 'weighted')
recall = recall_score(y_test, rf_pred, average = 'weighted')
rf_scores = [roc, f1, acc, prec, recall]
for i in range(0, 5):
    print(metrics[i] + ':' + str(round(rf_scores[i], 5)))


# ## **XGBoost Classifier Model**

# In[ ]:


#Parameter Tuning
eta = np.arange(0.1,0.11,0.02)
max_d = np.arange(3,6,1)
min_weight = np.arange(10,21,2)
sample = np.arange(0.74,0.95,0.04)
n_est = np.arange(100,251,50)
xgb_grid = {'eta':eta, 'max_depth':max_d, 'min_child_weight':min_weight, 'colsample_bytree':sample, 'n_estimators':n_est}


# In[ ]:


#GridSearchCV
xgb = XGBClassifier(random_state = 4263, eval_metric = roc_auc_score)
xgb_gscv = GridSearchCV(xgb, xgb_grid)
xgb_gscv.fit(X_train, y_train)
print(gscv.best_params_)
xgb_para = xgb_gscv.best_params_


# In[ ]:


#XGBoost Classifier Model
xgb = XGBClassifier(eta = xgb_para.get('eta'), max_depth = xgb_para.get('max_depth'), min_child_weight = xgb_para.get('min_child_weight'),
                    colsample_bytree = xgb_para.get('colsample_bytree'), n_estimators = xgb_para.get('n_estimators'), random_state = 4263, eval_metric = roc_auc_score)
#fit trained params
xgb.fit(X_train, y_train, verbose=0)
xgb_pred_prob = xgb.predict_proba(X_test)
xgb_pred = xgb.predict(X_test)

#Evaluation Metrics
metrics = ['roc', 'f1', 'acc', 'prec', 'recall']
roc = roc_auc_score(y_test, xgb_pred_prob, multi_class = 'ovo', average = 'weighted')
f1 = f1_score(y_test, xgb_pred, average = 'weighted')
acc = accuracy_score(y_test, xgb_pred)
prec = precision_score(y_test, xgb_pred, average = 'weighted')
recall = recall_score(y_test, xgb_pred, average = 'weighted')
xgb_scores = [roc, f1, acc, prec, recall]
for i in range(0, 5):
    print(metrics[i] + ':' + str(round(xgb_scores[i], 5)))

