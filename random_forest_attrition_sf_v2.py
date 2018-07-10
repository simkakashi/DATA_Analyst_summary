#2018/5/30: 基于现有数据的模型建立完毕，花费时间主要在数据清理上.因为没有绩效相关信息所以被动离职的测算不够精确.如果后续有新数据添加可更新.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

#import the data - offline now
data = pd.read_csv(r'/Users/guoxiujun/Documents/ANALYSIS/Attrition_Model/check20180619/train.csv')
data = data.set_index('unique_employee_id')
#change that:
#dataset = pd.get_dummies(data)
#change this:
dataset = data.astype('str')
for column in dataset.columns:
    dataset[column] = LabelEncoder().fit_transform(dataset[column])

#Splitting the data into independent and dependent variables
X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values

# Creating the Training and Test set from data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fitting RF Classification to the Training set, we'd better discuss about the # of n_estimators(trees)
#'gini or entropy', Classification so use sqrt(n_feature)
classifier = RandomForestClassifier(n_estimators = 230, criterion = 'gini', random_state = 100, max_features=7)
classifier.fit(X_train, y_train)

#need rush a definitions
df = pd.DataFrame({'ID':[0,1,2,3],'STATUS':['ACTIVE','VOL_LEAVE_RETAIN','VOL_LEAVE_NO_REATAIN','INVOL_LEAVE']})
x = pd.factorize(df.STATUS)
definitions = x[1]

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#Reverse factorize (converting y_pred from 0s,1s and 2s to active / voluntory left / involuntory left
reversefactor = dict(zip(range(4),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual_Status'], colnames=['Predicted_Status']))

#find results
importance = list(zip(dataset.columns[1:], classifier.feature_importances_))
#store to excel
importance_csv = pd.DataFrame(importance)
importance_csv.to_csv(r'/Users/guoxiujun/Documents/ANALYSIS/Attrition_Model/variables_importance_SF20180619.csv',index=False,sep=',')
##group by name
importance_temp = pd.DataFrame(importance)
name = list()
for i in dataset.columns[1:]:
    a = i[:10]
    name.append(a)
importance_temp['name'] = name
importance_temp = importance_temp.groupby('name').sum()
importance_temp = importance_temp.reset_index()
importance_temp.to_csv(r'/Users/guoxiujun/Documents/ANALYSIS/Attrition_Model/variables_importance_group_SF20180619.csv',index=False,sep=',')
##
#store the model
joblib.dump(classifier, 'random_forest_attrition_SF20180619.pkl')

print('RF is DONE')
#
# ## make a scatter_matrix
# plt.style.use('ggplot')
# draw_data = data.dropna(axis=0,how='any')
# c = draw_data.STATUS
# chart_scatter = pd.plotting.scatter_matrix(draw_data,diagonal = 'kde',alpha = 0.7, figsize = (16,8),c=c)
# plt.savefig(r'/Users/guoxiujun/Documents/ANALYSIS/Attrition_Model/Variables_Scatter_SF20180619.png')
# print('Saved and Show the variable_chart')
# #plt.show()
