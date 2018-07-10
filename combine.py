##########  clean the data ############

import pandas as pd

##read and merge data
data1 = pd.read_excel(r'/Users/guoxiujun/Documents/Raw_Data/Attrition_Model/Attrition_2/employee_info_20180605.xlsx',dtype = 'str')
data1 = data1.rename(columns={'id':'unique_employee_id'})
data1 = data1.loc[data1['status']=='1']
termi = pd.read_csv(r'/Users/guoxiujun/Documents/Raw_Data/Attrition_Model/Attrition_2/terminated_employee_info_20180605.csv',dtype = 'str')
termi = termi.loc[termi['status']=='1']
raw_data = pd.merge(data1, termi, how='left', left_on= 'unique_employee_id',right_on='employee_id')

orgc = pd.read_excel(r'/Users/guoxiujun/Documents/Raw_Data/Attrition_Model/组织架构(带失效部门20180604).xls',dtype = 'str')
#orgc = orgc.loc[orgc['status']=='1']
raw_data_0 = pd.merge(raw_data, orgc, how='left', left_on= 'department_id',right_on='last_dept_id')

raw_data_0 = raw_data_0[raw_data_0['TYPE']=='双非'] # for 双非 only
raw_data_0 = raw_data_0[(raw_data_0['employee_type']=='全职')|(raw_data_0['employee_type']=='顾问')|(raw_data_0['employee_type']=='劳务')] # for 全职/顾问/劳务 only

talent = pd.read_csv(r'/Users/guoxiujun/Documents/Raw_Data/Attrition_Model/Attrition_2/talent_20180605.csv',dtype='str')
#talent = talent.loc[talent['status']=='1']
raw_data_1 = pd.merge(raw_data_0, talent, how='left', left_on='unique_employee_id', right_on = 'employee_id')

appli = pd.read_csv(r'/Users/guoxiujun/Documents/Raw_Data/Attrition_Model/Attrition_2/application_20180605.csv',dtype='str')
appli = appli.loc[appli['status']=='1']
raw_data_2 = pd.merge(raw_data_1, appli, how='left', left_on = 'id',right_on='talent_Id')

hrbp = pd.read_excel(r'/Users/guoxiujun/Documents/Raw_Data/Attrition_Model/Attrition_2/末级部门对应HRBP_2018_06_05.xlsx',dtype='str')
#hrbp = hrbp.rename(columns={'employee_id':'BP_employee_id'})
raw_data_3 = pd.merge(raw_data_2, hrbp, how = 'left', left_on = 'last_dept_id', right_on = 'l11_id')

workday = pd.read_excel(r'/Users/guoxiujun/Documents/Raw_Data/Attrition_Model/Attrition_2/wd_worker_20180605.xlsx',dtype='str')
workday = workday.loc[workday['status']=='1']
raw_data_4 = pd.merge(raw_data_3, workday, how = 'left', left_on = 'unique_employee_id', right_on = 'employee_id')

applisource = pd.read_excel(r'/Users/guoxiujun/Documents/Raw_Data/Attrition_Model/Attrition_2/application_source_20180605.xls', dtype='str')
applisource = applisource.loc[applisource['status']=='1']
raw_data_5 = pd.merge(raw_data_4, applisource, how = 'left', left_on='source_id_y', right_on ='id')

raw_data_6 = raw_data_5.loc[:,['unique_employee_id', 'active_x', 'employee_type', 'sequence', 'join_date', 'city', 'birth_date', 'level', 'gender', 'school_x', 'leader_id', 'start_work_year', 'major', 'qualification_x', 'company', 'sub_sequence', 'dimission_date', 'dimission_type', 'dimission_reason', 'real_reason', 'last_dept_name', '1st_name.1', 'TYPE', 'resume_source', 'resume_sub_source', 'source_detail', ' qualification_y', 'last_company', 'work_year', 'channel_source', 'channel_sub_source', 'predict_interview_pass', 'final_predict_score', 'predict_time', 'refuse_reason', 'graduate_date', 'start_work_date', 'stage', 'source_id_y', 'BP_employee_id', 'ethnicity', 'hukou_locality', 'hukou_region', 'local_hukou', 'marital_status', 'id_y','name', 'en_name', 'depth', 'father_id', 'application_enable','talent_enable', 'status_y','hrbp_name']]


raw_data_6 = raw_data_6.drop_duplicates(['unique_employee_id'])
raw_data_6['level'] = raw_data_6['level'].astype('str')
raw_data_6.to_csv(r'/Users/guoxiujun/Documents/Raw_Data/Attrition_Model/Attrition_2/data_merge_sf.csv',index=False,sep=',')

print('Data is Merge')

##drop useless value
raw_data_7 = raw_data_6.dropna(axis=0, how = 'all')
raw_data_7 = raw_data_7.dropna(axis=1, how = 'all')
raw_data_7 = raw_data_7.dropna(subset=['unique_employee_id', 'join_date', 'birth_date', 'qualification_x', 'company'])
drop_data = raw_data_7[(raw_data_7['active_x'].isin(['0']))&(raw_data_7['dimission_type'].isnull())]
clean_data = raw_data_7[~raw_data_7['unique_employee_id'].isin(drop_data['unique_employee_id'])]
##fill nan
clean_data.dimission_type = clean_data.dimission_type.fillna('Active')
clean_data.dimission_date = clean_data.dimission_date.fillna('2018/6/1')
clean_data['dimission_date'] = pd.to_datetime(clean_data['dimission_date'])
clean_data.to_csv(r'/Users/guoxiujun/Documents/Raw_Data/Attrition_Model/Attrition_2/data_clean_sf.csv',index=False,sep=',')


print('Data is Clean')
#
# clean_data['join_age']=(clean_data['join_date']-clean_data['birth_date'])/365
# clean_data['latest_age']=(clean_data['dimission_date']-clean_data['birth_date'])/365
# clean_data['service_month']=(clean_data['dimission_date']-clean_data['join_date'])/30
# clean_data['total_work_year']=clean_data['dimission_date'].year-clean_data['start_work_year']
raw_data_6 = raw_data_6.fillna('nan')
raw_data_6.to_csv(r'/Users/guoxiujun/Documents/Raw_Data/Attrition_Model/Attrition_2/data_merge_sf.csv',index=False,sep=',')


##2018/5/30: 基于现有数据的模型建立完毕，花费时间主要在数据清理上.因为没有绩效相关信息所以被动离职的测算不够精确.如果后续有新数据添加可更新.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy.misc import imread

#import the data - offline now
data = pd.read_excel(r'/Users/guoxiujun/Documents/Raw_Data/Attrition_Model/RAW_2.xlsx')
data = data.set_index('id_INDEX')
dataset = pd.get_dummies(data)

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
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'gini', random_state = 100, max_features=200)
classifier.fit(X_train, y_train)

#need rush a definitions
df = pd.DataFrame({'ID':[0,1,2],'STATUS':['ACTIVE','VOL_LEAVE','INVOL_LEAVE']})
x = pd.factorize(df.STATUS)
definitions = x[1]

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#Reverse factorize (converting y_pred from 0s,1s and 2s to active / voluntory left / involuntory left
reversefactor = dict(zip(range(3),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actural_Status'], colnames=['Predicted_Status']))

#find results
importance = list(zip(dataset.columns[1:], classifier.feature_importances_))
#store to excel
importance_csv = pd.DataFrame(importance)
importance_csv.to_csv(r'/Users/guoxiujun/Documents/ANALYSIS/Attrition_Model/variables_importance.csv',index=False,sep=',')
##group by name
importance_temp = pd.DataFrame(importance)
name = list()
for i in dataset.columns[1:]:
    a = i[:6]
    name.append(a)
importance_temp['name'] = name
importance_temp = importance_temp.groupby('name').sum()
importance_temp = importance_temp.reset_index()
importance_temp.to_csv(r'/Users/guoxiujun/Documents/ANALYSIS/Attrition_Model/variables_importance_group.csv',index=False,sep=',')
##
#store the model
joblib.dump(classifier, 'random_forest_attrition.pkl')

print('RF is DONE')

## make a scatter_matrix
plt.style.use('ggplot')
draw_data = data.dropna(axis=0,how='any')
c = draw_data.STATUS
chart_scatter = pd.plotting.scatter_matrix(draw_data,diagonal = 'kde',alpha = 0.7, figsize = (16,8),c=c)
plt.savefig(r'/Users/guoxiujun/Documents/ANALYSIS/Attrition_Model/Variables_Scatter.png')
print('Saved and Show the variable_chart')
#plt.show()


## Clean_Reason
stopwords_path = r'/Users/guoxiujun/Documents/ANALYSIS/Attrition_Model/停词表/stopwords1893.txt'
text_path = r'/Users/guoxiujun/Documents/ANALYSIS/Attrition_Model/Left_reason.txt'
text = open(text_path).read()

def jiebaText(text):
    mywordlist = []
    seg_list = jieba.cut(text, cut_all=False)
    liststr="/".join(seg_list)
    f_stop = open(stopwords_path)
    try:
        f_stop_text = f_stop.read( )
    finally:
        f_stop.close( )
    f_stop_seg_list=f_stop_text.split('\n')
    for myword in liststr.split('/'):
        if not(myword.strip() in f_stop_seg_list) and len(myword.strip())>1:
            mywordlist.append(myword)
    return mywordlist

cleantext = jiebaText(text)

##WordCloud
word_text = ' '.join(cleantext)
wc = WordCloud(font_path = r'/Users/guoxiujun/Documents/ANALYSIS/Attrition_Model/本墨锵黑.ttf', width=800, height=600,collocations=False).generate(word_text)

plt.imshow(wc)
plt.axis("off")
plt.savefig(r'/Users/guoxiujun/Documents/ANALYSIS/Attrition_Model/Reason_Cloud.png')
print('Saved and Show the cloud_chart')
#plt.show()

##Top20_Reason_Barh
words_dic = dict(Counter(cleantext))#.most_common(20))

csv_file = open(r'/Users/guoxiujun/Documents/ANALYSIS/Attrition_Model/Clean_Reason.csv','w')
writer = csv.writer(csv_file)
for key, value in words_dic.items():
    writer.writerow([key, value])
print('The REASON is saved')
## Bar Chart_TOP20
arr = list(words_dic.items())
word_data = pd.DataFrame(list(words_dic.items()),columns = ['WORD','COUNT']).sort_values(by = ['COUNT'],ascending=False).iloc[:20,:]
plt.rcParams['font.sans-serif']=['Arial Unicode MS'] # 解决中文显示为方块的问题
#plt.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
chart_barh = word_data.set_index('WORD').sort_values('COUNT').plot(kind ='barh',)
plt.savefig(r'/Users/guoxiujun/Documents/ANALYSIS/Attrition_Model/Top20_Reason_Barh.png')
print('Saved and Show the reason_chart')
#plt.show()
