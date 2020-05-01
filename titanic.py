import pandas as pd 
import numpy as np
np.random.seed(32)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


training_data = pd.read_csv('train.csv')
testing_data = pd.read_csv('test.csv')

train,test = train_test_split(training_data,test_size=0.2,random_state=9)
print(train.shape,test.shape)
print (train.head()) 
print (train.describe())

# EDA

### P- Class
pclasses = sorted(train['Pclass'].unique())
ct = pd.crosstab(train.Pclass,train.Survived)
ct_pct =  ct.div(ct.sum(1).astype(float),axis = 0)
#ct_pct.plot(kind = 'bar',stacked = True)
#plt.show()

##  Gender 
genders = sorted(train['Sex'].unique())
gender_mapin =dict(zip(genders, range(0,len(genders)+1)))
train['Sex_Val'] = train['Sex'].map(gender_mapin)#.astype(int)
v = pd.crosstab(train.Sex_Val,train.Survived)
v_pct = v.div(v.sum(1).astype(float),axis = 0)
#v_pct.plot(kind = 'bar', stacked = True)
#plt.show()

#female survival rate across classes
female_df = train[train['Sex'] == 'female']
fd = pd.crosstab(female_df['Pclass'], train['Survived'])
fd_pct = fd.div(fd.sum(1).astype(float), axis = 0)
#fd_pct.plot.bar(stacked = True)
#plt.title('femlae survival rate')

#male survival rate across classes
male_df = train[train['Sex']== 'male']
md = pd.crosstab(male_df['Pclass'],train['Survived'])
md_pct = md.div(md.sum(1).astype(float), axis = 0)
#md_pct.plot.bar(stacked = True)
#plt.title('male survival rate')
#plt.show()

##  Embarked
#print train[train['Embarked'].isnull()]

tg = train['Embarked'].unique()
mapii = dict(zip(tg, range(len(tg)+1)))
#print mapii
train['Embarked_Val'] = train['Embarked'].map(mapii).astype(int)
#train['Embarked_Val'].hist(bins = len(tg))
#plt.show()
train['Embarked_Val'].replace(0, mapii['S'], inplace = True)
ctr = pd.crosstab(train['Embarked_Val'], train['Survived'])
ctr_pct  = ctr.div(ctr.sum(1).astype(float), axis = 0)
#ctr_pct.plot.bar(stacked = True)
#plt.show()
train.drop('Embarked_Val',axis=1,inplace=True)


#DATA WRANGLING

def clean_data(df, drop_passenger_id = True):

 sexes = sorted(df['Sex'].unique())
 gender_mapping = dict(zip(sexes, range(len(sexes))))
 df['Sex_Val'] = df['Sex'].map(gender_mapping).astype(int)
 
 embarked_loc = df['Embarked'].unique()
 embarked_mapping = dict(zip(embarked_loc, range(len(embarked_loc)+1))) 
 df[df['Embarked'].isnull()]['Embarked_Val'] = embarked_mapping['S']
 df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix = 'Embarked_Val',drop_first=True)], axis = 1)
 
 
 df['Fare'].fillna(df['Fare'].mean(),inplace=True)
 
 df['Age_Fill'] = df['Age']
 df['Age_Fill'] = df['Age'].groupby([df['Sex_Val'],df['Pclass']]).apply(lambda x: x.fillna(x.median()))
 df['Age_groups'] = pd.cut(df['Age_Fill'],[-1,0,12,18,35,60,100], labels = ['missing', 'child','teen', 'young adult', 'adult', 'senior'])
 age_dum = pd.get_dummies(df['Age_groups'],prefix='age_group',drop_first=True)
 df = pd.concat([df,age_dum],axis=1)
 
 df['Family_Size'] = df['Parch']+df['SibSp']
 
 df=pd.concat([df,pd.get_dummies(df['Pclass'],prefix='Pclass',drop_first=True)],axis=1)
 
 df.drop(['Name','Sex','Age','Pclass','Age_Fill', 'Age_groups','SibSp','Parch', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
 if drop_passenger_id:
  df.drop(['PassengerId'], axis = 1, inplace = True)
  
 return df 


train_data = clean_data(train) 
test_data = clean_data(test)

#TRAINING 
'''
clf = RandomForestClassifier(n_estimators = 100)
train_d = train_data.drop('Survived',axis=1)
train_target_d = train_data['Survived']

clf = clf.fit(train_d,train_target_d)
score = clf.score(train_d,train_target_d)
print ('train accuracy is ', score)
'''
# Using Grid Search for Hyper paramter tuning
train_d = train_data.drop('Survived',axis=1)
train_target_d = train_data['Survived']
clf = RandomForestClassifier()
param_grid =[
{'n_estimators':[80,100],'max_depth':[6,7,8],'max_features':['auto','log2'] }
 ]
grid_search = GridSearchCV(clf,param_grid,scoring='accuracy',cv=5)
grid_search.fit(train_d,train_target_d)
print (grid_search.best_params_)
print (grid_search.best_estimator_)

results = grid_search.cv_results_
for mean_scores, params in zip(results['mean_test_score'], results['params']):
 print (mean_scores, params)

new_clf =grid_search.best_estimator_

score = new_clf.score(train_d,train_target_d)
print ('train accuracy is ', score)

# TESTING 
test_d = test_data.drop('Survived',axis=1)
test_target_d=test_data['Survived']
predictions= new_clf.predict(test_d)
score_t = accuracy_score(test_target_d,predictions)

print('test accuracy is',score_t)


#For Submission 
p_id = testing_data['PassengerId']
testing_data = clean_data(testing_data)

prediction =new_clf.predict(testing_data)
df = pd.DataFrame({'PassengerId':p_id,'Survived':prediction})
df.to_csv('Titanic_prediction',index=False)

