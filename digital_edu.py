import pandas as pd
import random
df = pd.read_csv('train.csv')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

print(df['career_end'].value_counts())

df.drop(['langs', 'city', 'last_seen', 'occupation_name'], axis = 1, inplace = True)

def god(z):
    z=str(z)
    if len(z)>=4:
        x=z[-4]+z[-3]+z[-2]+z[-1]
        if '.' in x:
            return 0
        else:
            return int(x)

def god2(z):
    global zzz
    if z==0:
        return int(zzz)

def status1(z):
    if z=="None":
        return 0
    elif z=="Undergraduate applicant":
        return 1
    elif z=="Student (Bachelor's)":
        return 2
    elif z=="Alumnus (Bachelor's)" or z=="Alumnus (Specialist)":
        return 3
    elif z=="Student (Master's)":
        return 4
    elif z=="Alumnus (Master's)":
        return 5
    elif z== "Candidate of Sciences":
        return 6
    elif z=="PhD":
        return 7

def form1(z):
    if z=="None":
        return 0
    elif z=="Distance Learning":
        return 1
    elif z=="Part-time ":
        return 2
    elif z=="Full-time":
        return 3

df['bdate'] = df['bdate'].apply(god)
df['bdate'].fillna(0, inplace = True)
df['bdate'].replace(0, 1988,inplace=True)

df['education_status'] = df['education_status'].apply(status1)
df['education_status'].fillna(0, inplace = True)

df['education_form'] = df['education_form'].apply(form1)
df['education_form'].fillna(0, inplace = True)

df['life_main'].replace("False", 0,inplace=True)
df['people_main'].replace("False", 0,inplace=True)
df['occupation_type'].replace(['work','university'],[0,1],inplace=True)
df['occupation_type'].fillna(0, inplace = True)

df['career_start'].replace("False", 2007,inplace=True)
df['career_end'].replace("False", 2009,inplace=True)
# print(df.info())

# print(df['people_main'].value_counts())

X_train = df.drop('result', axis = 1)
y_train = df['result']

df2 = pd.read_csv('test.csv')

print(df2['career_start'].value_counts())

# print(df2['career_start'].value_counts())
# print(df2.info())

df2.drop(['langs', 'city', 'last_seen', 'occupation_name'], axis = 1, inplace = True)

df2['education_status'] = df2['education_status'].apply(status1)
df2['education_status'].fillna(0, inplace = True)

df2['education_form'] = df2['education_form'].apply(form1)
df2['education_form'].fillna(0, inplace = True)

df2['bdate'] = df2['bdate'].apply(god)
df2['bdate'].fillna(0, inplace = True)
df2['bdate'].replace(0, 1988,inplace=True)

df2['life_main'].replace("False", 0,inplace=True)
df2['people_main'].replace("False", 0,inplace=True)

df2['occupation_type'].replace(['work','university'],[0,1],inplace=True)
df2['occupation_type'].fillna(0, inplace = True)
df2['career_start'].replace("False", 2008,inplace=True)
df2['career_end'].replace("False", 2017,inplace=True)


X_test =df2
 
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
 
y_pred = classifier.predict(X_test)

print(y_pred)

ID = df2['id']

df3 = pd.DataFrame({'ID': ID, 'result': y_pred})

df3.to_csv('rez.csv', index = False)

# print(df.info())