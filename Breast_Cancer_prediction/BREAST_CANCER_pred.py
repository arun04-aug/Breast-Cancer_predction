import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

print("ALL CLEAR >")

# PATH UPLOADING>>>

path = r'C:\Users\arunsathya\Desktop\ML\excel_data\data.csv'

dataset = pd.read_csv(path)
print(dataset.head(5))

print(dataset.shape)

"""a=dataset.columns[dataset.isna().any()]
print(a)
dataset.['Unnamed: 32']=dataset.['Unnamed: 32'].fillna(dataset.['Unnamed: 32'].mean())"""



# MAPPING>>>

if 'Unnamed: 32' in dataset.columns:
    dataset = dataset.drop(['Unnamed: 32'], axis=1)
dataset = dataset.dropna()

dataset['diagnosis'] = dataset['diagnosis'].map({'B':0,'M':1}).astype(int)
X = dataset.iloc[:,2:].values
Y = dataset.iloc[:,1].values

# SPLITING >>>>

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)


# Feature Scaling...........

sc=StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Algorthim

models = []

models.append(('LR',LogisticRegression(solver = 'liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('svm',SVC(gamma='auto')))
models.append(('CART',DecisionTreeClassifier()))

result=[]
names=[]
res=[]
for name, model in models:
    kfold=StratifiedKFold(n_splits=10,random_state=None)
    cv_res=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    result.append(cv_res)
    names.append(name)
    res.append(cv_res.mean())
    print('%s:%f'%(name,cv_res.mean()))

plt.ylim(.900,.999)
plt.bar(names,res,color='maroon',width=0.6)
plt.title('AlgorithmComparsion')
plt.show()

# Training >>>

model=LogisticRegression()
model.fit(X_train,Y_train)

# Prediction >>>

Y_pred=model.predict(X_test)

#acuuracy...........

print("Accuracy : {0}%".format(accuracy_score(Y_test,Y_pred)*100))


#evalution >>>


test=[[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259

]]

re=model.predict(sc.transform(test))
print(re)

if re==1:
    print("MALINIM")
else :
    print("BELINIM")
