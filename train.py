import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dt
from ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def load( filename):
    with open(filename, "rb") as f:
        data = pickle.load( f)
        return data


n_limit = 500



features_face = load('features_face')
features_face_array = np.array(features_face)[0:n_limit]


n_pos_sample = features_face_array.shape[0]
n_feature = features_face_array.shape[1]


features_nonface = load('features_nonface')
features_nonface_array = np.array(features_nonface)[0:n_limit]
n_neg_sample = features_nonface_array.shape[0]

X = np.concatenate((features_face_array,features_nonface_array) ,axis=0)

y = np.array([1]*n_pos_sample + [-1]*n_neg_sample)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)



# model = dt()
# model.fit(X_train,y_train)
# s = model.score(X_test,y_test)
# print(s)

Ada = AdaBoostClassifier(dt,10)
Ada.fit(X_train,y_train)
pred = Ada.predict(X_test)
acc = accuracy_score(pred,y_test)
print('acc:',acc)

f = open('report.txt','w')
content = classification_report(pred,y_test)
f.write(content)
f.close()




