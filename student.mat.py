import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sm = pd.read_csv('//content/student_mat.csv')
sm.head()
sm.tail()
sm.describe()
sm.info()
x_sm = sm.drop(['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'],axis=1)
x_sm
y_sm = sm['sex']
y_sm
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(x_sm, y_sm)
Xtrain.head()
Xtrain.shape #75%
Xtest.shape #25%
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()                       
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)
st.write(a)

from sklearn.metrics import classification_report
print(classification_report(ytest, y_model))

# Confusion Matrix
from sklearn.metrics import confusion_matrix 
confusion_matrix(ytest, y_model)

#Confusion Matrix
import matplotlib.pyplot as plt
from sklearn import metrics

import numpy as np
confusion_matrix = metrics.confusion_matrix(ytest, y_model)
st.write(confusion_matrix)

fig = plt.figure(figsize=(10, 4))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=np.unique(y_mc))
cm_display.plot()
plt.show()
fig=plt.figure(figsize=(10,4))
st.pyplot(fig)

from sklearn.metrics import classification_report
st.write(classification_report(ytest, y_model))