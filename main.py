import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score

df=pd.read_csv("mail_data.csv")
# print(df)
data =df.where((pd.notnull(df)),'') #only taking valid values
# print(data.head(10))

#data.info()
#data.shape()
data.loc[data['Category']=='spam','Category',]=0
data.loc[data['Category']=='ham','Category',]=1 #category defife  for binary

X=data['Message']
Y=data['Category']

# print(X)
# print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)
#test size=0.2  20 %for test  80 % for train  randomstate hyper parameter for interge value to specify the split in the data
# print(X.shape) #total data
# print(X_train.shape) #train data 80 %
# print(X_test.shape) #testisng  data 20 %

#TRANFORM  TEXT DATA TO FEATURE DATA SO USE AS INPUT TO LOGISTIC REGRESSION
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

#stop words are the english words that doesnot mean much to the text /sentence safely be ignored
X_train_featres=feature_extraction.fit_transform(X_train)
X_test_featres=feature_extraction.transform(X_test)
#Y TRAIN AND TEST VALUES AS INTEGER

Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')
 # print(X_train)
# print(X_train_feautres)
#train the model
Model=LogisticRegression()
Model.fit(X_train_featres,Y_train)
prediction_on_training_data=Model.predict(X_train_featres)
accuracy_on_training_Data=accuracy_score(Y_train,prediction_on_training_data)
print('acc on training data',accuracy_on_training_Data)
prediction_on_test_data=Model.predict(X_test_featres)
accuracy_on_test_Data=accuracy_score(Y_test,prediction_on_test_data)
print('accuracy on test data ',accuracy_on_test_Data)

# input_your_mail=['']
# input_data_featres=feature_extraction.transform(input_your_mail)
# prediction=Model.predict(input_data_featres)
# print(prediction)
# if (prediction[0]==0):
#     print('its is a spam mail ')
# else:
#     print("its ham mail ")
#Saving model in binary
# import pickle
# with open ('model','wb') as file:
#      pickle.dump(Model,file)

while True:
    user_input = input("Enter the mail you want to check (type 'exit' to stop): ")

    if user_input.lower() == 'exit':
        break

    input_data_features = feature_extraction.transform([user_input])
    prediction = Model.predict(input_data_features)

    if prediction[0] == 0:
        print('It is a spam mail.')
    else:
        print('It is a ham mail.')
