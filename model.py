#Importing Dependencies
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Import Dataset
df = pd.read_csv('diabetes.csv')
print(df)

df.head()

#Checking the number of row & column
df.shape

#Check for missing value
df.isnull().sum()

df["Outcome"].value_counts()

df.groupby("Outcome").mean()

#separating the data and labels
X = df.drop(columns="Outcome")
Y = df["Outcome"]

X

Y

#Data Standardization
scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

#Alternative way to fit & transform: scaler.fit_trasform(X)

standardized_data

#Việc Standardization (Chuẩn hóa $Z$-score) các đặc trưng (trục X) trước khi huấn luyện mô hình dự đoán béo phì
#là một bước tiền xử lý dữ liệu cực kỳ quan trọng.Nó được thực hiện để đảm bảo rằng tất cả các đặc trưng đầu vào
#(ví dụ: tuổi, cân nặng, chiều cao, thu nhập,...) có cùng một thang đo (scale) và trọng số ảnh hưởng ngang nhau đối
#với mô hình, từ đó cải thiện tốc độ hội tụ và độ chính xác của mô hình.

X = standardized_data
Y = df["Outcome"]

#Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)

#Training the model
classifier = svm.SVC(kernel='linear')

#Training the Support Vector Machine Classifier
classifier.fit(X_train,Y_train)

#accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accurancy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy score on training data: ", training_data_accurancy)

#accuracy on testing data
X_test_prediction = classifier.predict(X_test)
testing_data_accurancy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy score on testing data: ", testing_data_accurancy)

#Making a predictive system
input_data = (5,12,74,27,0,29,0.203,100)

#Changing the input to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array to different (2D) dimension
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardized the input data
std_data = scaler.transform (input_data_reshaped)

#print prediction
prediction = classifier.predict(std_data)
print(prediction[0])
# if prediction[0] == 0:
#   print("You're fine")
# else:
#   print("You're diabetes!!!!")


import pickle

# Save both the classifier AND the scaler (you need both!)
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(classifier, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Model and scaler saved successfully!")