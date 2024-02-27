# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

 Neural Network regression model is a type of machine learning algorithm inspired by the structure of the brain. It excels at identifying complex patterns within data and using those patterns to predict continuous numerical values.his includes cleaning, normalizing, and splitting your data into training and testing sets. The training set is used to teach the model, and the testing set evaluates its accuracy. This means choosing the number of layers, the number of neurons within each layer, and the type of activation functions to use.The model is fed the training data.Once trained, you use the testing set to see how well the model generalizes to new, unseen data. This often involves metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).Based on the evaluation, you might fine-tune the model's architecture, change optimization techniques, or gather more data to improve its performance.

## Neural Network Model

![image](https://github.com/Rakshithadevi/basic-nn-model/assets/94165326/c4bd48d6-f353-4b4f-a5b8-fbe20f7a2d95)


This layer has a single neuron (R¹ means it accepts one-dimensional data). This suggests your dataset likely has just one feature or predictor variable.Include the neural network model diagram.There are two hidden layers, each with two neurons (R² means each layer has two-dimensional output). These layers process the input data and learn complex patterns within it.The final layer has one neuron, indicating the model predicts a single numerical value, typical for regression problems.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:
### Register Number:
```
Name:Rakshitha Devi J
reg no:212221230082
```
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('EX1').sheet1

data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'input':'float'})
dataset1 = dataset1.astype({'output':'float'})
dataset1.head(20)

import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = dataset1[['input']].values
y = dataset1[['output']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train1 = scaler.transform(X_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units=2,activation='relu',input_shape=[1]),
    Dense(units=2,activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='rmsprop',loss='mse')
model.fit(X_train1,y_train,epochs=10000)

loss= pd.DataFrame(model.history.history)
loss.plot()

X_test1 =scaler.transform(X_test)
model.evaluate(X_test1,y_test)

X_n1=[[10]]
X_n1_1=scaler.transform(X_n1)
model.predict(X_n1_1)


```
## Dataset Information

![image](https://github.com/Rakshithadevi/basic-nn-model/assets/94165326/f9623860-f4a2-4a52-915a-e40892ac76fa)


## OUTPUT
### Epoch Training:

![image](https://github.com/Rakshithadevi/basic-nn-model/assets/94165326/87fdb53a-5fdd-4a36-899e-b8e77b007e96)


### Training Loss Vs Iteration Plot

![image](https://github.com/Rakshithadevi/basic-nn-model/assets/94165326/0f48c49a-e64a-4434-834e-3c9a2459ca98)


### Test Data Root Mean Squared Error

![image](https://github.com/Rakshithadevi/basic-nn-model/assets/94165326/2cd87753-f4f0-4205-9696-57d0cbd6e94d)


### New Sample Data Prediction

![image](https://github.com/Rakshithadevi/basic-nn-model/assets/94165326/65af48d4-e8c2-4402-8616-c767d28652f1)

## RESULT

Thus the neural network regression model for the given dataset is executed successfully.
