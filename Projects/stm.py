import streamlit as stm
import pandas as pd
import numpy as np
import tensorflow
import matplotlib.pyplot as plt

# Title
stm.title("Crude Price Forecating")

stm.sidebar.title("Inputs from User")

df= pd.read_csv("crudeprice.csv")
df.columns = ["Date","Price"]
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Price"]!="."]
df["Price"] = df["Price"].apply(lambda x: float(x))
df1 = df.reset_index()['Price']
df.index = df["Date"]


# scaling data

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()

df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

#train- test split
w = 100
train_size = int(len(df1)*0.95)
test_size = len(df1) - train_size 

train_data = df1[:train_size+w:]
test_data = df1[train_size::]

def create_dataset(dataset,time_step):
    dataX,dataY = [], []

    for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])

    return np.array(dataX), np.array(dataY)

test_data = np.array(test_data).reshape(-1,1)
X_train, y_train = create_dataset(train_data,100)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test, y_test = create_dataset(test_data,100)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

model = tensorflow.keras.models.load_model("crude_forec.h5")

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

stm.markdown("Model is loaded")

x_input = test_data[-100:].reshape(1,-1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

# Number of days to forecast
num_days = stm.sidebar.number_input("Number of days to forecast", min_value=1, step=1)
num_days = int(num_days)

b = stm.sidebar.button("Forecast")

if b:
    lst_output = []
    n_steps = 100
    i = 0

    while i < num_days:
        if(len(temp_input)>100):
            #print(temp_input)
            x_input = np.array(temp_input[1:]) # Taking x_input values from 2nd value onward, so that total value will be 100
            print('{} day input {}'.format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1,n_steps,1)) #making tensor of 1 batch, with n rows and 1 column
            #print(x_input)
            yhat = model.predict(x_input,verbose = 1)
            print('{} day output {}'.format(i,yhat))
            temp_input.extend(yhat[0].tolist()) #Adding forecasted value to the temp_input, for further forecasting, now there are 102 values in temp_input
            temp_input = temp_input[1:] #Because after adding the above yhat[0], total number of elements in temp_input is 102, so we will select last 101 elements so that again if loop will go on running for 30 days
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i = i+1
        else:        #first loop will go inside this
            x_input = x_input.reshape((1,n_steps,1))  #last 100 days data, nsteps = 100 and reshaping it so that we can feed it in LSTM
            yhat = model.predict(x_input,verbose = 0) #Taking prediction from model 
            print(yhat[0])
            temp_input.extend(yhat[0].tolist()) #Adding predicted value of 101 day in temp_input, so that this value can be used for forecasting values for days starting from day 102
            print(len(temp_input))
            lst_output.extend(yhat.tolist())  #Adding 101 day forecast to Output forecasting list 
            i = i+1
    
    forecast = scaler.inverse_transform(lst_output)

    forecast = forecast.reshape(num_days,)
    ddf = df[100:]
    t_fut = pd.date_range('2023-03-14 00:00:00+00:00',periods=num_days)

    dffor = pd.DataFrame({'Forecast Price':forecast},index = t_fut)
    stm.markdown(f"Forecasting Plot")
    plt.figure(figsize=(16,7))

    fig,ax = plt.subplots()

    ax.grid(True)
    ax.set_title("Crude Price Forecating Training and Validation")
    ax.plot(df.index,df['Price'],label = "Original Value",c = "blue")
    ax.plot(df[w:train_predict.shape[0]+w].index,train_predict,label = "Predicted Training Price",c = "Red")
    ax.plot(df[train_predict.shape[0]+w:].index,test_predict,label = "Predicted Validation Price",c = "Green")

    ax.plot(dffor.index,dffor['Forecast Price'],label = 'Forecasted Price',c = 'black')

    ax.axvline(df.index[train_predict.shape[0]+w], color='black',lw=3)
    ax.axvline(dffor.index[0],color = 'black',lw=2)

    stm.pyplot(fig)
    
