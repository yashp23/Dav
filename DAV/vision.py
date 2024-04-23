import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("c:\java\salary_data.csv");
print(df)

x=df.iloc[:,:-1].values
y=df.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

LinearRegression()

y_pred=regressor.predict(x_test)
plt.scatter(x_train,y_train,color='green')
plt.plot(x_test,y_pred,color='red')

plt.show()