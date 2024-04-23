import pandas as pd
data={
      'store':['a','a','b','b','c','c'],
      'product':['x','y','x','y','x','y'],
      'sales':[100,200,150,250,120,180]
      }

df=pd.DataFrame(data)

agg_data=df.groupby('store')['sales'].sum().reset_index()

print(agg_data)