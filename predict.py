#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pymysql
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn import preprocessing
import datetime
from scipy.interpolate import interp1d


host_name = 'solarproject-db.cpbktrao1hsy.ap-northeast-2.rds.amazonaws.com'
host_port = 3306
username = 'admin'
password = 'com404404'
database_name = 'webtest'

db = pymysql.connect(
    host=host_name,     # MySQL Server Address
    port=host_port,     # MySQL Server Port
    user=username,      # MySQL username
    passwd=password,    # password for MySQL username
    db=database_name,   # Database name
    charset='utf8'
)

#커서로 mysql 실행
cursor = db.cursor()

#sql문 
select_SQL = "SELECT * FROM solarData"
insert_SQL = "INSERT INTO expectSolar (expectVal) VALUES (105)"
select_exsolar = "SELECT * FROM expectSolar"

#cursor.execute(insert_SQL)
#db.commit()

#데이터 테이블에 추가
#cursor.execute(select_exsolar)

#배열형태로 레코드 저장
#result = cursor.fetchall()


df = pd.read_sql(select_SQL, db)

finalReal = df.iloc[:24, 1:]

print(finalReal)
#print(result)

cursor.close()
db.close()


# In[45]:


#h5 모델 가져오기

model = keras.models.load_model('test2model.h5')

#finalReal.index=finalReal['Date']
#finalReal.index=pd.to_datetime(finalReal.index)
#finalReal.drop(columns='Date',inplace=True)

#finalReal
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
# 스케일을 적용할 column을 정의합니다.
scale_cols = ['month', 'day', 'hour', 'sol','sun', 'wspd', 'hum','pow']
print(finalReal)
# 스케일 후 columns
scaled = scaler.fit_transform(finalReal[scale_cols])
print(scaled)
dfscaled = pd.DataFrame(scaled, columns=scale_cols)
#testX = np.reshape(scaled.shape[0], 24, scaled.shape[8])
#print(dfscaled.shape)

#ex_test_data = tf.expand_dims(dfscaled, axis =1)


# In[57]:


#데이터 셋 만들어주기

def windowed_dataset(x, window_size):
    print(x)
    ds_x = tf.data.Dataset.from_tensor_slices(x)
    print(ds_x)
    ds_x = ds_x.window(WINDOW_SIZE, shift=1,stride=1, drop_remainder=True)
    print(ds_x)
    ds_x = ds_x.flat_map(lambda x: x.batch(WINDOW_SIZE))
    print(ds_x)
    
    #ds = tf.data.Dataset.zip((ds_x))
    return ds_x.batch(1).prefetch(1)


WINDOW_SIZE=24

test_data = windowed_dataset(dfscaled, WINDOW_SIZE)
print(test_data)

# 데이터 shape, 미리보기로 체크
for x in test_data.take(3):
    print('X:', x.shape)
    print(x)
    print('-'*100)


# In[58]:


a=model.predict(test_data)
print(a)


# In[ ]:




