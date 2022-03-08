import pandas as pd
import numpy as np
import os
import tensorflow as tf
import time
import datetime

cf15= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/cf15.csv', low_memory=False, header=None)
gf25= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/gf25.csv', low_memory=False, header=None)
gf34= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/gf34.csv', low_memory=False, header=None)
gm01= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/gm01.csv', low_memory=False, header=None)
hc10= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/hc10.csv', low_memory=False, header=None)
nc46= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/nc46.csv', low_memory=False, header=None)
nc74= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/nc74.csv', low_memory=False, header=None)
nf11= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/nf11.csv', low_memory=False, header=None)
no22= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/no22.csv', low_memory=False, header=None)
wc16= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/wc16.csv', low_memory=False, header=None)
wc41= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/wc41.csv', low_memory=False, header=None)
wo27= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/wo27.csv', low_memory=False, header=None)
wo34= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/wo34.csv', low_memory=False, header=None)
wo36= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/wo36.csv', low_memory=False, header=None)
wo37= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/wo37.csv', low_memory=False, header=None)
wo40= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/wo40.csv', low_memory=False, header=None)
wo45= pd.read_csv('/home/jordan/Documents/DataSets/DimensionData/wo45.csv', low_memory=False, header=None)

frames =[cf15, gf25,gm01,nc46, nc74,nf11, wc16, wc41,wo27,wo34, wo36, wo37, wo40, wo45]
data = pd.concat(frames)


def snip(data):
    #Snip unwanted characters 

    itime= data.iloc[:,0]
    itime=itime.map(lambda x: str(x[6:]))
    itime=itime.map(lambda x: float(x))

    date=itime
    minute=date.map(lambda x: time.gmtime(x+7200)[4])
    hour=date.map(lambda x: time.gmtime(x+7200)[3])
    day=date.map(lambda x: time.gmtime(x+7200)[2])
    month=date.map(lambda x: time.gmtime(x+7200)[1])

    interface= data.iloc[:,-14]
    interface=interface.map(lambda x: str(x))
    interface=interface.map(lambda x: str(x[15:]))
    interface=interface.map(lambda x: str(x[:-1]))

    dev_id=data.iloc[:,3]
    dev_id=dev_id.map(lambda x: str(x))    
    dev_id=dev_id.map(lambda x: str(x[7:]))
    dev_id=dev_id.map(lambda x: str(x[:-1]))

    jitter=data.iloc[:, -13]
    jitter=jitter.map(lambda x: str(x))
    jitter=jitter.map(lambda x: str(x[8:]))
    jitter=jitter.map(lambda x: str(x[:-1]))
    jitter=jitter.map(lambda x: float(x))

    latency= data.iloc[:,-12]
    latency=latency.map(lambda x: str(x))
    latency=latency.map(lambda x: str(x[9:]))
    latency=latency.map(lambda x: str(x[:-1]))
    latency=latency.map(lambda x: float(x))

    inband= data.iloc[:,-15]
    inband=inband.map(lambda x: str(x))
    inband=inband.map(lambda x: str(x[22:]))
    inband=inband.map(lambda x: str(x[:-5]))
    inband=inband.map(lambda x: float(x))

    outband= data.iloc[:, -5]
    outband=outband.map(lambda x: str(x))
    outband=outband.map(lambda x: x.split('"')[1])
    outband=outband.map(lambda x: str(x[:-4]))
    outband=outband.map(lambda x: float(x))

    packet=data.iloc[:,-4]
    packet=packet.map(lambda x: str(x))
    packet=packet.map(lambda x: str(x[12:]))
    packet=packet.map(lambda x: str(x[:-2]))
    packet=packet.map(lambda x: float(x))

    status=data.iloc[:,-2]
    status=status.map(lambda x: str(x))
    status=status.map(lambda x: str(x[8:]))
    status=status.map(lambda x: str(x[:-1]))

    status[:] = np.where(status=="up", 1,0)
    status=status.map(lambda x: float(x))
    #status=status.rolling(20).mean()
    #status=status.map(np.floor)

    train_init=pd.concat([itime, minute, hour, day, month, interface, dev_id, inband, outband, latency, packet, jitter, status], axis=1)
    train_init.columns=['Time','Minute', 'Hour','Day','Month', 'Interface', 'Dev_id', 'Inband', 'Outband', 'Latency', 'Packet', 'Jitter', 'Status']

    #Dedupp based on interfaces
    train=train_init[train_init.Interface != 'NI2A-WWW1']
    train=train[train.Interface != 'NI2B-WWW1']
    train=train[train.Interface != 'NI2A-WWW2']
    train=train[train.Interface != 'NI2B-WWW2']
    train=train[train.Interface != 'NI2A-WWW3']    
    train=train[train.Interface != 'NI2B-WWW3']
    train=train[train.Interface != 'NI2A-MPLS']
    train['Interface']= train['Interface'].astype('category')
    train['Interface']=train['Interface'].cat.codes

    #Remove unwanted columns
    train_label=train.filter(['Status'])
    train=train.drop(['Status','Inband', 'Outband'], axis=1)
    train= train.sort_values(by=['Dev_id', 'Time'])
    train=train.reset_index(drop=True)
    train=train.drop(['Dev_id'], axis=1)

    return train, train_label

train,train_label=snip(data)
train=train.iloc[:-700, :]
train.to_csv(path_or_buf='/home/jordan/Downloads/train1')
train=train.to_numpy()
train_label=train_label.iloc[700:]

print(train)