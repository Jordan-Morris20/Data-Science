import pandas as pd
import numpy as np

list= ['/home/jordan/Documents/DataSets/DimensionData/19_01.csv', '/home/jordan/Documents/DataSets/DimensionData/20_01.csv',
'/home/jordan/Documents/DataSets/DimensionData/21_01.csv','/home/jordan/Documents/DataSets/DimensionData/22_01.csv','/home/jordan/Documents/DataSets/DimensionData/23_01.csv', 
'/home/jordan/Documents/DataSets/DimensionData/24_01.csv', '/home/jordan/Documents/DataSets/DimensionData/25_01.csv', '/home/jordan/Documents/DataSets/DimensionData/26_01.csv',
 '/home/jordan/Documents/DataSets/DimensionData/27_01.csv', '/home/jordan/Documents/DataSets/DimensionData/28_01.csv','/home/jordan/Documents/DataSets/DimensionData/29_01.csv']

for i in list:
    with open(i, 'r+') as df:
        lines=df.readlines()
        df.seek(0)
        lines = df.writelines(lines[1:])