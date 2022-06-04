import pandas as pd
import numpy as np

list= ['/home/jordan/Documents/DataSets/DimensionData/01_02.csv','/home/jordan/Documents/DataSets/DimensionData/31_01.csv','/home/jordan/Documents/DataSets/DimensionData/30_01.csv', 
'/home/jordan/Documents/DataSets/DimensionData/13_01.csv', '/home/jordan/Documents/DataSets/DimensionData/14_01.csv', '/home/jordan/Documents/DataSets/DimensionData/15_01.csv',
 '/home/jordan/Documents/DataSets/DimensionData/16_01.csv', '/home/jordan/Documents/DataSets/DimensionData/17_01.csv','/home/jordan/Documents/DataSets/DimensionData/18_01.csv']

for i in list:
    with open(i, 'r+') as df:
        lines=df.readlines()
        df.seek(0)
        lines = df.writelines(lines[1:])