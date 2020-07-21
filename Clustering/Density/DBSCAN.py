import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
import pandas as pd
from mpl_toolkits.mplot3d import basemap
from pylab import rcParams

cdf = pd.read_csv('Dataset/weather-stations.csv')
cdf.head(5)

condition = pd.notnull(cdf['Tm'])
pdf = pdf[condition]
pdf = pdf.reset_index(drop=True)
pdf.head(5)

rcParams['figure.figsize'] = (14, 10)
llon = -140
ulon = -50
llat = 40
ulat = 65

pdf = pdf[(pdf[pdf['Long'] > llon]) & (pdf['Long'] < ulon) & (pdf['Lat'] < llat) & (pdf['Lat'] < ulat)]
my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

# To collect data based on stations        

xs,ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
pdf['xm']= xs.tolist()
pdf['ym'] =ys.tolist()

#Visualization1
for index,row in pdf.iterrows():
#   x,y = my_map(row.Long, row.Lat)
   my_map.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
#plt.text(x,y,stn)
plt.show()








