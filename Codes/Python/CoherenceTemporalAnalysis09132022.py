#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:02:46 2022

@author: shanshanli
"""

#
# This file is used to generate time series files of coherence
# compare coherence with water level observation from gauge stations
# select stations or points from vegetation map
# generate scatter plots to explore correlation for each region
#  
# Shanshan Li
# 09/13/2022

#import arcpy  # ArcGIS,python二次开发
import matplotlib.ticker as ticker
import icepyx as ipx
#icepyx is the software package related to icesat-2 data
import os
#导入标准库os，利用其中的API
import shutil
import h5py
import xarray as xr
import getpass
from topolib import icesat2_data
import glob
import rasterio
from topolib import gda_lib
from topolib import dwnldArctic
import numpy as np
import geopandas as gpd
from multiprocessing import Pool
import contextily as ctx
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as ro
from rasterio.plot import reshape_as_raster, reshape_as_image
from email import utils
import datetime
import time
 #coordinate transform
import math
from pyproj import CRS
from pyproj import Transformer
from pyproj import  _datadir, datadir
import matplotlib
from shutil import copy

#function caltimedays is defined to calculate days between date1 and date2 in format 20220101
def caltimedays(date1,date2):
    date1t = datetime.datetime.strptime(date1,'%Y%m%d')
    date1 = datetime.datetime.strftime(date1t,'%Y-%m-%d')
    date1 = time.strptime(date1,"%Y-%m-%d")
    date2t = datetime.datetime.strptime(date2,'%Y%m%d')
    date2 = datetime.datetime.strftime(date2t,'%Y-%m-%d')
    date2 = time.strptime(date2,"%Y-%m-%d")
    
    #根据上面需要计算日期还是日期时间，来确定需要几个数组段，下标0表示年，下标1表示月份，以此类推
    date1= datetime.date(date1[0],date1[1],date1[2])
    date2 = datetime.date(date2[0],date2[1],date2[2])
    
    #返回两个变量相差的值，就是相差天数
    return (date2-date1).days

############# Coherence Analysis ##################################
# The file path for interested coherence files

pathvv = '/Volumes/lss2022/2022CoherencePaper/Heming/Sentinel/vv/'

#pathvv = '/Volumes/lss2022/2022CoherencePaper/Heming/Sentinel/vh/'

#pathvh = '../Heming/Sentinel/vh/'

#pathvh
pathvv

# #os.listdir 用以返回path路径下的文件和文件夹的名字列表
allpathvvs = os.listdir(pathvv)

allpathvvs[0]

rfiletest = pathvv + allpathvvs[0]
srctest = ro.open(rfiletest)
arraytest = srctest.read(1)
widtht = srctest.width
heightt = srctest.height

#allpathvhs = os.listdir(pathvh)

#allpathvhs[0]

dursdays1 = np.empty(shape=(len(allpathvvs),1))

#allcohr = np.empty(shape=())

n6files = 0
n12files = 0
n24files = 0
n36files = 0

#create four empty lists
datetexts6 = []
datetexts12 = []
datetexts24 = []
datetexts36 = []
datetexts12s = []

# store the new list showing files with repeating cycle 12 days

Cohfiles12 = []
dates12 = []

for filenum1 in range(len(allpathvvs)):
    datetextstr = allpathvvs[filenum1]
    #extract date string from a string
    datetext = datetextstr[5:22]
    
    startdate = datetextstr[5:13]
    enddate = datetextstr[14:22]
    
    durdays = caltimedays(startdate,enddate)
    dursdays1[filenum1] = durdays
    
  
    #当repeat cycle is 6 days,do the spatial analysis, calculate mean and standard deviation
    if durdays == 6:
        n6files = n6files + 1
        datetexts6.append(datetext)
    if durdays == 12:
        newfolderpath = '/Volumes/lss2022/2022CoherencePaper/Heming/Sentinel/vv12days/'
        n12files = n12files + 1
        from_path = os.path.join(pathvv,datetextstr)
        to_path = os.path.join(newfolderpath,datetextstr)
        copy(from_path,to_path)
        datetexts12.append(datetext)
        datetexts12s.append(startdate) # for sorting later
        Cohfiles12.append(datetextstr)
        
        
    if durdays == 24:
        n24files = n24files + 1
        datetexts24.append(datetext)
    if durdays == 36:
        n36files = n36files + 1
        datetexts36.append(datetext)
        
# sorting the files with 12 days, for plotting time series later
for fs1 in range(0,len(datetexts12s),1):
    date1n = datetime.datetime.strptime(datetexts12s[fs1],'%Y%m%d')
    date1 = datetime.datetime.strftime(date1n,'%Y-%m-%d')
    dates12.append(date1)
    
result_list = sorted(dates12)

result_listindex = []
result_cohfilelist =  []
result_dateslist = []
   
for p in range(0,len(result_list),1):
    for q in range(0,len(dates12),1):
        if dates12[q] == result_list[p]:
            result_listindex.append(q)

for t in range(0,len(dates12),1):
    st = result_listindex[t]
    result_cohfilelist.append(Cohfiles12[st])
    result_dateslist.append(datetexts12[st])
 
# list "result_cohfilelist" is the right order       
           
        
#obtain coherence values for specific location points
#lists that store all EDEN stations in all repeating observations for each ground track

# loop from each ground track

secEDENstations1 = []
secEDENstationslon1 = []
secEDENstationslat1 = []

file = '/Volumes/lss2022/2022Icesat2_Everglade_WaterLevelDepth/GauageStations/EDEN_stations_info.xlsx'

df_gauge = pd.read_excel(file, engine='openpyxl')

# #selected gauges
# select_gauges = ['3AN1W1', 'W11', '3AS3W1', 'P36', 'P38','SR1']
select_gauge_lon, select_gauge_lat = [], []
# for xx in select_gauges:
#     temp = df_gauge[df_gauge['EDEN Station Name']==xx].iloc[0]['Latitude (NAD83)']
#     select_gauge_lat.append(float(temp[:2])+ float(temp[3:5])/60+ float(temp[6:len(temp)-1])/60/60)
#     temp = df_gauge[df_gauge['EDEN Station Name']==xx].iloc[0]['Longitude (NAD83)']
#     select_gauge_lon.append(-( float(temp[1:3])+ float(temp[4:6])/60+ float(temp[7:len(temp)-1])/60/60 ))
    
#since we are not sure which stations should be used for each ground track, we need to plot them all and find the overlapping gauage stations
for xx in df_gauge['EDEN Station Name']:
    temp=df_gauge[df_gauge['EDEN Station Name']==xx].iloc[0]['Latitude (NAD83)']
    select_gauge_lat.append(float(temp[:2])+ float(temp[3:5])/60+ float(temp[6:len(temp)-1])/60/60)
    temp = df_gauge[df_gauge['EDEN Station Name']==xx].iloc[0]['Longitude (NAD83)']
    select_gauge_lon.append(-( float(temp[1:3])+ float(temp[4:6])/60+ float(temp[7:len(temp)-1])/60/60 ))   

# till now, we have the lon and lat for each station, and name of each station
#select_gauge_lon, select_gauge_lat, df_gauge['EDEN Station Name']      

numstations = len(df_gauge['EDEN Station Name'])  # 313
numcohwl = n12files # 75

#water level difference
# mean of water level
# coherence resampling
# current coherence: arraylon[iheight][iwidth], arraylat[iheight][iwidth], array[iheight][iwidth]
# for coherence, height = 5759   widtht = 5327
# for water level, h1 =    w1 = 

# #interpolation,method 1 is used for
# from scipy.interpolate import griddata
# def  grid_interp_to_station(all_data,station_lon,station_lat,method='nearest'):
#  	#func:将等经纬度网格值插值到离散站点，使用griddata 进行插值
#  	# inputs:
# 		#all_data,形式为 [grid_lon, grid_lat, data] 即 [经度网格， 纬度网格，数值网格]
# 		#station_lon: 站点经度
# 		#station_lat: 站点纬度，可以是单个点，列表或者一维数组
# 		# method: 插值方法，默认使用cubic
#  	station_lon = np.array(station_lon).reshape(-1,1)
#  	station_lat = np.array(station_lat).reshape(-1,1)
 	
#  	lon = all_data[0].reshape(-1,1)
#  	lat = all_data[1].reshape(-1,1)
#  	data = all_data[2].reshape(-1,1)
 	
#  	points = np.concatenate([lon,lat], axis = 1)
 	
#  	station_value = griddata(points,data,(station_lon,station_lat),method= method)
 	
#  	station_value  = station_value[:,:,0]
 	
#  	return station_value

# # re-calculate the value
# def grid_resample_to_station(arrayglon,arrayglat,arraycohern,station_lon,station_lat):
#     lon = arrayglon.reshape(-1,1)
#     lat = arrayglat.reshape(-1,1)
#     data = arraycohern.reshape(-1,1)
    
    
    
    
#     return station_cohr

# Resampling and get coherence for all image pairs of all gauge stations
# step1: get arraylon and arraylat for all coherence files
rfile =  pathvv + result_cohfilelist[0] # this is only for one coherence file example
#rfile =  pathvv + allpathvvs[filenum]
src = ro.open(rfile)
src.name

array = src.read(1)
array.shape

width = src.width
height = src.height
    
arraylon = np.empty(shape=(height,width))
arraylat = np.empty(shape=(height,width))

# store all coherence from all 6 days repeat cycle image pairs in a 3D array

for iwidth in range(0,width,1):
    for iheight in range(0,height,1):
        #获取对应行列号的像素坐标
        #np.shape(array)
        c1 = src.xy(iheight,iwidth)
        arraylon[iheight][iwidth] = c1[0]
        arraylat[iheight][iwidth] = c1[1]

#step 2: loop each file and obtain coherence for gauge stations of each image pair
#griddata(points,values,xi,method='nearest',fill_value=nan,rescale = False)
#for all coherence files, arraylon and arraylat should be the same, however, array is different for different files

gaugecoh = np.empty(shape=(numcohwl,numstations))
icohwl = 0
datetextstrs = []
gaugelonlat = []

for icohfile in range(len(result_cohfilelist)):
    
    cohfile = pathvv + result_cohfilelist[icohfile]
    srct = ro.open(cohfile)
    arraycoh = srct.read(1)
    
    datetextstr = result_cohfilelist[icohfile]
    datetextstrs.append(datetextstr)
    #extract date string from a string
    datetext = datetextstr[5:22]
    
    startdate = datetextstr[5:13]
    enddate = datetextstr[14:22]
    
    durdays = caltimedays(startdate,enddate)
    dursdays1[icohfile] = durdays
    
    if durdays == 12:
        
        for xx in range(0,numstations,1):
            gaugelonlat = [select_gauge_lon[xx],select_gauge_lat[xx]]
            rowt,colt = srct.index(*gaugelonlat)
            gaugecoh[icohfile][xx] = arraycoh[rowt][colt]
        #all_data = [arraylon, arraylat, arraycoh]
        #station_value = grid_interp_to_station(all_data, select_gauge_lon, select_gauge_lat, method='nearest')
        #for it in range(len(station_value)):
         #   gaugecoh[icohwl][it] = station_value[it]
        
    
#https://blog.csdn.net/SunStrongInChina/article/details/110577696

# longitude and latitude of gauge stations change from list to array.
stationlon = np.array(select_gauge_lon)
stationlat = np.array(select_gauge_lat)


############### Water Level/Surfaces from EDEN Gauge Stations ###################

tif_file_path = '/Volumes/lss2022/2022CoherencePaper/GauageStations/water-level-EDEN-Gauge/allwaterlevelfiles/'

gaugewld = np.empty(shape=(numcohwl,numstations))

gaugewlm = np.empty(shape=(numcohwl,numstations))

for filenum1 in range(len(result_cohfilelist)):
    datetextstr = result_cohfilelist[filenum1]
    #extract date string from a string
    datetext = datetextstr[5:22]
    
    startdate = datetextstr[5:13]
    enddate = datetextstr[14:22]
    
    wlfile1 = tif_file_path + 's_'+startdate+'.tif'
    wlfile2 = tif_file_path + 's_' +enddate + '.tif'
    
    srcwl1 = ro.open(wlfile1)
    artest1 = srcwl1.read(1)
    w1 = srcwl1.width
    h1 = srcwl1.height

    srcwl2 = ro.open(wlfile2)
    artest2 = srcwl2.read(1)
    w2 = srcwl2.width
    h2 = srcwl2.height

    # store the x and y

    ayx = np.empty(shape=(h1, w1))
    ayy = np.empty(shape=(h1, w1))

    ayx2 = np.empty(shape=(h2, w2))
    ayy2 = np.empty(shape=(h2, w2))


    for iwidth in range(0, w1, 1):
        for iheight in range(0, h1, 1):
            #获取对应行列号的像素坐标
            #np.shape(array)
            c1 = srcwl1.xy(iheight, iwidth)
            ayx[iheight][iwidth] = c1[0]
            ayy[iheight][iwidth] = c1[1]

    for iwidth in range(0, w2, 1):
        for iheight in range(0, h2, 1):
            #获取对应行列号的像素坐标
            #np.shape(array)
            c1 = srcwl2.xy(iheight, iwidth)
            ayx2[iheight][iwidth] = c1[0]
            ayy2[iheight][iwidth] = c1[1]
            
    from_crs1 = srcwl1.crs
    crs_wgs84 = CRS.from_epsg(4326)

    # transformer = Transformer.from_crs(from_crs1,crs_wgs84)

    # lat,lon = transformer.transform(ayx,ayy) 
    
    # plt.figure(figsize=(10, 10),dpi=600)
    # norm = matplotlib.colors.Normalize(vmin = -999, vmax = 999)
    # plt.figure(1)
    # ax1 = plt.subplot(121)
    # im = plt.imshow(artest1,cmap='rainbow',extent=(np.amin(lon),np.amax(lon),np.amin(lat),np.amax(lat)), norm = norm)
    # plt.colorbar(im, shrink=0.5)
    # plt.title(startdate,fontsize = 25)

    # ax1 = plt.subplot(122)
    # im = plt.imshow(artest2,cmap='rainbow',extent=(np.amin(lon),np.amax(lon),np.amin(lat),np.amax(lat)), norm = norm)
    # plt.colorbar(im, shrink=0.5)
    # plt.title(enddate,fontsize = 25)
    
    # figname = '/Volumes/lss2022/2022CoherencePaper/GauageStations/water-level-EDEN-Gauge/Figures/WaterLevelCompEDEN_'+startdate+'-'+enddate+'.png'
    
    # plt.savefig(figname)
    # plt.show()
    # plt.close()
    
    #calculate water level difference between two dates
    #srcwl1 is water level for startdate
    #srcwl2 is water level for enddate
    #lat, lon are the coordinates
    waldiff = np.empty(shape=(h1, w1))
    walmean = np.empty(shape =(h1,w1))

    for iwidth in range(0, w1, 1):
        for iheight in range(0, h1, 1):
            waldiff[iheight][iwidth] = artest2[iheight][iwidth] - artest1[iheight][iwidth]
            walmean[iheight][iwidth] = np.mean(
                [artest2[iheight][iwidth], artest1[iheight][iwidth]])

    for xx in range(0, numstations, 1):
        gaugelonlat = [select_gauge_lon[xx], select_gauge_lat[xx]]
        transformer1 = Transformer.from_crs(crs_wgs84, from_crs1)
        ayx1, ayy1 = transformer1.transform(select_gauge_lat[xx], select_gauge_lon[xx])
        gaugexy = [ayx1, ayy1]
        rowtw, coltw = srcwl2.index(*gaugexy)
        # waldiff.shape[0]获取行数，waldiff.shape[1]获取列数
        if rowtw <= waldiff.shape[0] and coltw <=waldiff.shape[1]:
            gaugewld[filenum1][xx] = waldiff[rowtw][coltw]
            gaugewlm[filenum1][xx] = walmean[rowtw][coltw]
        else:
            gaugewld[filenum1][xx] = 0.0
            gaugewlm[filenum1][xx] = 0.0



    
    # waterleveldifffile = '/Volumes/lss2022/2022CoherencePaper/GauageStations/water-level-EDEN-Gauge/waterleveldifffiles/12days/wldiff_'+startdate+'-'+enddate+'_09192022.tif'
    
    # waterlevelmeanfile = '/Volumes/lss2022/2022CoherencePaper/GauageStations/water-level-EDEN-Gauge/waterlevelmean/12days/wlmean_'+startdate+'-'+enddate+'_09222022.tif'
    
    # from rasterio.warp import calculate_default_transform as calcdt
    
    # filepre = wlfile1

    # with ro.open(filepre,'r') as src:
    #     ndata = src.read()
    #     profile = src.profile

    # #indexes标记band information. write需要一个shape的数组(band,row,col)，我们可以重塑数组
    # #也可以使用write(newarray, indexes = 1)
    # with ro.open(waterleveldifffile,'w',**profile) as dst:
    #     dst.write(waldiff,indexes = 1)
    
    # with ro.open(waterlevelmeanfile,'w',**profile) as dst:
    #     dst.write(walmean,indexes = 1)    


###### Start to plot comparison results between coherence and water level ###########

# # Region WCA1
# ggstations = ['SITE_7','NORTH_CA1','WCA1ME','SITE_8T','SITE_9','SOUTH_CA1']

# regions = ['WCA1']*6

# # Region WC2A

# ggstations = ['EDEN_11','WC2A159','WCA2F1','WCA2E1','WCA2F4','WCA2E4','WCA2U3','SITE_17','WCA2U1','SITE_19','WCA2RT']


# #regions = ['WC2A','WC2A','WC2A','WC2A','WC2A','WC2A','WC2A','WC2A','WC2A','WC2A','WC2A']

# regions = ['WC2A']*11

# #no station '2A300' in this file
# ggstations = ['2A300']
# regions = ['WC2A']

# Region WCA2B

ggstations = ['EDEN_13','SITE_99']

regions = ['WCA2B']*2

# # Region  WCA3A

# ggstations = ['3A10','3ANW','3A11','S339_T','EDEN_9','3ANE','SITE_63','3A12','SITE_62','EDEN_5',\
#               '3A9','S340_H','EDEN_4','3AS','EDEN_14','3A-5','EDEN_12','W15','W18','SITE_64',\
#               'W11','W14','3AS3W1','SITE_65','W2','W5']

# regions = ['WCA3A']*26

# # Region WCA3B

# ggstations = ['SITE_76','EDEN_7','SITE_71','TI-9','TI-8','SRS1','EDEN_10','3B-SE','3BS1W1']
    

# regions = ['WCA3B']*9

# Region BCNP
# ggstations = ['BCA2','BCA17','BCA3','BCA12','BCA18','L28_GAP','BCA13','BCA15','EDEN_6','BCA16','BCA14','BCA4',\
#               'BCA5','BCA8','Tamiami_Canal_Monroe_to_Carnestown','EDEN_1','Tamiami_Canal_40-Mile_Bend_to_Monroe',\
#                   'BCA11','BCA9','LOOP2_H','BCA10','BCA20']

# regions = ['BCNP']*22

# # Region ENP
# ggstations = ['SPARO','NP201','MET-1','NESRS3','G-3577','NESRS1','NP202','G-620','NP203','P33','NESRS4',\
#               'NESRS5','G-1502','RG1','RG2','G-3437','RG2','NP206','P36','EDEN_3','P35','MO-215','A13',\
#                   'CR3','CR2','NP44','NP62','R3110','NP72','NTS14','TSB','TS2','SP','DO2','P38','SR1',\
#                       'DO2','DO1','R127','CY3','CY2','NP67','TSH','P37','NP46','NP67','TSH','E146','EVER5A']

# regions = ['ENP']*49

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
#from ices2analysis import linefit, point2line
from math import radians, cos, sin, asin, sqrt
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

### plots for scatter plots of coherence vs water level difference, the best fit line
for stt in range(0, len(ggstations), 1):

    # gaugecoh[numcohwl,numstations] 75*313
    gstaname = []

    gaugenames = df_gauge['EDEN Station Name']

    choselect = np.empty(shape=(numcohwl, 1))

    for xx in range(0, numstations, 1):

        if gaugenames[xx] == ggstations[stt]:  # region WCA1
            for yy in range(0, numcohwl, 1):
                choselect[yy] = gaugecoh[yy][xx]


    choselectwld = np.empty(shape=(numcohwl, 1))

    choselectwlm = np.empty(shape=(numcohwl, 1))

    for xx in range(0, numstations, 1):

        if gaugenames[xx] == ggstations[stt]:  # region WCA1
            for yy in range(0, numcohwl, 1):
                choselectwld[yy] = gaugewld[yy][xx]
                choselectwlm[yy] = gaugewlm[yy][xx]


    x = list(range(75))
    xsmall = [0, 25, 49,  74]
    xlabelsmall = [result_list[0], result_list[25],
                result_list[49], result_list[74]]


    # from scipy import stats
    
    
   
    # #calculate correlation coefficient
    # def get_p_value(arrayA,arrayB):
    #     r = stats.pearsonr(arrayA,arrayB)
    
    coher=np.array(choselect).reshape(-1,1)
    wld=np.array(choselectwld).reshape(-1,1)
    wlm=np.array(choselectwlm).reshape(-1,1)
    
    #fitting between coherence and water level difference
    
    # Create linear regression object
    regr1 = linear_model.LinearRegression()  
    
    regr1.fit(wld, coher)
    
    # Make predictions using the testing set
    y_pred = regr1.predict(wld)
    
    # The coefficients
    print('y = {0:.2f}*x + {1:.2f}'.format(regr1.coef_[0][0],regr1.intercept_[0]))
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(coher, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(coher, y_pred))
    
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(211)
    ax.plot(wld, coher,'b^',markersize=4)
    ax.plot(wld, y_pred,'k--',alpha=0.5)
    # axes[t].text(-.4, 2.9, 'y = {0:.2f}*x + ({1:.2f})'.format(regr.coef_[0][0],regr.intercept_[0]+crstest))
    # axes[t].text(-.4, 2.6, 'R^2 = {0:.2f}'.format(r2_score(alt, y_pred)))
    # for tempt in range(0,len(gauge)):
        
    #     axes[t].text(gauge[tempt], altempt[tempt]+0.025, gauge_stations[tempt], c='k', fontsize=6)
    # #axes[t].text(gauge, alt+crstest+0.025, gauge_stations, c='k', fontsize=6)
    
    # #plt.subplots_adjust(hspace=0.02,wspace=0.05,bottom=0.18,top=0.9,left=0.10,right=0.95)

    # axes[t].set_ylim([-.5,4.2])
    # axes[t].set_xlim([-.5,4.2])
    # axes[t].grid(alpha=0.15)
    # axes[t].set_title(date_list[index2])
    ax.set_xlabel('WaterLevelDifference (feet)')
    ax.set_ylabel('Coherence')
    ax.set_title(ggstations[stt])

    # Create linear regression object
    regr2 = linear_model.LinearRegression()

    regr2.fit(wlm, coher)

    # Make predictions using the testing set
    y_pred2 = regr2.predict(wlm)

    # The coefficients
    print('y = {0:.2f}*x + {1:.2f}'.format(regr2.coef_[0][0], regr2.intercept_[0]))
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(coher, y_pred2))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(coher, y_pred2))


    ax2 = fig.add_subplot(212)
    ax2.plot(wlm, coher, 'b^', markersize=4)
    ax2.plot(wlm, y_pred2, 'k--', alpha=0.5)
# axes[t].text(-.4, 2.9, 'y = {0:.2f}*x + ({1:.2f})'.format(regr.coef_[0][0],regr.intercept_[0]+crstest))
# axes[t].text(-.4, 2.6, 'R^2 = {0:.2f}'.format(r2_score(alt, y_pred)))
# for tempt in range(0,len(gauge)):

#     axes[t].text(gauge[tempt], altempt[tempt]+0.025, gauge_stations[tempt], c='k', fontsize=6)
# #axes[t].text(gauge, alt+crstest+0.025, gauge_stations, c='k', fontsize=6)

# #plt.subplots_adjust(hspace=0.02,wspace=0.05,bottom=0.18,top=0.9,left=0.10,right=0.95)

# axes[t].set_ylim([-.5,4.2])
# axes[t].set_xlim([-.5,4.2])
# axes[t].grid(alpha=0.15)
# axes[t].set_title(date_list[index2])
    ax2.set_xlabel('WaterLevelMean (feet)')
    ax2.set_ylabel('Coherence')
    #ax2.set_title(ggstations[stt])



    
    
    
    

#     # plot scatter plot and fitting line for one selected station
#     fig = plt.figure(figsize=(10, 10), dpi=300)
#     ax = fig.add_subplot(211)
#     lin1 = plt.plot(x, choselect, color='g', marker='o',
#                 markersize=10, label='Coherence')
#     plt.title('Timeseries of Coherence for station '+ggstations[stt]+' for region '+ regions[stt])
#     plt.xlabel('Dates Period', fontsize=20)  # remove x axis label
#     plt.ylabel('Coherence')

#     #plt.xticks(xsmall,labels = xlabelsmall,fontsize=15,rotation=0,ha='center',va='center')
#     ax.xaxis.set_major_formatter(ticker.NullFormatter())

#     ax1 = ax.twinx()
#     lin2 = ax1.plot(x, choselectwld, color='b', marker='o',
#                 markersize=10, label='WaterLevel Difference')
#     #lin3 = ax1.plot(x,choselectwlm,color='m',marker='o',markersize=10,label='WaterLevel Mean')

#     lins = lin1+lin2
#     labs = [l.get_label() for l in lins]
#     ax.legend(lins, labs, loc="upper left", fontsize=15)
#     ax.set_ylim(0,1) # coherence range
#     #ax1.set_ylim(-20,50) #water level difference in region WCA3A
#     #ax1.set_ylim(-60,60) #water level difference in region WC2A
#     # ax1.set_ylim(-15,30) #water level difference in region WCA1
#     #ax1.set_ylim(-15,35) #water level difference in region WCA2B
#     #ax1.set_ylim(-15,35) #water level difference in region WCA3B
#     #ax1.set_ylim(-50,100) #water level difference in region BCNP
#     ax1.set_ylim(-40,90) #water level difference in region ENP

#     axt = fig.add_subplot(212)
#     lin1 = plt.plot(x, choselect, color='g', marker='o',
#                 markersize=10, label='Coherence')
#     #plt.title('Timeseries of Coherence for station SITE_7 for region WCA1')
#     #remove x axis label
#     plt.ylabel('Coherence')

#     plt.xticks(xsmall, labels=xlabelsmall, fontsize=15, rotation=-30)


#     ax2 = axt.twinx()
#     #lin2 = ax1.plot(x,choselectwld,color='',marker='o',markersize=10,label='WaterLevel Difference')
#     lin2 = ax2.plot(x, choselectwlm, color='m', marker='o',
#                 markersize=10, label='WaterLevel Mean')

#     lins = lin1+lin2
#     labs = [l.get_label() for l in lins]
#     axt.legend(lins, labs, loc="upper left", fontsize=15)
#     axt.set_ylim(0,1)
#     #ax2.set_ylim(200,360) #water level mean in region WCA3A
#     #ax2.set_ylim(270,420) #water level mean in region WC2A
#     #ax2.set_ylim(430,500) #water level mean in region WCA1   
#     #ax2.set_ylim(200,320) #water level mean in region WCA2B 
#     #ax2.set_ylim(200,320) #water level mean in region WCA3B
#     #ax2.set_ylim(-60,460) #water level mean in region BCNP
#     ax2.set_ylim(-30,260) #water level mean in region ENP
    
#     figname = '/Volumes/lss2022/2022CoherencePaper/ResearchSummary/TimeSeriesCoherence_WaterLevel09282022/'+regions[stt]+'/' + ggstations[stt] +'_'+regions[stt]+'_09302022.png'


#     plt.savefig(figname)
#     plt.show()
#     plt.close()

# #end of ### plots for showing coherence vs water level difference, coherence vs water level mean


# ### plots for showing coherence vs water level difference, coherence vs water level mean
# for stt in range(0, len(ggstations), 1):

#     # gaugecoh[numcohwl,numstations] 75*313
#     gstaname = []

#     gaugenames = df_gauge['EDEN Station Name']

#     choselect = np.empty(shape=(numcohwl, 1))

#     for xx in range(0, numstations, 1):

#         if gaugenames[xx] == ggstations[stt]:  # region WCA1
#             for yy in range(0, numcohwl, 1):
#                 choselect[yy] = gaugecoh[yy][xx]


#     choselectwld = np.empty(shape=(numcohwl, 1))

#     choselectwlm = np.empty(shape=(numcohwl, 1))

#     for xx in range(0, numstations, 1):

#         if gaugenames[xx] == ggstations[stt]:  # region WCA1
#             for yy in range(0, numcohwl, 1):
#                 choselectwld[yy] = gaugewld[yy][xx]
#                 choselectwlm[yy] = gaugewlm[yy][xx]


#     x = list(range(75))
#     xsmall = [0, 25, 49,  74]
#     xlabelsmall = [result_list[0], result_list[25],
#                 result_list[49], result_list[74]]


#     # plot time series of coherence for one selected station
#     fig = plt.figure(figsize=(10, 10), dpi=300)
#     ax = fig.add_subplot(211)
#     lin1 = plt.plot(x, choselect, color='g', marker='o',
#                 markersize=10, label='Coherence')
#     plt.title('Timeseries of Coherence for station '+ggstations[stt]+' for region '+ regions[stt])
#     plt.xlabel('Dates Period', fontsize=20)  # remove x axis label
#     plt.ylabel('Coherence')

#     #plt.xticks(xsmall,labels = xlabelsmall,fontsize=15,rotation=0,ha='center',va='center')
#     ax.xaxis.set_major_formatter(ticker.NullFormatter())

#     ax1 = ax.twinx()
#     lin2 = ax1.plot(x, choselectwld, color='b', marker='o',
#                 markersize=10, label='WaterLevel Difference')
#     #lin3 = ax1.plot(x,choselectwlm,color='m',marker='o',markersize=10,label='WaterLevel Mean')

#     lins = lin1+lin2
#     labs = [l.get_label() for l in lins]
#     ax.legend(lins, labs, loc="upper left", fontsize=15)
#     ax.set_ylim(0,1) # coherence range
#     #ax1.set_ylim(-20,50) #water level difference in region WCA3A
#     #ax1.set_ylim(-60,60) #water level difference in region WC2A
#     # ax1.set_ylim(-15,30) #water level difference in region WCA1
#     #ax1.set_ylim(-15,35) #water level difference in region WCA2B
#     #ax1.set_ylim(-15,35) #water level difference in region WCA3B
#     #ax1.set_ylim(-50,100) #water level difference in region BCNP
#     ax1.set_ylim(-40,90) #water level difference in region ENP

#     axt = fig.add_subplot(212)
#     lin1 = plt.plot(x, choselect, color='g', marker='o',
#                 markersize=10, label='Coherence')
#     #plt.title('Timeseries of Coherence for station SITE_7 for region WCA1')
#     #remove x axis label
#     plt.ylabel('Coherence')

#     plt.xticks(xsmall, labels=xlabelsmall, fontsize=15, rotation=-30)


#     ax2 = axt.twinx()
#     #lin2 = ax1.plot(x,choselectwld,color='',marker='o',markersize=10,label='WaterLevel Difference')
#     lin2 = ax2.plot(x, choselectwlm, color='m', marker='o',
#                 markersize=10, label='WaterLevel Mean')

#     lins = lin1+lin2
#     labs = [l.get_label() for l in lins]
#     axt.legend(lins, labs, loc="upper left", fontsize=15)
#     axt.set_ylim(0,1)
#     #ax2.set_ylim(200,360) #water level mean in region WCA3A
#     #ax2.set_ylim(270,420) #water level mean in region WC2A
#     #ax2.set_ylim(430,500) #water level mean in region WCA1   
#     #ax2.set_ylim(200,320) #water level mean in region WCA2B 
#     #ax2.set_ylim(200,320) #water level mean in region WCA3B
#     #ax2.set_ylim(-60,460) #water level mean in region BCNP
#     ax2.set_ylim(-30,260) #water level mean in region ENP
    
#     figname = '/Volumes/lss2022/2022CoherencePaper/ResearchSummary/TimeSeriesCoherence_WaterLevel09282022/'+regions[stt]+'/' + ggstations[stt] +'_'+regions[stt]+'_09302022.png'


#     plt.savefig(figname)
#     plt.show()
#     plt.close()

#end of ### plots for showing coherence vs water level difference, coherence vs water level mean

# # ####test run### water level difference between two dates
# # tif_file_path = '/Volumes/lss2022/2022CoherencePaper/GauageStations/water-level-EDEN-Gauge/surface_zipfiles/2015_q1_tiff_v3/'

# # datetext = '20150101_20150128' # this can be obtained from coherence file name

# # wlstartdate = datetext[0:8]
# # wlenddate = datetext[9:17]

# # wlfile1 = tif_file_path + 's_'+wlstartdate+'.tif'
# # wlfile2 = tif_file_path + 's_' +wlenddate + '.tif'

# # srcwl1 = ro.open(wlfile1)
# # artest1 = srcwl1.read(1)
# # w1 = srcwl1.width
# # h1 = srcwl1.height

# # srcwl2 = ro.open(wlfile2)
# # artest2 = srcwl2.read(1)
# # w2 = srcwl2.width
# # h2 = srcwl2.height

# # # store the x and y

# # ayx = np.empty(shape=(h1,w1))
# # ayy = np.empty(shape=(h1,w1))

# # ayx2 = np.empty(shape=(h2,w2))
# # ayy2 = np.empty(shape=(h2,w2))


# # for iwidth in range(0,w1,1):
# #     for iheight in range(0,h1,1):
# #         #获取对应行列号的像素坐标
# #         #np.shape(array)
# #         c1 = srcwl1.xy(iheight,iwidth)
# #         ayx[iheight][iwidth] = c1[0]
# #         ayy[iheight][iwidth] = c1[1]

# # for iwidth in range(0,w2,1):
# #     for iheight in range(0,h2,1):
# #         #获取对应行列号的像素坐标
# #         #np.shape(array)
# #         c1 = srcwl2.xy(iheight,iwidth)
# #         ayx2[iheight][iwidth] = c1[0]
# #         ayy2[iheight][iwidth] = c1[1]

# # from_crs1 = srcwl1.crs
# # crs_wgs84 = CRS.from_epsg(4326)

# # transformer = Transformer.from_crs(from_crs1,crs_wgs84)

# # lat,lon = transformer.transform(ayx,ayy) 

# # # example to plot one water level
# # plt.figure(figsize=(10, 10),dpi=600)
# # norm = matplotlib.colors.Normalize(vmin = -999, vmax = 999)
# # plt.figure(1)
# # ax1 = plt.subplot(121)
# # im = plt.imshow(artest1,cmap='rainbow',extent=(np.amin(lon),np.amax(lon),np.amin(lat),np.amax(lat)), norm = norm)
# # plt.colorbar(im, shrink=0.5)
# # plt.title('20150101',fontsize = 25)

# # ax1 = plt.subplot(122)
# # im = plt.imshow(artest2,cmap='rainbow',extent=(np.amin(lon),np.amax(lon),np.amin(lat),np.amax(lat)), norm = norm)
# # plt.colorbar(im, shrink=0.5)
# # plt.title('20150128',fontsize = 25)












