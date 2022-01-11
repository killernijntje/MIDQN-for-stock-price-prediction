# %%
# import required packages 
import matplotlib.dates as mpdates 
import matplotlib.pyplot as plt 
import mplfinance as mpf
import matplotlib as mpl
from PIL import Image
import pandas as pd 
import math as math
import numpy as np
import io   as io
import gc   as gc
import os as os



#set run instance number, and the total number of concurrent instances
run=1
tot_runs = 1; 

#timeframe
tf = 32

#set_pixels
img_size = 32

#set the colour scheme
pure =1
if pure == 1:
    # pure colour scheme
    col_up = '#00FF00'
    col_down = '#FF0000'
    col_vol = "#0000FF"
    
    #set directory
    direct = "C:/Users/robin/1 - Scriptie/images/pure_colour_scheme"
else: 
    # mixed colour scheme 
    col_up = '#55FF00'
    col_down = '#FF5500'
    col_vol = "#0000FF"

    #set directory
    direct = "C:/Users/robin/1 - Scriptie/images/mixed_colour_scheme"




#loading the data
data = pd.read_csv(r'D:\1 - School\Econometrics\2020 - 2021\Scriptie\Explainable AI\Scripts\Data\stock_data_clean.csv',header=[0, 1] , index_col = 0 )
data.index=pd.to_datetime(data.index)

#taking the logartimic difference
data = np.log(data) - np.log(data.shift(1))

#subsetting the data
total_symbols = math.floor(len(data.columns.unique(level=0))/tot_runs)
symbols = data.columns.unique(level=0)[(run-1)*total_symbols:]

#set the plot parameters
mc = mpf.make_marketcolors(up = col_up ,down = col_down, edge='inherit', volume= col_vol, wick='inherit')
s  = mpf.make_mpf_style(marketcolors=mc)   


# %%
# creating candlestick chart with volume
def plot_candle(i,j,data,symbols,s,mc,direct,img_size, tf):
     
    #slicing data into 30 trading day windows
    data_temp=data[symbols[j]][i-tf:i]  

    #creating and saving the candlestick charts
    buf = io.BytesIO()
    save = dict(fname= buf, rc = (["boxplot.whiskerprops.linewidth",10]), 
                    pad_inches=0,bbox_inches='tight')
    mpf.plot(data_temp,savefig=save,type='candle',style=s, volume=True, axisoff=True,figratio=(1,1),closefig=True,
             ylim=[-0.17,0.17],addplot =[mpf.make_addplot(data_temp.Volume,type='bar',color=col_vol,ylim=[-1.2,1.2],panel=1)])
    buf.seek(0)
    im = Image.open(buf).resize((img_size,img_size))
    im.save(direct+"/"+str(symbols[j])+"_"+str(i-tf+1)+".png", "PNG")
    buf.close()
    plt.close("all")


# %%
#use different backend for mpl
mpl.use('agg')

#check if images folder excists, if not, create it. 
if not os.path.exists(direct):
    os.mkdir(direct)

for j in range(0,len(symbols)):

    for i in range(tf,len(data)):

        #check if the file has already been created
        if not os.path.exists(direct+"/"+str(symbols[j])+"_"   +str(i-tf+1)+".png"):
            #call the functions and create the 
            plot_candle(i , j , data , symbols ,s ,mc ,direct , img_size, tf)
            gc.collect()


