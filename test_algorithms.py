import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import math
import Q_learning as Ql

class test_model: 

    def __init__ (self, model, train_split=0.8, validation_split=0.10, test_split=0.10,images_directory=r'D:\1 - School\Econometrics\2020 - 2021\Scriptie\Explainable AI\Scripts\Data\binary_colour_scheme.32frames', data_directory=r'D:\1 - School\Econometrics\2020 - 2021\Scriptie\Explainable AI\Scripts\Data\stock_data_clean.csv'): 
        self.model          = model
        self.train_split    = train_split
        self.test_split     = test_split
        self.val_split      = validation_split
        self.img_dir        = images_directory
        self.data_dir       = data_directory
        self.timeframe      = 32 

    def K_top_bottom(self, k=0.05,CNN=1, LSTM=1, img='binary', MIDQN=0, diff=0 ): 
        #note: still needs to be avaraged over the No. companies. Otherwise way to high. 
        
        #format the data
        data            = pd.read_csv(self.data_dir,header=[0, 1] , index_col = 0)
        data.index      = pd.to_datetime(data.index)
        data            = np.log(data) #  - np.log(data.shift(1))
        symbols         = data.columns.unique(level=0)
        timeframes      = data.index.shape[0] - self.timeframe+1
        train_tfs       = np.round(timeframes*self.train_split)           
        val_tfs         = [train_tfs + 1, train_tfs + np.round(timeframes*self.val_split)]

        #Amount of trades in top/bottom
        K               = np.round(len(symbols)*k).astype(int)
        
        #data storages prep
        longs              = np.zeros(len(symbols))
        shorts             = np.zeros(len(symbols))
        Q_val              = np.zeros((len(symbols),2))
        Q_diff_val         = np.zeros(len(symbols))
        R                  = np.zeros(len(symbols))
        profit             = np.zeros((val_tfs[1]-val_tfs[0]+1).astype(int))

        T=0
        for t in range(val_tfs[0].astype(int),val_tfs[1].astype(int)-1):
            T+=1
            C=0
            for c in symbols: 
                #get the predicted values and corresponding A,R
                if CNN ==1 and LSTM ==1: 
                    if img =='binary': 
                        S_img_dir   = self.img_dir +'\\'+ str(c) + '_' + str(t+1) + '.txt'
                        S_img = pd.read_csv(S_img_dir, sep=" ", header=None)
                    else: 
                        #get data  A  
                        S_img_dir   = self.img_dir +'\\'+ str(c) + '_' + str(t+1) + '.png' #t.astype(int)
                        S_img       = tf.keras.preprocessing.image.load_img(S_img_dir)
                        S_img       = keras.preprocessing.image.img_to_array(S_img)/255 #normalize the data

                    S_data      = data[c][t:t+20].Close -data[c][t-1:t+19].Close
                    if MIDQN == 1: 
                        Q_val[C]    = self.model([tf.expand_dims(S_img,0), tf.expand_dims(S_data,0)], training=False)[1].numpy()[0]
                    else: 
                        Q_val[C]    = self.model([tf.expand_dims(S_img,0), tf.expand_dims(S_data,0)], training=False).numpy()[0]
                elif CNN==0 and LSTM==1:
                    #get data  A  
                    S_data      = data[c][t:t+20].Close

                    Q_val[C]     = self.model([tf.expand_dims(S_data,0)], training=False).numpy()[0]
                elif CNN ==1 and LSTM ==0:
                    if img =='binary': 
                        S_img_dir   = self.img_dir +'\\'+ str(c) + '_' + str(t+1) + '.txt'
                        S_img = pd.read_csv(S_img_dir, sep=" ", header=None)
                    else:  
                        #get data  A  
                        S_img_dir   = self.img_dir +'\\'+ str(c) + '_' + str(t+1) + '.png' #t.astype(int)
                        S_img       = tf.keras.preprocessing.image.load_img(S_img_dir)
                        S_img       = keras.preprocessing.image.img_to_array(S_img)/255 #normalize the data

                    Q_val[C]     = self.model([tf.expand_dims(S_img,0)], training=False).numpy()[0]

                Q_diff_val[C]      = Q_val[C][0]-Q_val[C][1]

                R[C]           = data[c][t+20:t+21].Close-data[c][t+20:t+21].Open
 
                C +=1

            #masks = tf.one_hot(A_val,3,on_value=None, off_value=None,)
            #Q_vals = tf.multiply(Q_val, masks) 

            #rank the Q-values
            if diff == 1 : 
                Top_K_indices = (Q_diff_val).argsort()[-K:]     # indext Q_val correctly
                Bottom_K_indices = (Q_diff_val).argsort()[:K]     # indext Q_val correctly
            else: 
                Top_K_indices = (Q_val[:,0]).argsort()[-K:]     # indext Q_val correctly
                Bottom_K_indices = (Q_val[:,1]).argsort()[:K]     # indext Q_val correctly

            longs[Top_K_indices]        +=1
            shorts[Bottom_K_indices]    +=1

            profit[T]   = tf.reduce_sum(R[Top_K_indices])- tf.reduce_sum(R[Bottom_K_indices])
            if T%10==0:
                string= "{} out of {} loops done. Current profit:{}"
                total_loops=(val_tfs[1]-val_tfs[0]).astype(int)+1
                tot_prof = np.sum(profit)
                print(string.format(T,total_loops,tot_prof))

        return profit, longs, shorts

    def buy_hold(self,):
        #format the data
        data            = pd.read_csv(self.data_dir,header=[0, 1] , index_col = 0)
        data.index      = pd.to_datetime(data.index)
        data            = np.log(data) - np.log(data.shift(1))
        symbols         = data.columns.unique(level=0)
        timeframes      = data.index.shape[0] - self.timeframe+1
        train_tfs       = np.round(timeframes*self.train_split)           
        val_tfs         = [train_tfs + 1, train_tfs + np.round(timeframes*self.val_split)]
        
        profit             = np.zeros((val_tfs[1]-val_tfs[0]+1).astype(int))
        T=0
        for t in range(val_tfs[0].astype(int),val_tfs[1].astype(int)-1):
            T+=1
            temp_prof=0
            for c in symbols: 
                temp_prof += data[c][t+20:t+21].Close
            profit[T] += temp_prof

        return profit

    def max_K(self,k=0.05):

        #format the data
        data            = pd.read_csv(self.data_dir,header=[0, 1] , index_col = 0)
        data.index      = pd.to_datetime(data.index)
        data            = np.log(data)
        symbols         = data.columns.unique(level=0)
        timeframes      = data.index.shape[0] - self.timeframe+1
        train_tfs       = np.round(timeframes*self.train_split)           
        val_tfs         = [train_tfs + 1, train_tfs + np.round(timeframes*self.val_split)]
    
        profit             = np.zeros((val_tfs[1]-val_tfs[0]+1).astype(int))
        #Amount of trades in top/bottom
        K               = np.round(len(symbols)*k).astype(int)
        #data storages prep
        longs              = np.zeros(len(symbols))
        shorts             = np.zeros(len(symbols))
        R                  = np.zeros(len(symbols))
        profit             = np.zeros((val_tfs[1]-val_tfs[0]+1).astype(int))
        profit_longs       = np.zeros((val_tfs[1]-val_tfs[0]+1).astype(int))
        profit_shorts      = np.zeros((val_tfs[1]-val_tfs[0]+1).astype(int))
        T=0
        
        for t in range(val_tfs[0].astype(int),val_tfs[1].astype(int)-1):
            
            C=0
            
            for c in symbols: 
                R[C] = data[c][t:t+1].Close-data[c][t:t+1].Open
                C   += 1

            Top_K_indices = R[:].argsort()[-K:]  
            Bottom_K_indices = R[:].argsort()[:K] 

            longs[Top_K_indices]        +=1
            shorts[Bottom_K_indices]    +=1
            
            profit_longs[T]     = tf.reduce_sum(R[Top_K_indices])
            profit_shorts[T]    = tf.reduce_sum(R[Bottom_K_indices])
            profit[T]           = tf.reduce_sum(R[Top_K_indices])- tf.reduce_sum(R[Bottom_K_indices])
            
            if t%10==0:
                string= "{} out of {} loops done. Current profit:{}"
                total_loops=(val_tfs[1]-val_tfs[0]).astype(int)+1
                tot_prof = np.sum(profit)
                print(string.format(t,total_loops,tot_prof))

            T+=1

        return profit, profit_longs, profit_shorts   

    def calc_Q(self, CNN=1, LSTM=1,MIDQN=0, img='binary') :
               
        #format the data
        data            = pd.read_csv(self.data_dir,header=[0, 1] , index_col = 0)
        data.index      = pd.to_datetime(data.index)
        data            = np.log(data) #  - np.log(data.shift(1))
        symbols         = data.columns.unique(level=0)
        timeframes      = data.index.shape[0] - self.timeframe+1

        Q_values        = np.zeros((timeframes,len(symbols),2))
        
        T=0
        for t in range(5216):
            T+=1
            C=0
            for c in symbols: 
                #get the predicted values and corresponding A,R
                if CNN ==1 and LSTM ==1: 
                    if img =='binary': 
                        S_img_dir   = self.img_dir +'\\'+ str(c) + '_' + str(t+1) + '.txt'
                        S_img = pd.read_csv(S_img_dir, sep=" ", header=None)
                    else: 
                        #get data  A  
                        S_img_dir   = self.img_dir +'\\'+ str(c) + '_' + str(t+1) + '.png' #t.astype(int)
                        S_img       = tf.keras.preprocessing.image.load_img(S_img_dir)
                        S_img       = keras.preprocessing.image.img_to_array(S_img)/255 #normalize the data

                    S_data      = data[c][t:t+31].Close -data[c][t-1:t+31].Close
                    if MIDQN == 1: 
                        Q_values[t,C,:]    = 5/7 * self.model([tf.expand_dims(S_img,0), tf.expand_dims(S_data,0)], training=False)[1].numpy()[0] +1/7*(self.model([tf.expand_dims(S_img,0), tf.expand_dims(S_data,0)], training=False)[0].numpy()[0]+ self.model([tf.expand_dims(S_img,0), tf.expand_dims(S_data,0)], training=False)[2].numpy()[0])
                    else: 
                        Q_values[t,C,:]  = self.model([tf.expand_dims(S_img,0), tf.expand_dims(S_data,0)], training=False).numpy()[0]
                
                elif CNN==0 and LSTM==1:
                    #get data  A  
                    S_data      = data[c][t:t+32].Close

                    Q_values[t,C,:]      = self.model([tf.expand_dims(S_data,0)], training=False).numpy()[0]
                elif CNN ==1 and LSTM ==0:
                    if img =='binary': 
                        S_img_dir   = self.img_dir +'\\'+ str(c) + '_' + str(t+1) + '.txt'
                        S_img = pd.read_csv(S_img_dir, sep=" ", header=None)
                    else:  
                        #get data  A  
                        S_img_dir   = self.img_dir +'\\'+ str(c) + '_' + str(t+1) + '.png' #t.astype(int)
                        S_img       = tf.keras.preprocessing.image.load_img(S_img_dir)
                        S_img       = keras.preprocessing.image.img_to_array(S_img)/255 #normalize the data

                    Q_values[t,C,:]      = self.model([tf.expand_dims(S_img,0)], training=False).numpy()[0]
                C +=1 
        return Q_values