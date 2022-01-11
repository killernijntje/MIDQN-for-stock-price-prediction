import numpy as np
from numpy import random
    
class Experience_replay:
    def __init__(self, M,LSTM=1,CNN=1):
        #initialtize buffer size
        self.M = M
        self.LSTM = LSTM
        self.CNN = CNN


        #Experience replay buffers
        if self.LSTM == 1 : 
            self.S_data_history         = []
            self.S_data_next_history    = []

        if self.CNN == 1 : 
            self.S_img_history          = []
            self.S_img_next_history     = []
        
        self.A_history              = []
        self.R_history              = []
        

    def remmember(self,S_img,S_data,A,R,S_img_next,S_data_next):
        
        if self.LSTM == 1 : 
            self.S_data_history.append(S_data)
            self.S_data_next_history.append(S_data_next)
        
        if self.CNN == 1 :
            self.S_img_history.append(S_img)
            self.S_img_next_history.append(S_img_next)

        self.A_history.append(A)
        self.R_history.append(R)

        # If the memory buffer is full delete oldest memory
        if (len(self.A_history) >= self.M): 
            
            if self.LSTM == 1 : 
                del self.S_data_history[0]
                del self.S_data_next_history[0]
            
            if self.CNN == 1 : 
                del self.S_img_history[0]
                del self.S_img_next_history[0]

            del self.A_history[0]
            del self.R_history[0]

        return 

    def get_batch(self, batch_size):
        
        #get random indices for replay buffer
        idx = np.random.randint(0,len(self.A_history)-1, batch_size)

        #Create sample data for batch
        if self.LSTM == 1: 
            S_data_sample       = [self.S_data_history[i] for i in idx]
            S_data_next_sample  = [self.S_data_next_history[i] for i in idx]

        if self.CNN == 1 :     
            S_img_sample        = [self.S_img_history[i] for i in idx]
            S_img_next_sample   = [self.S_img_next_history[i] for i in idx]
            
        A_sample            = [self.A_history[i] for i in idx]
        R_sample            = [self.R_history[i] for i in idx]
        
        

        if self.LSTM == 1 and self.CNN ==1 : 
            return S_img_sample, S_data_sample, A_sample, R_sample,S_img_next_sample,S_data_next_sample 
        
        elif  self.LSTM == 1 and self.CNN ==0:
            return S_data_sample, A_sample, R_sample,S_data_next_sample 
        
        elif self.LSTM == 0 and self.CNN == 1: 
            return S_img_sample, A_sample,S_img_next_sample, R_sample