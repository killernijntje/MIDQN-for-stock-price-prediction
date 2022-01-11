from keras.layers import Conv2D, BatchNormalization, Activation,MaxPool2D, Add, AveragePooling2D, Input, ZeroPadding2D, concatenate, Dense, Flatten,LSTM
from keras.models import Model
import tensorflow as tf
from ConstantPadding2D import ConstantPadding2D
import keras
import numpy as np
import random
import os 

class Q_model: 

    def __init__(self,width, heigth,seed, depth=3, N_actions=3, activation_func = 'relu'):
        self.seed = seed
        self.W = width
        self.H = heigth
        self.D = depth
        self.N_actions = N_actions
        self.initializer = tf.keras.initializers.HeUniform()
        self.activation_func = activation_func
    def Q_model_setup(self, include_CNN=1, include_LSTM=1, MIDQN=0, time_steps=20): 

        #set seeds 
        os.environ['PYTHONHASHSEED']=str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.compat.v1.set_random_seed(self.seed)

        #create the required branches
        if include_CNN == 1:
            CNN     = self.CNN_branch()


        if include_LSTM == 1:
            LSTM    = self.LSTM_branch(time_steps=time_steps)


        #create the required heads and compile the model
        if MIDQN == 0: 

            if include_CNN == 1 and include_LSTM == 1:
                
                #CNN-LSTM-DQN
                head       = self.Head_branch(CNN,model2=LSTM)
                model      = Model([CNN.input,LSTM.input], head)

            elif include_CNN == 1 and include_LSTM == 0:
                #CNN-DNQ
                head       = self.Head_branch(CNN)
                model      = Model(CNN.input, head)

            elif include_CNN == 0 and include_LSTM == 1:
                #LSTM-DQN
                head       = self.Head_branch(LSTM)
                model      = Model(LSTM.input, head)

        elif MIDQN == 1:

            #MIDQN
            LSTM_head      = self.Head_branch(LSTM)
            CNN_head       = self.Head_branch(CNN)
            MI_head        = self.Head_branch(CNN,model2=LSTM)
            model      = Model(inputs=[CNN.input,LSTM.input], outputs=[CNN_head, LSTM_head, MI_head] )

        return model

    def CNN_branch(self, short=1):

        initializer = tf.keras.initializers.HeUniform()

        input_shape = (self.W , self.W, self.D)
        inputs =Input(shape=input_shape)
        
        x = ConstantPadding2D((3, 3))(inputs)

        x = Conv2D(filters=64,kernel_size=(7,7), strides=2, kernel_initializer=initializer)(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation(self.activation_func)(x)
        x = MaxPool2D(pool_size=(3,3) , strides=(2,2))(-1*x)

        # Convolutional block 
        x = self.resnet_conv_block(-1*x, [64, 64, 256], stride=1)
        # 2x identity block 
        x = self.resnet_identity_block(x,[64, 64, 256])
        x = self.resnet_identity_block(x,[64, 64, 256])

        if short==0:
            # Convolutional block 
            x = self.resnet_conv_block(x, [128, 128, 512])
            # 3x identity block 
            x = self.resnet_identity_block(x,[128, 128, 512])
            x = self.resnet_identity_block(x,[128, 128, 512])
            x = self.resnet_identity_block(x,[128, 128, 512])

        x = AveragePooling2D((2,2))(x)

        model = Model(inputs=inputs, outputs = x)

        return model
    
    def resnet_conv_block(self, x, filters,f=3, stride=2): 
        
        F1, F2, F3 = filters
        x_short = x

        #main conv block branch
        x = Conv2D(filters=F1, kernel_size=(1,1),strides = (stride,stride), kernel_initializer=self.initializer)(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation(self.activation_func)(x)

        x = Conv2D(filters=F2, kernel_size=(f,f), strides = (1,1),padding = 'same',kernel_initializer=self.initializer)(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation(self.activation_func)(x)

        x = Conv2D(filters=F3, kernel_size=(1,1),strides = (1,1), padding = 'valid',kernel_initializer=self.initializer)(x)
        x = BatchNormalization(axis = 3)(x)

        #shortcut branch

        x_short = Conv2D(filters=F3, kernel_size=(1,1),strides = (stride,stride), padding = 'valid',kernel_initializer=self.initializer)(x_short)
        x_short = BatchNormalization(axis = 3)(x_short)

        # add shortcut to main branch 
        x = Add()([x,x_short])
        x = Activation(self.activation_func)(x)

        return x

    def resnet_identity_block(self, x, filters,f=3): 
        
        F1, F2, F3 = filters
        x_short = x

        #main identity block branch
        x = Conv2D(filters=F1, kernel_size=(1,1),strides = (1,1),padding = 'valid', kernel_initializer=self.initializer)(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation(self.activation_func)(x)

        x = Conv2D(filters=F2, kernel_size=(f,f), strides = (1,1),padding = 'same',kernel_initializer=self.initializer)(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation(self.activation_func)(x)

        x = Conv2D(filters=F3, kernel_size=(1,1),strides = (1,1), padding = 'valid',kernel_initializer=self.initializer)(x)
        x = BatchNormalization(axis = 3)(x)

        # add shortcut to main branch 
        x = Add()([x,x_short])
        x = Activation(self.activation_func)(x)

        return x

    def LSTM_branch(self, time_steps=20, N_features=1, N_layers=1):
        
        #define the input
        input_shape =  (time_steps, N_features)
        inputs = Input(shape=input_shape)

        #create N layers
        for i in range(N_layers):

            if i == 0:
                # first add the inputs
                x = inputs

            if i < N_layers-1:
                x = tf.keras.layers.LSTM(time_steps,return_sequences=True)(x)      
        
        #add last layer
        x = tf.keras.layers.LSTM(time_steps)(x)
        
        #create the model
        model = Model(inputs = inputs, outputs = x)
        return model

    def Head_branch(self, model1, N_layers=3, model2=None,):

        #concantinate output layers
        if model2 != None:
            inputs1 = concatenate([Flatten()(model1.output), Flatten()(model2.output)])
        else:
            inputs1 = Flatten()(model1.output)
    

        units = [100, 50, 25] #np.linspace(N_First_FC_nodes,self.N_actions,N_layers)

        #create the FC layers
        if N_layers==-1:
            FC=inputs1
        else:        
                for i in range(N_layers-1): 
                    if i == 0 :
                        FC = inputs1
                    FC = Dense(units = round(units[i]),  activation=self.activation_func ,kernel_initializer=self.initializer)(FC)
                
        FC = Dense(units = self.N_actions, activation='linear')(FC)
        

        return FC    