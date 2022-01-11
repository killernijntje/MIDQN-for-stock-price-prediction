import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import random
import math
import copy
import os
import Expierence_replay as ExRep


class Q_learning:

    def __init__(self, model, target_model, batch_size, Max_iterations, learning_rate,
                 minimal_eploration_rate, Gamma, parameter_update_int, target_parameter_update_int,
                 transA_penalty, memory_capacity, minimal_memory_needed , exploration_rate, seed=1,GPU=1, timeframe=20):

        self.model = model
        self.target_model = target_model
        self.batch_size = batch_size
        self.MaxIter = Max_iterations
        self.lr = learning_rate
        self.B = parameter_update_int
        self.C = target_parameter_update_int
        self.P = transA_penalty
        self.Gamma = Gamma
        self.eps = exploration_rate
        self.eps_min = minimal_eploration_rate
        self.N_actions = model.output_shape[-1]
        self.timeframe = model.input_shape[1][1]
        self.M = memory_capacity
        self.M_min = minimal_memory_needed
        self.seed = seed

        if GPU == 1:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    def Q_train(self, CNN_branch=1, LSTM_branch=1, QL=0, img='pure', data_directory=r'D:\1 - School\Econometrics\2020 - 2021\Scriptie\Explainable AI\Scripts\Data\stock_data_clean.csv', train_split=0.8, validation_split=0.10, test_split=0.10):
        
        '''

            - CNN_branch: 1 if the model includes a CNN branch, 0 of not. Default is set to 1. 
            - LSTM_branch: 1 if the model includes a CNN branch, 0 of not. Default is set to 1. 
            - QL: Select Q-learing method: 1 = Q-learning, 2 = Double Q-learning, 3 = MI Double Q-learning. Default set to 1.
            - img: Used to select different data types from local machine. Not required for training if directory is set manually.
        '''


        #set seeds 
        os.environ['PYTHONHASHSEED']=str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.compat.v1.set_random_seed(self.seed)

        # set target and online network weights equal
        self.target_model.set_weights(self.model.get_weights())

        # format the data
        data = pd.read_csv(data_directory, header=[0, 1], index_col=0)
        data.index = pd.to_datetime(data.index)
        data = np.log(data) - np.log(data.shift(1))
        symbols = data.columns.unique(level=0)
        timeframes = data.index.shape[0] - self.timeframe+1
        train_tfs = np.round(timeframes*train_split)

        # set image directory
        if img == 'pure':
            images_directory = r'C:\Users\robin\1 - Scriptie\images\intraday\pure_colour_scheme'
        elif img == 'mixed':
            images_directory = r'C:\Users\robin\1 - Scriptie\images\intraday\mixed_colour_scheme'
        elif img == 'binary':
            if self.timeframe == 64: 
                images_directory = r'D:\1 - School\Econometrics\2020 - 2021\Scriptie\Explainable AI\Scripts\Data\binary_colour_scheme.64frames'
            elif self.timeframe == 32: 
                images_directory = r'D:\1 - School\Econometrics\2020 - 2021\Scriptie\Explainable AI\Scripts\Data\binary_colour_scheme.32frames'
        # set optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)

        # set memory buffer
        memory = ExRep.Experience_replay(self.M, LSTM_branch, CNN_branch)

        # set training performance trackers
        loss_data = np.zeros((int(np.round(self.batch_size * self.MaxIter/self.B)+1),1))
        reward_data = np.zeros((self.MaxIter,1))

        # iteration counter
        b = 1
        loss_print = 0
        total_R = 0
        esp_R = 0

        while True:
            # randomly choose Company and time
            c = np.random.choice(symbols)
            t = np.random.randint(2, train_tfs, 1)

            # create memory for buffer
            if LSTM_branch == 1 and CNN_branch == 1:

                # retrieve the image for t
                if img =='binary': 
                    S_img_dir_t = S_img_dir_t = images_directory + '\\' + \
                        str(c) + '_' + str(t[0]) + '.txt'
                    S_img_t = 255*np.genfromtxt(S_img_dir_t, delimiter=" ").reshape((self.timeframe,self.timeframe,1))

                else: 
                    S_img_dir_t = images_directory + '\\' + \
                        str(c) + '_' + str(t[0]) + '.png'
                    S_img_t = tf.keras.preprocessing.image.load_img(S_img_dir_t)
                    S_img_t = keras.preprocessing.image.img_to_array(S_img_t)
                # retrieve the data t
                S_data_t = data[c][t[0]:t[0]+self.timeframe].Close

                # Use greedy exploration for t-1
                if self.eps > np.random.rand(1)[0]:  # b < exploration_time or
                    # random action
                    A = np.random.choice(self.N_actions)
                else:
                    # Make prediction
                    A_pred = self.model(
                        [tf.expand_dims(S_img_t, 0), tf.expand_dims(S_data_t, 0)], training=False)
                    if QL==2: 
                        A = tf.argmax(A_pred[1][0]).numpy()  
                    else:
                        A = tf.argmax(A_pred[0]).numpy()  

                    

                # Calculate the reward
                R = self.calculate_reward(A, data[c][t[0]+self.timeframe:t[0]+self.timeframe+1].Close)
                
                # retrieve the image for t+1
                if img =='binary': 
                    S_img_dir_next = S_img_dir_t = images_directory + '\\' + \
                        str(c) + '_' + str(t[0]) + '.txt'
                    S_img_next = 255*np.genfromtxt(S_img_dir_next, delimiter=" ").reshape((self.timeframe,self.timeframe,1))
                else: 
                    S_img_dir_next = images_directory + '\\' + \
                        str(c) + '_' + str(t[0]+1) + '.png'
                    S_img_next = tf.keras.preprocessing.image.load_img(
                        S_img_dir_next)
                    S_img_next = keras.preprocessing.image.img_to_array(S_img_next)


                # retrieve the data t+1
                S_data_next = data[c][t[0]+1:t[0]+self.timeframe+1].Close

                # store S,A,R,S' in the memory buffer
                memory.remmember(S_img_t, S_data_t, A, R,
                                 S_img_next, S_data_next)

            elif LSTM_branch == 1 and CNN_branch == 0:

                # retrieve the data t
                S_data_t = data[c][t[0]:t[0]+self.timeframe].Close

                # Use greedy exploration for t-1
                if self.eps > np.random.rand(1)[0]:
                    # random action
                    A = np.random.choice(self.N_actions)

                else:
                    # Make prediction
                    A_pred = self.model([tf.expand_dims(S_data_t, 0)], training=False)
                    A = tf.argmax(A_pred[0]).numpy()

                # Calculate the reward
                R = self.calculate_reward(
                    A, data[c][t[0]+self.timeframe:t[0]+self.timeframe+1].Close, self.P)

                # retrieve the data t+1
                S_data_next = data[c][t[0]+1:t[0]+self.timeframe+1].Close

                # store S,A,R,S' in the memory buffer
                memory.remmember([], S_data_t, A, R, [], S_data_next)

            elif LSTM_branch == 0 and CNN_branch == 1:

                # retrieve the image for t
                if img =='binary': 
                    S_img_dir_t = images_directory + '\\' + \
                        str(c) + '_' + str(t[0]) + '.txt'
                    S_img_t = pd.read_csv(S_img_dir_t, sep=" ", header=None)
                else: 
                    S_img_dir_t = images_directory + '\\' + \
                        str(c) + '_' + str(t[0]) + '.png'
                    S_img_t = tf.keras.preprocessing.image.load_img(S_img_dir_t)
                    S_img_t = keras.preprocessing.image.img_to_array(S_img_t)

                # Use greedy exploration for t-1
                if self.eps > np.random.rand(1)[0]:  # b < exploration_time or
                    # random action
                    A = np.random.choice(self.N_actions)
                else:
                    # Make prediction
                    A_pred = self.model(
                        [tf.expand_dims(S_img_t, 0), tf.expand_dims(S_data_t, 0)], training=False)
                    A = tf.argmax(A_pred[0]).numpy()

                # Calculate the reward
                R = self.calculate_reward(A, data[c][t[0]+self.timeframe:t[0]+self.timeframe+1].Close)

                # retrieve the image for t+1
                if img =='binary': 
                    S_img_dir_next = images_directory + '\\' + \
                        str(c) + '_' + str(t[0]) + '.txt'
                    S_img_next = pd.read_csv(S_img_dir_next, sep=" ", header=None)
                else: 
                    S_img_dir_next = images_directory + '\\' + \
                        str(c) + '_' + str(t[0]+1) + '.png'
                    S_img_next = tf.keras.preprocessing.image.load_img(
                        S_img_dir_next)
                    S_img_next = keras.preprocessing.image.img_to_array(S_img_next)

                # store S,A,R,S' in the memory buffer
                memory.remmember(S_img_t, [], A, R, S_img_next, [])

            #update total reward
            total_R += R
            reward_data[b-1]=R
            # caculate the episode reward
            if b >= self.M_min-self.C:
                esp_R += R
                
            # Update exploration rate
            if self.eps > self.eps_min:
                self.eps = self.eps ** b 

            # update the online network
            if (b >= self.M_min) and (b % self.B == 0):

                # choose the Q-learing algorithm
                # Q-learning : QL=0
                # Double Q-learning : QL=1
                # MI Double Q-learning : Ql=2

                if QL == 0:  # QL
                    if LSTM_branch == 1 and CNN_branch == 1:
                        # Get memory batch
                        S_img_sample, S_data_sample, A_sample, R_sample, S_img_next_sample, S_data_next_sample = memory.get_batch(self.batch_size)
                        # Get possible future rewards
                        R_pred = self.target_model.predict([np.array(S_img_next_sample), np.array(S_data_next_sample)])
                        R_pred_val = self.target_model.predict([np.array(S_img_next_sample), np.array(S_data_next_sample)])

                    elif LSTM_branch == 1 and CNN_branch == 0:
                        # Get memory batch
                        S_data_sample, A_sample, R_sample, S_data_next_sample = memory.get_batch(self.batch_size)
                        # Get possible future rewards
                        R_pred = self.target_model.predict([np.array(S_data_next_sample)])

                    elif LSTM_branch == 0 and CNN_branch == 1:
                        S_img_sample, A_sample, R_sample, S_img_next_sample = memory.get_batch(self.batch_size)
                        # Get possible future rewards
                        R_pred = self.target_model.predict([np.array(S_img_next_sample)])

                    # Calculate Q-values
                    Q_values_updated = R_sample + self.Gamma * tf.reduce_max(R_pred, axis=1)
                    
                elif QL == 1 :  # DQL
                    if LSTM_branch == 1 and CNN_branch == 1:
                        # Get memory batch
                        S_img_sample, S_data_sample, A_sample, R_sample, S_img_next_sample, S_data_next_sample = memory.get_batch(self.batch_size)
                        R_pred      = self.target_model.predict([np.array(S_img_next_sample), np.array(S_data_next_sample)])
                        R_val_pred  = self.model.predict([np.array(S_img_next_sample), np.array(S_data_next_sample)])
                    
                    elif LSTM_branch == 1 and CNN_branch == 0:
                        # Get memory batch      
                        S_data_sample, A_sample, R_sample, S_data_next_sample = memory.get_batch(self.batch_size)
                        
                        R_pred = self.target_model.predict([np.array(S_data_next_sample)])
                        R_val_pred  = self.model.predict([np.array(S_data_next_sample)])
                    
                    elif LSTM_branch == 0 and CNN_branch == 1:
                        # Get memory batch      
                        S_img_sample, A_sample, R_sample, S_img_next_sample = memory.get_batch(self.batch_size)

                        R_pred = self.target_model.predict([np.array(S_img_next_sample)])
                        R_val_pred  = self.model.predict([np.array(S_img_next_sample)])
                    
                    A_pred = tf.argmax(R_pred, axis=1)
                    A_pred_masks = tf.one_hot(A_pred, self.N_actions, on_value=None, off_value=None)

                    Q_double_val = tf.reduce_sum(tf.multiply(A_pred_masks, R_val_pred),axis=1)
                    Q_values_updated = R_sample + tf.math.scalar_mul(self.Gamma, Q_double_val) 
                elif QL == 2: # MIDQL
                    # Get memory batch
                    S_img_sample, S_data_sample, A_sample, R_sample, S_img_next_sample, S_data_next_sample = memory.get_batch(self.batch_size)
                    R_pred      = self.target_model.predict([np.array(S_img_next_sample), np.array(S_data_next_sample)])
                    R_val_pred  = self.model.predict([np.array(S_img_next_sample), np.array(S_data_next_sample)])
                    
                    A_pred = tf.argmax(R_pred[1][0]).numpy()
                    A_pred_masks = tf.one_hot(A_pred, self.N_actions, on_value=None, off_value=None)

                    Q1_double_val = tf.reduce_sum(tf.multiply(A_pred_masks, R_val_pred[1]),axis=1)
                    Q2_double_val = tf.reduce_sum(tf.multiply(A_pred_masks, R_val_pred[0]),axis=1)
                    Q3_double_val = tf.reduce_sum(tf.multiply(A_pred_masks, R_val_pred[2]),axis=1)
                    
                    Q1_values_updated = R_sample + tf.math.scalar_mul(self.Gamma, Q1_double_val)
                    Q2_values_updated = R_sample + tf.math.scalar_mul(self.Gamma, Q2_double_val)
                    Q3_values_updated = R_sample + tf.math.scalar_mul(self.Gamma, Q3_double_val)

                # Create masks to calculate loss for the updated Q_values
                masks = tf.one_hot(A_sample, self.N_actions, on_value=None, off_value=None)

                with tf.GradientTape() as tape:
                    # train the model
                    if LSTM_branch == 1 and CNN_branch == 1 :
                        Q_values = self.model([np.array(S_img_sample), np.array(S_data_sample)])

                    elif LSTM_branch == 1 and CNN_branch == 0 :
                        Q_values = self.model(np.array(S_data_sample))

                    elif LSTM_branch == 0 and CNN_branch == 1 :
                        Q_values = self.model(np.array(S_img_sample))

                    # Apply Mask
                    if QL==2: 
                        Q1_actions = tf.reduce_sum(tf.multiply(Q_values[1], masks),axis=1)
                        Q2_actions = tf.reduce_sum(tf.multiply(Q_values[0], masks),axis=1)
                        Q3_actions = tf.reduce_sum(tf.multiply(Q_values[2], masks),axis=1)

                       
                        loss        = (5/7) * self.loss_function(Q1_values_updated, Q1_actions) + (1/7)*self.loss_function(Q2_values_updated, Q2_actions) +(1/7)*self.loss_function(Q3_values_updated, Q3_actions) 
                        loss_print  = (5/7) * self.loss_function(Q1_values_updated, Q1_actions) + (1/7)*self.loss_function(Q2_values_updated, Q2_actions) + (1/7)*self.loss_function(Q3_values_updated, Q3_actions) 
                    else:
                        Q_actions = tf.reduce_sum(tf.multiply(Q_values, masks),axis=1)

                        # calculate the loss
                        loss = self.loss_function(Q_values_updated, Q_actions)
                        loss_print = self.loss_function(
                            Q_values_updated, Q_actions)
                    
                    loss_data[b-1] = loss_print

                    # Backpropegation
                    gradients = tape.gradient(
                        loss, self.model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(gradients, self.model.trainable_variables))
            # update target network
            if (b >= self.M_min) and (b % self.C == 0):

                # update weights of target model
                self.target_model.set_weights(self.model.get_weights())

                # calculate the running reward

                # print progress
                print_temp = "Episode: {}, Loss:{:.5f}, Eps reward: {}, Avg total Reward: {}"
                print(print_temp.format(b/(self.C),
                                        loss_print, esp_R/self.C, total_R/b))
                esp_R = 0

            # update counter
            b += 1

            # Check if done and save model
            if(b >= self.MaxIter):
                self.target_model.save(
                    r'D:\1 - School\Econometrics\2020 - 2021\Scriptie\Explainable AI\Scripts\Models')
                print('Finish!')
                return loss_data, reward_data

    def loss_function(self, Q_up, Q_A):

        # set loss function
        loss = tf.math.divide(tf.reduce_sum(tf.square(Q_up - Q_A)), self.batch_size)

        return loss


    def calculate_reward(self, A_next, R, simple_reward=1):

        curr_action = 1-2*A_next
        reward = np.array(100*curr_action*(np.exp(R)-1)) 
        return reward
