# MIDQN-for-stock-price-prediction

This repo consists of the code used in my thesis where I applied Mixed Input Deep Q-learning to the stockmarket. In the repo you can find the following scripts:

1) `data_scrapper.py` : Given a list of tickers it scrappes historical data for stocks. 
2) `data_cleaner.ipynb` : Cleans the data outputted by the scapper.
3) `Binary data generator.ipynb` : Takes the cleaned data and creates candlestick charts formatted and stores the numerical pixel data, where pixel values are scaled to [0,1].
4) `Canlestick data generator.py` : Creates candlestick plots from the cleaned data. 
5) `Main_script.ipynb` : Here the models are trained and tested.
6) `create_Q_models.py` : used to create model architecture using TensorFLow.
7) `Q_learning.py` : The Q-learning training algorithms. Contains: Q-learning, Double Q-learning and Mixed Input Q-Learning.
8) `Expierence_replay.py` : Allows for experience replay during training.
9) `test_algorithms.py` : Contains the testing enviroment and algorithms.
10) `final plots.ipynb` : Creates plots for loss and reward functions during training, perfomance during testing period (returns and mean-variance analysis).
