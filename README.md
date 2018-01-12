# kaggle_porto_seguro

This repo documents my model for the Porto Seguro's Safe Driver Prediction competition on Kaggle (https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) and summarizes lessons learned.

prep*.py cleaned data. 

model1*.py are all the models used in this ensemble.

Technically we only need l1main.py and l2main.py which are first and second level models, respectively. But it took very long to run one script with many models and unexpected termination caused trouble. Thus, we broke l*main to pieces. Two interesting features in this pipline are that
1. Multiprocessing is invoked for each CV fold.
2. Hyperparameters are determined automatically by random search or Bayesian optimization.

Here are the libraries used in this pipeline: Scikit learn, Catboost, LightGB, Vowpal Wabbit, XGB, BayesOptimization.

In this contest, the two level stacking strategy is not working as expected, mainly because I made a fatal mistake that I didn't scale the predicted probabilities from different models properly. Below I summarize some interesting solutions:
Representation learning with Autoencoders: learned an embedding matrix from the feature space, https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
