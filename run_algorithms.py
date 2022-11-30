#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:56:24 2022

@author: savvina
"""
#%%
import time
import numpy as np
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import NMF, SVDpp, SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split, KFold, GridSearchCV
from surprise import accuracy
from collections import defaultdict
from scipy import stats
import random as rd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
my_seed = 0
rd.seed(my_seed)
np.random.seed(my_seed)
#%%
def prepare_dataset(df_events, predict_col, item_col, seed = 0, test_size = 0.2):
    print('Min rating: ' + str(df_events[predict_col].min()))
    print('Max rating: ' + str(df_events[predict_col].max()))

    reader = Reader(rating_scale=(df_events[predict_col].min(), df_events[predict_col].max()))
    
    for col in df_events.columns:
        if (col!="user") and (col!=predict_col) and (col!=item_col):
            df_events = df_events.drop(col, axis=1)
    print(df_events.head())
    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df_events, reader)
    trainset, testset = train_test_split(data, test_size = test_size, random_state = seed)
    return trainset, testset

def prepare_dataset_kf(df_events, predict_col, item_col, seed = 0, test_size = 0.2, n_splits = 5):
    print('Min rating: ' + str(df_events[predict_col].min()))
    print('Max rating: ' + str(df_events[predict_col].max()))

    reader = Reader(rating_scale=(df_events[predict_col].min(), df_events[predict_col].max()))
    
    for col in df_events.columns:
        if (col!="user") and (col!=predict_col) and (col!=item_col):
            df_events = df_events.drop(col, axis=1)
    print(df_events.head())
    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df_events, reader)
    kf = KFold(n_splits = n_splits, random_state = seed, shuffle = True )
    return data, kf

def prepare_music_dataset(df_events, predict_col):
    scaled_df_events = pd.DataFrame()
    for user_id, group in df_events.groupby('user'):
        
        scaler = MinMaxScaler(feature_range=(1, 1000)) #"βαθμολογια" 1 με 1000
        scaled_ratings = scaler.fit_transform(group[predict_col].values.reshape(-1, 1).astype(float))
        new_rows = group.copy()
        new_rows[predict_col] = scaled_ratings
        scaled_df_events = scaled_df_events.append(new_rows)
    df_events = scaled_df_events
    return df_events

#%%
def run_grid_search(data, algorithm_name):
    if algorithm_name == "UserItemAvg":
        algorithm = BaselineOnly
        param_grid = {"bsl_options":{"method":["als","sgd"],
                                     "n_epochs":[10,20],
                                     "reg":[0,0.2,10]}}
        
    elif algorithm_name == "UserKNN":
        algorithm = KNNBasic
        param_grid = {"k":[10, 50, 100],
                      "min_k":[1, 2, 5],
                      "sim_options":{"name":["msd", "cosine", "pearson", "pearson_baseline",
                                             ]}}
    elif algorithm_name == "ItemKNN":
        algorithm = KNNBasic
        param_grid = {"k":[10, 50, 100],
                      "min_k":[1, 2, 5],
                      "sim_options":{"name":["msd", "cosine", "pearson", "pearson_baseline",
                                             ], "user_based": [False]}}
    elif algorithm_name == "UserKNNAvg":
        algorithm = KNNWithMeans
        param_grid = {"k":[10, 50, 100],
                      "min_k":[1, 2, 5],
                      "sim_options":{"name":["msd", "cosine", "pearson", "pearson_baseline",
                                             ]}}
    elif algorithm_name == "NMF":
        algorithm = NMF
        param_grid = {"n_factors":[10,50,100], "n_epochs":[10,20,50], "biased":[True, False]}
    elif algorithm_name == "SVD":
        algorithm = SVD
        param_grid = {"n_factors":[10,50,100], "n_epochs":[10,20,50], "biased":[True, False],
                        "lr_all":[0, 0.005, 0.1, 1], "reg_all":[0, 0.02, 1],}
    gs = GridSearchCV(algorithm, param_grid, measures= ["rmse"], cv = 5, joblib_verbose = 10, n_jobs = -1)
    gs.fit(data)
    return gs.best_params["rmse"]
#%%
def train_algorithms(df_item_dist, trainset, testset, item_dist, no_users, low_users, medium_users, high_users):
    sim_users = {'name': 'cosine', 'user_based': True}  # compute cosine similarities between users
    # είδος αποστασης=κοσαιν, ειδος κνν = χρηστη
    algos = [] # Random and MostPopular is calculated by default
    algos.append(None)#Random())
    algos.append(None)#MostPopular())
    algos.append(BaselineOnly()) #αυτο το UserItemAvg, τι ειναι?
    algos.append(KNNBasic(sim_options = sim_users, k=40)) 
    #algos.append(KNNBasic(sim_options = {'name': 'cosine', 'user_based': False}, k=40)) 
    algos.append(KNNWithMeans(sim_options = sim_users, k=40)) 
    algos.append(NMF(n_factors = 15))
    #algos.append(SVDpp(n_factors = 15))
    algos.append(SVD(n_factors = 15))
    algo_names = ['Random',
                  'MostPopular',
                  'UserItemAvg',
                  'UserKNN',
                  #"ItemKNN",
                  'UserKNNAvg',
                  'NMF',
                  'SVD']
    
    i = 0
    low_rec_gap_list = [] # one entry per algorithmus
    medium_rec_gap_list = []
    high_rec_gap_list = []
    start = time.time()
    for i in range(0, len(algo_names)): #για καθε αλγοριθμο
        print("~~~~~~~~~~~~~~~~NEW~~~~~~~~~~~~~~~~~")
        df_item_dist[algo_names[i]] = 0 #προσθετω στηλη στο ποπιουλαριτι για τον αλγοριθμο i 
        low_rec_gap = 0
        medium_rec_gap = 0
        high_rec_gap = 0
        
        # get accuracy for personalized approaches
        if algo_names[i] != 'Random' and algo_names[i] != 'MostPopular': #για μη χαζους αλγοριθμους
            algos[i].fit(trainset) #κανεις φιτ
            predictions = algos[i].test(testset) #προβλεπεις
            print(algo_names[i])#λες τελειωσε ο ταδε αλγοριθμος
            get_mae_of_groups(predictions, low_users, medium_users, high_users) #υπολογιζεις τα λαθη και τα τυπωνεις
        
        # get top-n items and calculate gaps for all algorithms
        # κανεις προτασεις αναλογα τις προβλεψεις
        if algo_names[i] == 'Random':
            top_n = get_top_n_random(testset, item_dist, n=10)
        elif algo_names[i] == 'MostPopular':
            top_n = get_top_n_mp(testset, item_dist, n=10)
        else:
            top_n = get_top_n(predictions, n=10)
        #υπολογιζεις τα gap
        low_count = 0
        med_count = 0
        high_count = 0
        for uid, user_ratings in top_n.items():
            iid_list = []
            for (iid, _) in user_ratings:
                df_item_dist.loc[iid, algo_names[i]] += 1
                iid_list.append(iid)
            gap = sum(item_dist[iid_list] / no_users) / len(iid_list)
            if uid in low_users.index:
                low_rec_gap += gap
                low_count += 1
            elif uid in medium_users.index:
                medium_rec_gap += gap
                med_count += 1
            elif uid in high_users.index:
                high_rec_gap += gap
                high_count += 1
        low_rec_gap_list.append(low_rec_gap / low_count)
        medium_rec_gap_list.append(medium_rec_gap / med_count)
        high_rec_gap_list.append(high_rec_gap / high_count)
        i += 1 # next algorithm
        end = time.time()
        print("It took " + str(np.round(end-start)) + " seconds.")
        start = time.time()
    return df_item_dist, low_rec_gap_list, medium_rec_gap_list, high_rec_gap_list

#%%
def train_algorithms_kf(df_item_dist, data, kf, item_dist, no_users, low_users, medium_users, high_users):
    algos = [] # Random and MostPopular is calculated by default
    #algos.append(None)#Random())
    #algos.append(None)#MostPopular())
    algos.append(BaselineOnly(bsl_options={"method":"sgd",
                                            "n_epochs":20, 
                                            "reg":0}))
    algos.append(KNNBasic(sim_options = {'name': 'pearson_baseline', 
                                        'user_based': True}, 
                                        k=100, min_k=5)) 
    #algos.append(KNNBasic(sim_options = {'name': 'cosine', 'user_based': False}, k=40)) 
    algos.append(KNNWithMeans(sim_options = {'name': 'pearson', 
                                        'user_based': True}, 
                                        k=100, min_k=5)) 
    algos.append(NMF(n_factors = 100, n_epochs = 50, biased=False))
    algos.append(SVD(n_factors = 10, n_epochs = 10, biased=True))
     #['Random',
                  #'MostPopular',
    algo_names =['UserItemAvg',
                 'UserKNN',
                  #"ItemKNN",
                  'UserKNNAvg',
                  'NMF',
                  'SVD']
    
    i = 0
    low_rec_gap_list = [] # one entry per algorithmus
    medium_rec_gap_list = []
    high_rec_gap_list = []
    start = time.time()
    for i in range(0, len(algo_names)): #για καθε αλγοριθμο
        print("~~~~~~~~~~~~~~~~NEW~~~~~~~~~~~~~~~~~")
        df_item_dist[algo_names[i]] = 0 #προσθετω στηλη στο ποπιουλαριτι για τον αλγοριθμο i 
        low_rec_gap = 0
        medium_rec_gap = 0
        high_rec_gap = 0
        low_count = 0
        med_count = 0
        high_count = 0
        for trainset, testset in kf.split(data):
            # get accuracy for personalized approaches
            if algo_names[i] != 'Random' and algo_names[i] != 'MostPopular': #για μη χαζους αλγοριθμους
                algos[i].fit(trainset) #κανεις φιτ
                predictions = algos[i].test(testset) #προβλεπεις
                print(algo_names[i])#λες τελειωσε ο ταδε αλγοριθμος
                get_mae_of_groups(predictions, low_users, medium_users, high_users) #υπολογιζεις τα λαθη και τα τυπωνεις
            
            # get top-n items and calculate gaps for all algorithms
            # κανεις προτασεις αναλογα τις προβλεψεις
            if algo_names[i] == 'Random':
                top_n = get_top_n_random(testset, item_dist, n=10)
            elif algo_names[i] == 'MostPopular':
                top_n = get_top_n_mp(testset, item_dist, n=10)
            else:
                top_n = get_top_n(predictions, n=10)
            #υπολογιζεις τα gap
            
            for uid, user_ratings in top_n.items():
                iid_list = []
                for (iid, _) in user_ratings:
                    df_item_dist.loc[iid, algo_names[i]] += 1
                    iid_list.append(iid)
                gap = sum(item_dist[iid_list] / no_users) / len(iid_list)
                if uid in low_users.index:
                    low_rec_gap += gap
                    low_count += 1
                elif uid in medium_users.index:
                    medium_rec_gap += gap
                    med_count += 1
                elif uid in high_users.index:
                    high_rec_gap += gap
                    high_count += 1
        low_rec_gap_list.append(low_rec_gap / low_count)
        medium_rec_gap_list.append(medium_rec_gap / med_count)
        high_rec_gap_list.append(high_rec_gap / high_count)
        i += 1 # next algorithm
        end = time.time()
        print("It took " + str(np.round(end-start)) + " seconds.")
        start = time.time()
    return df_item_dist, low_rec_gap_list, medium_rec_gap_list, high_rec_gap_list
#%%
def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def get_top_n_random(testset, item_dist, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r in testset:
        if len(top_n[uid]) == 0:
            for i in range(0, 10):
                top_n[uid].append((rd.choice(item_dist.index), i))
    return top_n



def get_top_n_mp(testset, item_dist, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r in testset:
        if len(top_n[uid]) == 0:
            for iid, count in item_dist[:n].items():
                top_n[uid].append((iid, count))
    return top_n



def get_mae_of_groups(predictions, low_users, medium_users, high_users):
    print('All: ')
    accuracy.mae(predictions)
    low_predictions = []
    med_predictions = []
    high_predictions = []
    for uid, iid, true_r, est, details in predictions:
        prediction = [(uid, iid, true_r, est, details)]
        if uid in low_users.index:
            low_predictions.append(accuracy.mae(prediction, verbose=False))
        elif uid in medium_users.index:
            med_predictions.append(accuracy.mae(prediction, verbose=False))
        else:
            high_predictions.append(accuracy.mae(prediction, verbose=False))
    print('LowMS: ' + str(np.mean(low_predictions)))
    print('MedMS: ' + str(np.mean(med_predictions)))
    print('HighMS: ' + str(np.mean(high_predictions)))
    print(stats.ttest_ind(low_predictions, high_predictions))