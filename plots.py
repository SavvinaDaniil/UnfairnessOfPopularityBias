#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 18:28:20 2022

@author: savvina
"""
#%%
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random as rd
my_seed = 0
rd.seed(my_seed)
np.random.seed(my_seed)
#%%
def plot_data_distribution(item_dist, item_col, dividing = [False, 0], log = False, save = False, addition = ""):
    plt.figure()
    ax = plt.axes()
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['left'].set_zorder(0)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    ax.set_facecolor("aliceblue")
    plt.grid(color = "w",linewidth = 2 )
    if dividing[0]:
        x0 = int(len(item_dist.values)*dividing[1])
        y = range(len(item_dist))
        plt.plot(y[:x0+1], item_dist.values[:x0+1], label = "Popular "+item_col+"s", linewidth = 5)
        plt.plot(y[x0:], item_dist.values[x0:], label = "Non Popular "+item_col+"s", linewidth = 5)
    else:
        plt.plot(item_dist.values)
    plt.xticks(fontsize='13')
    plt.yticks(fontsize='13')
    add = ""
    if log:
        plt.xscale('log')
        plt.yscale('log')
        add = "_(log)"
    plt.xlabel(item_col+add, fontsize='14')
    plt.ylabel('Number of users' + add, fontsize='15')
    if save:
        if dividing[0]:
            plt.savefig('graphs/'+item_col+add+"_dist_div"+addition+".png", bbox_inches='tight')
        else:
            plt.savefig('graphs/'+item_col+add+"_dist"+addition+".png", bbox_inches='tight')
    plt.show(block=True)
def plot_popularity_distribution(pop_fraq, item_col, dividing = [False,0], save = False, addition = ""):
    plt.figure()
    ax = plt.axes()
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['left'].set_zorder(0)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
    ax.set_facecolor("aliceblue")
    plt.grid(color = "w",linewidth = 2 )
    if dividing[0]:
        y = range(len(pop_fraq))
        x0 = int(len(y)*dividing[1]) 
        x1 = int(len(y)*(1-dividing[1]))
        x= sorted(pop_fraq)
        plt.plot(y[:x0+1],x[:x0+1], label="LowMS users", linewidth = 5)
        plt.plot(y[x0:x1+1],x[x0:x1+1], label = "MedMS users", linewidth = 5)
        plt.plot(y[x1:],x[x1:], label = "HighMS users", linewidth =5)
    else:
        plt.plot(sorted(pop_fraq))
    plt.xlabel('User', fontsize='15')
    plt.xticks(fontsize='13')
    plt.ylabel('Ratio of popular '+item_col+'s', fontsize='15')
    plt.yticks(fontsize='13')
    plt.axhline(y=0.8, color='black', linestyle='--', label='80% ratio of popular '+item_col+'s')
    plt.legend(fontsize='15')
    #plt.savefig('data/ECIR/user_artist_ratio.png', dpi=300, bbox_inches='tight')
    if save:
        if dividing[0]:
            plt.savefig('graphs/'+item_col+"_pop_dist_div"+addition+".png", bbox_inches='tight')
        else:
            plt.savefig('graphs/'+item_col+"_pop_dist"+addition+".png", bbox_inches='tight')
    plt.show(block=True)

def plot_Lorenz(movs,cdf, item_col = "movie", save = False, addition = ""):
    def f(t):
        return t
    plt.plot(movs*100, cdf*100, linewidth = 3, color = "red", label = "L(x) actual")
    plt.plot(movs*100, movs*100, linewidth = 3, color = "blue", label = "L(x) = x, distributional equality")
    plt.xlabel("100x% least consumed items")
    plt.ylabel("100y% of total amount of consumptions")
    section = movs*100
    plt.legend()
    plt.fill_between(section, f(section), color = "lightgrey")
    plt.fill_between(movs*100, cdf*100, color = "gray")
    if save:
        plt.savefig('graphs/'+item_col+"_data_Lorenz"+addition+".png", bbox_inches='tight')
    plt.show(block=True)

def plot_profile_size_vs_popularity(pop_metric, user_hist, way, item_col, save = False, addition = ""):
    plt.figure()
    ax = plt.axes()
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['left'].set_zorder(0)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
    ax.set_facecolor("aliceblue")
    plt.grid(color = "w",linewidth = 2 )
    slope, intercept, r_value, p_value, std_err = stats.linregress(user_hist, pop_metric)
    print('R-value: ' + str(r_value))
    line = slope * np.array(user_hist) + intercept
    plt.plot(user_hist, pop_metric, 'o', user_hist, line)
    plt.xlabel('User profile size', fontsize='15')
    plt.xticks(fontsize='13')
    if way == "count":
        ylabel = "Number of popular "+item_col+"s"
    elif way == "percentage":
        ylabel = 'Percentage of popular '+item_col+'s'
    else:
        ylabel = "Average popularity of "+item_col+"s"
    plt.ylabel(ylabel, fontsize='15')
    plt.yticks(fontsize='13')
    #plt.savefig('data/ECIR/corr_user_pop.png', dpi=300, bbox_inches='tight')
    if save:
        plt.savefig('graphs/'+item_col+"_"+way+"_vs_size"+addition+".png", bbox_inches='tight')
    plt.show(block=True)
def plot_group_characteristics(low_nr, med_nr, high_nr, way, item_col, save = False, addition = ""):
    plt.figure()
    ax = plt.axes()
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['left'].set_zorder(0)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
    ax.set_facecolor("aliceblue")
    plt.bar(np.arange(3), [low_nr, med_nr, high_nr])
    plt.xticks(np.arange(3), ['LowMS', 'MedMS', 'HighMS'])
    plt.xlabel('User group')
    if way=="size":
        ylabel = 'Average user profile size'
    else:
        ylabel = "Number of users per group"
    plt.ylabel(ylabel)
    
    print('LowMS: ' + str(low_nr))
    print('MedMS: ' + str(med_nr))
    print('HighMS: ' + str(high_nr))
    if save:
        plt.savefig('graphs/'+item_col+"_"+way+"_groups"+addition+".png", bbox_inches='tight')
    plt.show(block=True)
def plot_algorithm_results(algo_names, df_item_dist, item_col, save = False, addition = ""):
    for i in range(0, len(algo_names)):
        plt.figure()
        ax = plt.axes()
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        ax.spines['left'].set_zorder(0)
        ax.xaxis.set_ticks_position('none') 
        ax.yaxis.set_ticks_position('none') 
        print("OIK")
        ax.set_facecolor("aliceblue")
        plt.grid(color = "w",linewidth = 2 )
        x = df_item_dist['count']
        y = df_item_dist[algo_names[i]]
        #slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        #line = slope * np.array(x) + intercept
        #print(r_value)
        #if algo_names[i] != 'Random' and algo_names[i] != 'MostPopular':
         #   plt.gca().set_ylim(0,300)
        plt.plot(x, y, 'o')#, x, line)
        plt.xlabel(item_col+' popularity', fontsize='15')
        plt.ylabel('Recommendation frequency', fontsize='15')
        plt.xticks(fontsize='13')
        plt.yticks(fontsize='13')
        if save:
            plt.savefig('graphs/'+item_col+"_"+algo_names[i]+"_KF5.png", bbox_inches='tight')
        plt.show(block=True)
