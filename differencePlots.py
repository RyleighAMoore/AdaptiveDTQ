# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:10:36 2019

@author: Ryleigh
"""
import numpy as np
import matplotlib.pyplot as plt
val=[]

def plotMaxDiffTrajGraph(One,Two):
    sizing = min(len(One), len(Two))
    for i in range(sizing):
        value = np.max(np.max(abs(np.reshape(One[i],[50,50])-Two[i])))
        val.append(value)
        print(i)
    w = np.linspace(1, sizing,sizing)
    fig, ax = plt.subplots()
    ax.semilogy(w,val, linewidth='20')
    plt.xlabel('Iteration', fontsize='80')
    plt.ylabel('difference', fontsize='80') 
    plt.xticks(fontsize='80')
    plt.yticks(fontsize='80')
    plt.subplots_adjust(top=0.88,bottom=0.175,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    ax.yaxis.offsetText.set_fontsize(80)
    print(val)
    plt.show()
    
def plotL2DiffTrajGraph(One,Two):
    sizing = min(len(One), len(Two))
    for i in range(sizing):
        value = np.sqrt(np.sum(abs(np.reshape(One[i],[40,40])-Two[i])**2))
        val.append(value)
        print(i)
    w = np.linspace(1, sizing,sizing)
    fig, ax = plt.subplots()
    ax.plot(w,val, linewidth='20')
    plt.xlabel('Iteration', fontsize='80')
    plt.ylabel('difference', fontsize='80') 
    plt.xticks(fontsize='80')
    plt.yticks(fontsize='80')
    plt.subplots_adjust(top=0.88,bottom=0.175,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    ax.yaxis.offsetText.set_fontsize(80)
    print(val)
    plt.show()