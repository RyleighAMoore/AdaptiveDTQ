# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:10:36 2019

@author: Ryleigh
"""
import numpy as np
import matplotlib.pyplot as plt
val=[]


def plotMaxDiffTrajGraph(One,Two):
#    One = PdfTraj
#    Two = surfaces
    val=[]
    sizing = min(len(One), len(Two))
    for i in range(sizing):
        value = np.max(np.max(abs(np.reshape(One[i],[25,25])-Two[i])))
        val.append(value)
        print(i)
    w = np.linspace(1, sizing,sizing)
    fig, ax = plt.subplots()

    # We change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=30)
    plt.semilogy(w, np.abs(val), linewidth='20')
    ax.yaxis.get_offset_text().set_fontsize(24)

    plt.xlabel('Iteration', fontsize='50')
    plt.ylabel('difference', fontsize='50') 
    plt.xticks(fontsize='50')
    plt.yticks(fontsize='50')
    plt.subplots_adjust(top=0.88,bottom=0.175,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    
    print(val)
    plt.show()
    
plotMaxDiffTrajGraph(PdfTraj,surfaces)
    
def plotL2DiffTrajGraph(One,Two):
    sizing = min(len(One), len(Two))
    for i in range(sizing):
        value = np.sqrt(np.sum(abs(np.reshape(One[i],[25,25])-Two[i])**2))
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
    
    

import pickle
#pickle.dump(PdfTraj, open( "PDF.p", "wb" ) )
#pickle.dump(Meshes, open( "Meshes.p", "wb" ) )
pickle_in = open("PDF.p","rb")
PdfTraj = pickle.load(pickle_in)

pickle_in = open("Meshes.p","rb")
Meshes = pickle.load(pickle_in)