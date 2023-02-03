import os
import numpy as np
#
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd
#
from itertools import chain

def piecewise_function(x, breaks, slopes):
    '''
    A piecewise linear function from breaks a slopes
    '''
    # ADDING CONDITION FOR CONTINUITY
    coeff = [(slopes[0], 0)]
    for i in range(1, len(slopes)):
        q = coeff[-1][1] + (slopes[i-1]-slopes[i]) * breaks[i]
        coeff.append( (slopes[i], q) )
    return np.piecewise(x, [x >= b for b in breaks], [lambda x=x, m=c[0], q=c[1]: m*x + q for c in coeff])


def printMultiHorizon(results, horizon, listToPlot = [] , fig_path=None):
    '''
    results is a dict that contains one key for each employed method
    on a multistage setting.
    '''
    keys = {}

    if len(listToPlot) == 0: #plot all
        keys = results.keys()
    else:
        keys = listToPlot

    ####mean values
    meanCumProfits = {}
    for k in keys:
        meanCumProfits[k] = np.mean(results[k]['cumProfit'], axis =0)
    #data frames dict 
    dfCumProfit = {}
    dfAvgInventory = {}
    dfProduction = {}
    dfLostSales = {}
    #data frames population
    for k in keys:
        dfCumProfit[k] = pd.melt(frame = pd.DataFrame(results[k]['cumProfit']),
            var_name = 'column',
            value_name = 'value')
        #
        dfAvgInventory[k] = pd.melt(frame = pd.DataFrame(results[k]['inventory']),
            var_name = 'column',
            value_name = 'value')
        #
        dfProduction[k] = pd.melt(frame = pd.DataFrame(results[k]['production']),
            var_name = 'column',
            value_name = 'value')
        #
        dfLostSales[k] = pd.melt(frame = pd.DataFrame(results[k]['lostSales']),
            var_name = 'column',
            value_name = 'value')

    ##########
    ##########Plot performances

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

    rc('font', **font)
    rc('axes', linewidth=2)
    rc('xtick',labelsize=15)
    rc('ytick',labelsize=15)
    #rc("font", size="12")

    values = np.arange(0,horizon,2)
    colors = plt.cm.Dark2.colors
    lines = ["-","--","-.",":"]
    ########Cumulative profits
    ax = plt.subplot()
    for j,k in enumerate(dfCumProfit.keys()):
        sns.lineplot(ax = ax,
                data = dfCumProfit[k],
                x = 'column',
                y = 'value',
                linestyle = lines[j%4],
                sort = False, color = colors[j%6], label=k)
    plt.ticklabel_format(style='sci', axis='y',scilimits = [-4,4])
    plt.ylabel('Cumulative profits',fontsize=18,weight='bold')
    plt.xlabel('Month',fontsize=18,weight='bold')
    plt.xticks(values,np.arange(0,horizon,2)%12 + 1)
    plt.tight_layout()
    if fig_path:
        plt.savefig(os.path.join(fig_path, 'cumulativeProfits.pdf'))
    plt.show()
    plt.close()
    ########
    ########Production
    ########
    ax = plt.subplot()
    for j,k in enumerate(dfProduction.keys()):
        sns.lineplot(ax = ax,
                data = dfProduction[k],
                x = 'column',
                y = 'value',
                linestyle = lines[j%4],
                sort = False, color = colors[j%6], label=k)
    plt.ticklabel_format(style='sci', axis='y',scilimits = [-4,4])
    plt.ylabel('Production costs',fontsize=18,weight='bold')
    plt.xlabel('Month',fontsize=18,weight='bold')
    plt.xticks(values,np.arange(0,horizon,2)%12 + 1)
    plt.tight_layout()
    if fig_path:
        plt.savefig(os.path.join(fig_path, 'productionCost.pdf'))
    plt.show()
    plt.close()
    ########
    ########LostSales
    ########
    ax = plt.subplot()
    for j,k in enumerate(dfLostSales.keys()):
        sns.lineplot(ax = ax,
                data = dfLostSales[k],
                x = 'column',
                y = 'value',
                linestyle = lines[j%4],
                sort = False, color = colors[j%6], label=k)
    plt.ticklabel_format(style='sci', axis='y',scilimits = [-4,4])
    plt.ylabel('Lost Sales',fontsize=18,weight='bold')
    plt.xlabel('Month',fontsize=18,weight='bold')
    plt.xticks(values,np.arange(0,horizon,2)%12 + 1)
    plt.tight_layout()
    if fig_path:
        plt.savefig(os.path.join(fig_path, 'lostSales.pdf'))
    plt.show()
    plt.close()
    ########
    ########Inventory
    ########
    ax = plt.subplot()
    for j,k in enumerate(dfAvgInventory.keys()):
        sns.lineplot(ax = ax,
                data = dfAvgInventory[k],
                x = 'column',
                y = 'value',
                linestyle = lines[j%4],
                sort = False, color = colors[j%6], label=k)
    plt.ticklabel_format(style='sci', axis='y',scilimits = [-4,4])
    plt.ylabel('Inventory levels',fontsize=18,weight='bold')
    plt.xlabel('Month',fontsize=18,weight='bold')
    plt.xticks(np.concatenate([np.array([0]),np.arange(2,horizon+1,2)]),np.concatenate([np.array([0]),np.arange(0,horizon,2)%12 + 1]))
    plt.tight_layout()
    if fig_path:
        plt.savefig(os.path.join(fig_path, 'inventoryLevels.pdf'))
    plt.show()
    plt.close()
    ########
    ########Histograms of the orders
    ########
    profitsHist = {}
    for k in keys:
        profitsHist[k] = list(chain(*results[k]['profits']))
    ax = plt.subplot()
    for j,k in enumerate(profitsHist.keys()):
        ax.hist(profitsHist[k],label=k,color=colors[j%6],histtype = 'step')
    plt.ylabel('Frequency',fontsize=18,weight='bold')
    plt.xlabel('Profits',fontsize=18,weight='bold')
    plt.legend()
    plt.tight_layout()
    if fig_path:
        plt.savefig(os.path.join(fig_path, 'profitsHist.pdf'))
    plt.show()
    plt.close()
    ########
    ########Histograms of the time
    ########
    timesHist = {}
    for k in keys:
        timesHist[k] = list(chain(*results[k]['time']))
    ax = plt.subplot()
    for j,k in enumerate(timesHist.keys()):
        ax.hist(timesHist[k],label=k,color=colors[j%6],histtype = 'step')
    plt.ylabel('Frequency',fontsize=16)
    plt.xlabel('Time (s) per timestep',fontsize=16)
    plt.legend()
    plt.tight_layout()
    if fig_path:
        plt.savefig(os.path.join(fig_path, 'timeHist.pdf'))
    plt.show()
    plt.close()

