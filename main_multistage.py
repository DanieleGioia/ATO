#!/usr/bin/python3
# -*- coding: utf-8 -*-
#external libraries
import numpy as np
import time
import json
#instance
from instances import *
#samplers
from sampler import *
#solver
from solver.solverGurobi import *
#agent and envs
from envs import *
from agents import *
# FOSVA
from FOSVA import *
#Scenario Reducer
from scenarioReducer import *
#utils
from utils.utils import *

#numpy print options
np.set_printoptions(precision=4) 
np.set_printoptions(suppress=True)

# The following example retraces the models and the methods invistigated in 
# "Rolling horizon policies for multi-stage stochastic assemble-to-order problems",
# by Daniele Giovanni Gioia and Edoardo Fadda and Paolo Brandimarte.

# PLEASE NOTICE THAT IN THIS VERSION OF THE CODE, ALL THE PROBLEM ARE CONTINUOUS.
# THE ARTICLE CONSIDER INTEGER VERSIONS, WITH LONGER REQUIRED COMPUTATIONAL TIMES.

# list of methods
methods = ['TS','TS_noSeas','PI','FOSVA','MP_2','MP_3','MP_4','MS3','MS3_3','MS3_4']
#Namely:
#MP_<n>: multi-period, thus multistage that approximates the future demand trough the average value from the 2nd stage to the <n>_th.
#MS3: multi stage 3. Three stages of sight with full exploitation of the historical data, then truncated.
#MS3_<n>: Three stages of sight with full exploitation of the historical data, then approximates the future demand trough the average value from the third stage to the <n>_th.
#FOSVA: Two stages rolling policy with first order approximation of the unused components.
#TS: Two stages rolling policy with simple SAA policy.
#TS_noSeas: Two stages rolling policy with simple SAA policy that does not take into account the seasonality.
#PI: Policy with perfect information about the future demand

#list of samplers
samplers = ['hierBi'] #other available samplers ['Gaus','Bi']
#list of tightness. It limitates the machine availability w.r.t. the average demand.
tightness = [1.2]
#horizon of the simulation
horizon = 24
#fosva approximation rounds and finite difference-step
n_it_fosva = 10
step_derivative = 2
#years of in_sample historical data
yea = [3,5,10]

for tight in tightness:
    for sampler in samplers:
        for ye in yea:
            # Load setting
            fp = open(f"./etc/instance_Params.json", 'r')
            sim_setting = json.load(fp)
            fp.close()
            fp = open("./etc/ato_Params.json", 'r')
            ato_setting = json.load(fp)
            fp.close()
            fp = open("./etc/sampler_Params.json", 'r')
            smpl_setting = json.load(fp)
            fp.close()
            #tightness
            sim_setting['tightness'] = tight
            #set the sampler
            if sampler == 'hierBi':
                hierSam = HierarchicalSampler(smpl_setting,sim_setting['dict_gozinto'],BiGaussianSampler(smpl_setting))
            elif sampler == 'Bi':
                hierSam = BiGaussianSampler(smpl_setting)
            elif sampler == 'Gaus':
                #mean and std adj for comparability w.r.t. the bimodal setting
                smpl_setting['mu'] = smpl_setting['mu']*smpl_setting['p1'] + (1 - smpl_setting['p1'])*smpl_setting['mu2']
                smpl_setting['sigma'] = ((smpl_setting['sigma']*smpl_setting['p1'])**2 + ((1 - smpl_setting['p1'])*smpl_setting['sigma2'])**2)**(1/2)
                hierSam = GaussianSampler(smpl_setting)
            else:
                raise ValueError('sampler option not managed')
        
            #Conversion of the standard samplers to the multistage setting, thus enabling the seasonality
            sam = MultiStageSampler(smpl_setting,hierSam)

            #instance generation
            #the sampler is required to tune the machine availability w.r.t. the tightness
            instance = InstanceRandom(sim_setting,sam)
            # instance.plotGozinto() #It prints the gozinto matrix in the current directory

            ####training demand for a simulated Data-Driven approach
            #number of observed scenarios
            nObs = ye*12 #10 years, one per month.
            #sampling the demand the we assume to be known
            demand_known = sam.sample(nObs)
            #problem init
            settings = {}

            #initial inventory
            initialShare = 1 # % of the expected value per component
            #mean demand for components
            mean_demand = (np.mean(demand_known,axis=1) @ instance.gozinto).copy()
            #adjustment w.r.t. the assumed inventory share
            instance.inventory = initialShare * mean_demand

            ###FOSVA

            #Compute fosva
            #prblem that computes the value of a given inventory
            prb = AtoRP_approx_comp_v()
            #hyperparameters of the FOSVA algorithm
            fosva_res = run_multifosva_ato(
                instance,
                prb,
                demand_known,
                step_derivative=step_derivative,
                alpha_fun=lambda i: 10.0/(10.0 + i),
                n_iterations_fosva=n_it_fosva
            )
            #data management of the FOSVA approximation. They are translated into a dict format of a linear piecewise.
            for i in range(len(fosva_res)):
                fosva_res[i]['u'] = np.array(fosva_res[i]['u'])
                fosva_res[i]['v'] = np.array(fosva_res[i]['v'])
            #set the value function approximation
            settings['fosva_res'] = fosva_res

            ############
            ############TEST
            ############

            #Number of reps of the experiments
            reps = 3
            #init
            demand_test = []
            #initialization of the initial inventory
            instance.inventory = (initialShare * mean_demand).copy()
            #
            #initialization of the dictionary that will contains the results for each model
            results = dict.fromkeys(methods) #list of testes policies

            #out_of_sample sampling
            for i in range(reps):
                # horizon demand
                demand_test.append(sam.sample(horizon))

            for k in methods:
                #Inner initialization of the performance metrics per methodology
                results[k] = dict.fromkeys(['cumProfit','inventory','production','lostSales','profits','time'])
                results[k]['cumProfit'] = np.zeros([reps,horizon])
                results[k]['inventory'] = np.zeros([reps,horizon+1])
                instance.inventory = (initialShare * mean_demand).copy() 
                results[k]['inventory'][:,0] = sum (instance.inventory)
                results[k]['production'] = np.zeros([reps,horizon])
                results[k]['lostSales'] = np.zeros([reps,horizon ])
                results[k]['profits'] = []
                results[k]['time'] = []
                for i in range(reps): 
                    demand = demand_test[i]
                    #The initial inventory must be copied. Otherwise, the pointer would consider the final inventory of 
                    #one single reps to start the other one.
                    
                    instance.inventory = (initialShare * mean_demand).copy()
                    #inventory must be reinitialized
                    env = SimplePlant(instance, demand, seasonality=12)
            
                    #Here the sequential environment starts
                    done = False
                    obs = env.reset()
                    #agent setting
                    if k == 'TS_noSeas':
                        stoch_agent = TwoStageAgent(env, AtoRP_multi(**settings), demand_known, full=True)
                    elif k == 'FOSVA':
                        stoch_agent = TwoStageAgent(env, AtoRP_approx_comp(**settings), demand_known) #cave is a twoStageAgent
                    elif k == 'MS' or k == 'MS3_2' or k == 'MS3_3' or k == 'MS3_4' or k == 'MP' or k == 'MP_2' or k == 'MP_3' or k == 'MP_4' or k == 'MS3' or k == 'MS3+' or k == 'TS': #multiStages
                        if k == 'MP_4':
                            ato_setting["branching_factors"] = [ye, 1, 1, 1]
                        elif k == 'MP_3':
                            ato_setting["branching_factors"] = [ye, 1, 1]
                        elif k == 'MP_2':
                            ato_setting["branching_factors"] = [ye, 1]
                        elif k == 'MS3':
                            ato_setting["branching_factors"] = [ye, ye]
                        elif k == 'MS3_4':
                            ato_setting["branching_factors"] = [ye, ye, 1, 1]
                        elif k == 'MS3_3':
                            ato_setting["branching_factors"] = [ye, ye, 1]
                        elif k == 'TS':
                            ato_setting["branching_factors"] = [ye]
                        else:
                            raise ValueError('Method not available')
                        stoch_agent = MultiStageAgent(env, AtoRPMultiStage(**ato_setting), demand_known)
                    elif k == 'PI':
                        stoch_agent = PerfectInfoAgent(env, AtoPI(**ato_setting), demand)
                    else:
                        raise ValueError('Method not available')

                    #Initialization of the results of the single rep 
                    cumulative_profit = []
                    profit = [] 
                    times = []
                    demand = [] #currently not employed in any plot or stat
                    lost_sales = []
                    holding_costs = [] #currently not employed in any plot or stat
                    production_costs = []
                    total_inventory = []
                    #simulation
                    while not done:
                        #time
                        start = time.time()
                        #action computation
                        action, sold = stoch_agent.get_action(obs)
                        ######### Dynamic step with observation of the state of the system
                        obs, reward, done, info = env.step(action, sold)
                        #time
                        end = time.time()
                        comp_time = end - start
                        #write the results of the single time step
                        profit.append(info['profit'])
                        times.append(comp_time)
                        demand.append(info['total_demand'])
                        lost_sales.append(info['lost_sales']) 
                        holding_costs.append(info['holding_costs']) 
                        production_costs.append(info['production_costs']) 
                        total_inventory.append(info['total_inventory']) 
                        cumulative_profit.append(sum(profit))
                    ##### write the result of the entire rep
                    results[k]['profits'].append(profit)
                    results[k]['time'].append(times)
                    results[k]['cumProfit'][i,:] = cumulative_profit
                    print('iteration',i,'finalCumProfit method: ',k,' = ',cumulative_profit[-1],' avgTime: ',np.mean(times))
                    results[k]['inventory'][i,1:] = total_inventory
                    results[k]['production'][i,:] = production_costs
                    results[k]['lostSales'][i,:] = lost_sales

            ##############
            ###PLOTS 
            ##############


            #Plot them all
            #printMultiHorizon(results, horizon)
            #Plot only a subset 
            printMultiHorizon(results, horizon, listToPlot = ['TS','MP_4','FOSVA'])
