#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from instances import *
from sampler import *
from solver.solverGurobi import *
from envs import *
import json
#agent
from agents import *
# FOSVA
from FOSVA import *
#Scenario Reducer
from scenarioReducer import *
#utils
from utils.utils import *
import time

np.set_printoptions(precision=4) 
np.set_printoptions(suppress=True)


#list of methods
methods = ['MP','MS3','MS3+','MS_binary','FOSVA','WS','TS']
#list of samplers
samplers = ['hierBi','Bi','Gaus']
#list of tight
tightness = [0.8,1.2]
#horizon of the simulation
horizon = 12
#fosva approximation rounds and gradient-step
n_it_fosva = 3
step_derivative = 5

for sampler in samplers:
    for tight in tightness:
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
        #sampler
        if sampler == 'hierBi':
            hierSam = HierarchicalSampler(smpl_setting,sim_setting['dict_gozinto'],BiGaussianSampler(smpl_setting))
        elif sampler == 'Bi':
            hierSam = BiGaussianSampler(smpl_setting)
        elif sampler == 'Gaus':
            #mean and std adj for comparability
            smpl_setting['mu'] = smpl_setting['mu']*smpl_setting['p1'] + (1 - smpl_setting['p1'])*smpl_setting['mu2']
            smpl_setting['sigma'] = ((smpl_setting['sigma']*smpl_setting['p1'])**2 + ((1 - smpl_setting['p1'])*smpl_setting['sigma2'])**2)**(1/2)
            hierSam = GaussianSampler(smpl_setting)
        else:
            raise ValueError('sampler option not managed')

        sam = MultiStageSampler(smpl_setting,hierSam)

        #instance
        instance = InstanceRandom(sim_setting,sam)
        # instance.plotGozinto()

        ####training demand
        #number of observed scenarios
        nObs = 120 #10 years
        #sampling
        demand_known = sam.sample(nObs)
        #problem init
        settings = {}

        ###FOSVA

        #initial inventory
        initialShare = 0.5 # % of the expected value per component
        mean_demand = (np.mean(demand_known,axis=1) @ instance.gozinto).copy()
        instance.inventory = initialShare * mean_demand

        #Compute cave
        #prblem that computes the value of a given inventory
        prb = AtoRP_approx_comp_v()

        fosva_res = run_multifosva_ato(
            instance,
            prb,
            demand_known,
            step_derivative=step_derivative,
            eps_p_fun=lambda i:100.0/(2**np.floor(i/20)) ,
            eps_m_fun=lambda i: 100.0/(2**np.floor(i/20)) ,
            alpha_fun=lambda i: 10.0/(10.0 + i),
            n_iterations_fosva=n_it_fosva
        )

        for i in range(len(fosva_res)):
            fosva_res[i]['u'] = np.array(fosva_res[i]['u'])
            fosva_res[i]['v'] = np.array(fosva_res[i]['v'])
        #set cave
        settings['fosva_res'] = fosva_res

        ############TEST      
        reps = 2
        demand_test = []
        instance.inventory = (initialShare * mean_demand).copy()
        #

        #
        results = dict.fromkeys(methods) #list of testes policies 

        #out_of_sample
        for i in range(reps):
            # horizon demand
            demand_test.append(sam.sample(horizon))

        for k in methods:
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
                instance.inventory = (initialShare * mean_demand).copy()
                #inventory must be reinitialized
                env = SimplePlant(instance, demand, seasonality=12)
            
                done = False
                obs = env.reset()
                #agent setting
                if k == 'FOSVA':
                    stoch_agent = TwoStageAgent(env, AtoRP_approx_comp(**settings), demand_known) #cave is a twoStageAgent
                elif k == 'MP' or k == 'MS3' or k == 'MS3+' or k == 'MS_binary': #multiStages
                    if k == 'MP':
                        ato_setting["branching_factors"] = [10, 1, 1, 1, 1, 1, 1]
                    elif k == 'MS3':
                        ato_setting["branching_factors"] = [10, 10]
                    elif k == 'MS3+':
                        ato_setting["branching_factors"] = [10, 10, 1, 1, 1, 1, 1]
                    elif k == 'MS_binary':
                        ato_setting["branching_factors"] = [2, 2, 2, 2, 2, 2, 2]
                    else:
                        pass
                    stoch_agent = MultiStageAgent(env, AtoRPMultiStage(**ato_setting), demand_known)
                elif k == 'WS': #waitAndSee
                    prb = AtoPI() #no stochastic agent
                elif k == 'TS': #myopic twoStages
                    stoch_agent = TwoStageAgent(env, AtoRP(**ato_setting), demand_known)
                else:
                    raise ValueError('Method not available')
                #results
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
                    if k != 'WS':
                        action = stoch_agent.get_action(obs)
                    else:
                        _, action, _ = prb.solve(env.instance,env.demand) # solves w.r.t. to the future
                    #########    
                    obs, reward, done, info = env.step(action)
                    #time
                    end = time.time()
                    comp_time = end - start
                    profit.append(info['profit'])
                    times.append(comp_time)
                    demand.append(info['total_demand'])
                    lost_sales.append(info['lost_sales']) 
                    holding_costs.append(info['holding_costs']) 
                    production_costs.append(info['production_costs']) 
                    total_inventory.append(info['total_inventory']) 
                    cumulative_profit.append(sum(profit))
                #####
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
printMultiHorizon(results.item(), horizon, listToPlot = ['MS3','MP'])
