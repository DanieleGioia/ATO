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
# "Risk-averse Approaches for a Two-Stage Assembly-to-Order Problem.",
# by Edoardo Fadda and Daniele Giovanni Gioia and Paolo Brandimarte.

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

sam = BiGaussianSampler(smpl_setting)

#instance generation
#the sampler is required to tune the machine availability w.r.t. the tightness
instance = InstanceRandom(sim_setting,sam)

demand = sam.sample(
    n_scenarios=10
)

# Solution Computation
prb = AtoRP()
of_ato, sol_ato, comp_time_ato = prb.solve(instance, demand)
print(f"sol_ato: {of_ato}")

prb = AtoCVaR(**ato_setting)
of_atoCVaR, sol_atoCVaR, comp_time_atoCVaR = prb.solve(instance, demand)
print(f"sol_atoCVaR: {of_atoCVaR}")

