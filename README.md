# ATO

This library simulates and optimizes **Two-Stage** and **Multi-Stage** policies for *assemble-to-order* problem. Specifically, this strategy allows the manufacture of the components under demand uncertainty, while assembling end items only after demand is realized.

The code comprises several classes and two main as examples. Those are supported by two different articles that we reccomend to cite in case you use the library.

## Citing Us

Multi-Stage

```TeX
@misc{GioiaPrePrint,
  title={Rolling horizon policies for multi-stage stochastic assemble-to-order problems},
  author={Daniele Giovanni Gioia and Edoardo Fadda and Paolo Brandimarte},
  year = {2022},
  howpublished = {Available at SSRN: aggiungere link}
}
```

Two-Stage

```TeX
@incollection{Fadda2022,
  year = {2022},
  publisher = {Springer International Publishing},
  author = {Edoardo Fadda and Daniele Giovanni Gioia and Paolo Brandimarte},
  title = {Robust Approaches for a Two-Stage Assembly-to-Order Problem},
  booktitle = {{AIRO} Springer Series},
  note = {forthcoming}
}
```

# Code Structure

```bash
|____solver
| |______init__.py
| |____Ato.py
| |____solverGurobi
| | |____atoEV.py
| | |____atoRPMultiStage.py
| | |____atoRP_approx_comp.py
| | |____atoCVaRProfit.py
| | |______init__.py
| | |____atoG.py
| | |____atoCVaR.py
| | |____atoPI.py
| | |____atoRP.py

|____sampler
| |____Hierarchical_Sampl.py
| |______init__.py
| |____MultiStage_Sampl.py
| |____Sampler.py
| |____Gaussian_Sampl.py

|____instances
| |____Instance.py
| |______init__.py
| |____InstanceRandom.py
| |____InstanceRead.py

|____etc
| |____ato_Params.json
| |____instance_Params.json
| |____sampler_Params.json

|____agents
| |____twoStageAgent.py
| |______init__.py
| |______pycache__
| |____atoAgent.py
| |____multiStageAgent.py

|____utils
| |____Tester.py
| |____utils.py

|____requirements
|____main_multistage.py
|____README.md

|____FOSVA
| |____fosva_ato.py
| |____fosva.py
| |______init__.py

|____scenarioReducer
| |____scenario_reducer.py
| |______init__.py
| |____fast_forward_W2.py

|____scenarioTree
| |______init__.py
| |______pycache__
| |____scenarioTree.py

|____envs
| |____simplePlant.py
| |______init__.py

```

## Solver

## Sampler & scenarios

## Instace generation

## FOSVA

## Agents & Envs
