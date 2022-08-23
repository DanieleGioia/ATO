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

## Instace generation

## Sampler & scenarios

## Solver

Several solvers are available. They solve different problems in terms of both objective functions and constraints. However, all of them currently rely on [**Gurobi**](https://www.gurobi.com/). Extensions with other software are possible.\ Ato.py summarizes what a generic solver/problem should contain in its methods.\
AtoG.py works as interface (super-class) of the assembly-to-order solvers in Gurobi. Here the population (Gurobi model construction) and the solution process (that can relies on different algorithms) are separated.

Here it follows a table summing up the principal charateristics of the available solvers. All of them but atoEV and atoRPMultiStage are Two-stage environments, the latter works with several kind of scenario trees. The classes inherits from AtoG.py and defines how to populate the model thanks to polymorphism.

```diff
 !For a detailed formulation of the available models please refer to the cited papers.
 ```

| Solver | Charateristics |
| ------------- |:-------------:|
| atoCVaR  | It minimizes the $\text{CVaR}_{\alpha} $ following the $\alpha$ = **CVaR_alpha** selected in './etc/ato_Params', while providing a minimum expected net profit according to the **CVaR_expected_profit**.
| atoCVaRProfit   | It maximizes the expected net profit while bounding the $\text{CVaR}_{\alpha} $ according to the **atoProfitCVaR_limit** value in './etc/ato_Params' with an $\alpha$ level of *atoProfitCVaR_alpha*.
| atoEV  | It maximizes the expected net profit of the problem without a recourse function, thus operating in one single stage with averaged constraints.
| atoPI  | In this version of the ATO problem, we assume we have Perfect Information (PI) of the demand, thus produce optimally. This allows the calculation of the EVPI (Expected Value of Perfect Information).
| atoRP  | Standard Two-Stage stochastic LP model with recourse of the ATO problem, treated with the well-konwn Sampling Average Approximation (SAA).
| atoRPMultiStage  | This model represents the demand uncertainty by means of a scenario tree with personalizable length and branching factors. It supports seasonality throughout the scenario and can rely on multiple nodes per time-steps as well as average approximations. An extended discussion of the model is presented in our paper "**Rolling horizon policies for multi-stage stochastic assemble-to-order problems**".
| atoRP_approx_comp  | This class contains two sub-classes. On the one hand, **AtoRP_approx_comp_v** serves to approximate the value of the initial inventory according to a first-order analysis on a Two-Stage setting. On the other hand, **AtoRP_approx_comp** applies the approximate value of the inventory following a linear piecewise value function defined by its breakpoints and slopes.

## FOSVA

## Agents & Envs
