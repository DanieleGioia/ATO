# ATO

This library simulates and optimizes **Two-Stage** and **Multi-Stage** policies for *assemble-to-order* problem. Specifically, this strategy allows the manufacture of the components under demand uncertainty, while assembling end items only after demand is realized.

The code comprises several classes and two main(s) as examples. Those are supported by two different articles that we reccomend to cite in case you use the library.

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

Each instance of the ATO problem comprises:

- The available capacity for each machine.
- A set of technological and commercial parameters (costs and prices of components and end items).
- The Gozinto factors (amount of components required for assembling the end item).

In case of a random generation of an instance (by means of the **InstanceRandom** class), all these characteristics of the problems are tuned by the **instance_Params** json file in the etc folder.

Here it follows the role of the available parameters:

- "seed" $\rightarrow$ Initialization of the pseudorandom number generator.
- "n_items" $\rightarrow$ Number of end items.
- "n_components" $\rightarrow$ Number of components.
- "n_machines" $\rightarrow$ Number of machines.
- "profit_margin_{low, medium, high}" $\rightarrow$ Minimum and maximum of the uniform distribution domain employed to generate the margins (thus the price) of the end items.
- "perc_{low,medium}_margin_item" $\rightarrow$  Proportion of the low,medium and consequentially high margin classes of end items.
- "processing_time_interval" $\rightarrow$ Domain of the random variable that generates the required machines capacity.
- "dict_gozinto" $\rightarrow$ Characteristics of the Gozinto matrix, where each family has a fixed number of end items ("n_items_per_family")and, within a family, these items have a number of requested components ("n_components_per_family"), either individual or shared with other members ("n_common_components_per_family"). Moreover, we also introduce degenerate families, composed by one single item and we call them outcast ("n_outcast_items"). They either selects a component in their bill of materials or not with probability ("p_outcast_component").
- "component_cost" $\rightarrow$ Domain of the random variable that generates the cost of the components.
- "tightness" $\rightarrow$ It defines the machine availability w.r.t. the production rquired if the average demand is cosidered. (E.g., 0.8 means that we have enough machine availability to produce 80% of the average demand per per time step).
- "initial_inventory" $\rightarrow$ Initial inventory of the components. Note that the dimension must be equal to "n_components".
- "lost_sales" and "holding_costs" $\rightarrow$ Percentage that defines the lost sales and the holding costs by multiplying the end items price and the components cost.

All the Instance classes inherit a Gozinto matrix print function that associates the number of required components for each end item. We provide an example hereafter:

![plot](./etc/gozmatrix.png)

The instance can also be read from a json file through the **InstanceRead** class.

## Sampler & scenarios

Once an instance of the problem is formulated, the main source of uncertainty, according to our assumption on the ATO problem, is the demand for each end item. Specifically, it is uncertain and possibly subject to seasonality. 

The **Sampler** class defines the mandatory methods for the simulation of the demand. Generally speaking, given a number of scenarios, the sampler returns a random demand per end item per scenario, following different sampling methodologies.\
We list the available sampler and their characteristics hereafter.

| Sampler | Characteristics |
| ------------- |:-------------:|
|GaussianSampler |  The Gaussian sampler generates the demand according to **independent** and identically distributed Gaussian random variables with mean ("mu") and standard deviation ("sigma") specificated in "./etc/sampler_Params".
|BiGaussianSampler| The Gaussian sampler is available also in a bi-modal variant, where the additional mean ("mu2"), standard deviation ("sigma2") and mixing factor ("p1") have to be specified.
|HierarchicalSampler| This sampler consider a process composed by two nested steps, such that a family-correlation is generated. Firstly, we independently sample the aggregated demand for the entire family, then the overall demand per family is split among the items belonging to the family according to weights randomly sampled from a Dirichlet distribution.
|MultiStageSampler| It adapts the other samplers to a multistage setting. It generates as scenarios a fixed horizon number of sampled demand per end item. It is possible to set seasonality in a multiplicative ("multiplicativeSeas") or additive ("additiveSeas") way w.r.t. the mean and standard deviation of the employed distribution.

## Solver

Several solvers are available. They solve different problems in terms of both objective functions and constraints. However, all of them currently rely on [**Gurobi**](https://www.gurobi.com/). Extensions with other software are possible.\ Ato.py summarizes what a generic solver/problem should contain in its methods.\
AtoG.py works as interface (super-class) of the assembly-to-order solvers in Gurobi. Here the population (Gurobi model construction) and the solution process (that can relies on different algorithms) are separated.

Here it follows a table summing up the principal characteristics of the available solvers. All of them but atoEV and atoRPMultiStage are Two-stage environments, the latter works with several kind of scenario trees. The classes inherits from AtoG.py and defines how to populate the model thanks to polymorphism.

<dev style="color:red"> **For a detailed formulation of the available models please refer to the cited papers.** </dev>

| Solver | Characteristics |
| ------------- |:-------------:|
| atoCVaR  | It minimizes the $\text{CVaR}_{\alpha} $ following the $\alpha$ = **CVaR_alpha** selected in './etc/ato_Params', while providing a minimum expected net profit according to the **CVaR_expected_profit**.
| atoCVaRProfit   | It maximizes the expected net profit while bounding the $\text{CVaR}_{\alpha} $ according to the **atoProfitCVaR_limit** value in './etc/ato_Params' with an $\alpha$ level of *atoProfitCVaR_alpha*.
| atoEV  | It maximizes the expected net profit of the problem without a recourse function, thus operating in one single stage with averaged constraints.
| atoPI  | In this version of the ATO problem, we assume we have Perfect Information (PI) of the demand, thus produce optimally. This allows the calculation of the EVPI (Expected Value of Perfect Information).
| atoRP  | Standard Two-Stage stochastic LP model with recourse of the ATO problem, treated with the well-konwn Sampling Average Approximation (SAA).
| atoRPMultiStage  | This model represents the demand uncertainty by means of a scenario tree with personalizable length and branching factors trough the **branching_factors** vector in './etc/ato_Params'. It supports seasonality throughout the scenario and can rely on multiple nodes per time-steps as well as average approximations. An extended discussion of the model is presented in our paper "**Rolling horizon policies for multi-stage stochastic assemble-to-order problems**".
| atoRP_approx_comp  | This class contains two sub-classes. On the one hand, **AtoRP_approx_comp_v** serves to approximate the value of the initial inventory according to a first-order analysis on a Two-Stage setting. On the other hand, **AtoRP_approx_comp** applies the approximate value of the inventory following a linear piecewise value function defined by its breakpoints and slopes.

## FOSVA

## Agents & Envs
