# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from scenarioReducer import Scenario_reducer


def prod(val):  
    res = 1 
    for ele in val:  
        res *= ele  
    return res


class ScenarioTree(nx.DiGraph):
    '''
    This class implements a scenario tree.
    It extend the class nx.Digraph.
    '''
    def __init__(self, name: str, branching_factors: list, dim_observations: int, initial_value: np.ndarray , stoch_model: Scenario_reducer):
        """Initialize a graph.

        Args:
            name (str): the name of the tree
            branching_factors (list): it describes the branching factor of the tree e.g. [2, 2, 2] is a binary tree with depth 3
            dim_observations (int): dimension of each observation
            initial_value (np.darray): value ofserved at the root node
            stoch_model (Scenario_reducer): class used to generate new possible realization
        """
        nx.DiGraph.__init__(self)
        self.starting_node = 0
        self.dim_observations = dim_observations
        self.stoch_model = stoch_model
        # Adding root node
        self.add_node(
            self.starting_node,
            obs=initial_value,
            prob=1,
            t=0,
            id=0,
            stage=0
        )
        self.name = name
        self.breadth_first_search = []
        self.depth = len(branching_factors)
        self.branching_factors = branching_factors
        # Computing total number of scenarios
        self.n_scenarios = prod(self.branching_factors)
        count = 1
        last_added_nodes = [self.starting_node]
        n_nodes_per_level = 1
        # Generating other nodes
        for i in range(self.depth):
            next_level = []
            n_nodes_per_level *= self.branching_factors[i]
            # for each parent node add the children
            for parent_node in last_added_nodes:
                demand_reduced, probs_reduced = stoch_model[i].reduce(
                    self.branching_factors[i]
                )
                for j in range(self.branching_factors[i]):
                    id_new_node = count
                    self.add_node(
                        id_new_node,
                        obs=demand_reduced[:,j],
                        prob=self.nodes[parent_node]['prob'] * probs_reduced[j],
                        t=i + 1,
                        id=count,
                        stage=i + 1
                    )
                    self.add_edge(parent_node, id_new_node)
                    next_level.append(id_new_node)
                    count += 1
            last_added_nodes = next_level
            self.n_nodes = count
        self.leaves = last_added_nodes

    def get_leaves(self):
        # Return all the leaves of the tree
        return self.leaves

    def get_history_node(self, n):
        # Given the index of a node, it returns all the observation from it to the root node
        ris = self.nodes[n]['obs'].reshape((1, self.dim_observations))
        # if n is the root node it is simple
        if n == 0:
            return ris
        # otherwise we iterate backward
        while n != self.starting_node:
            n = list(self.predecessors(n))[0]
            ris = np.vstack(
                (self.nodes[n]['obs'].reshape((1, self.dim_observations)), ris)
            )
        return ris

    def set_scenario_chain(self, simulation_data):
        """
        Set the scenario simulation_data.
        Use it only in the perfect information case
        """
        for t in range(len(self.branching_factors)):
            self.nodes[t]['obs'] = simulation_data[:, t]

    def print_matrix_form_on_file(self, name_details=""):
        """It prints the tree in a csv file in the folder results.
        Usefull for debugging.
        """
        f = open(f"./results/tree_matrix_form_{name_details}.csv", "w")
        f.write("leaf, item")
        for ele in range(self.depth + 1):
            f.write(f",t{ele}")
        f.write("\n")
        for leaf in self.leaves:
            for obs in range(self.dim_observations):
                y = self.get_history_node(leaf)
                str_values = ",".join([f"{ele}" for ele in y[obs,:]])
                f.write(f"{leaf},{obs},{str_values}\n")
        f.close()

    def plot(self):
        """It prints on the file path "./results/graph_{self.name}.png" the graph
        """
        pos = graphviz_layout(self, prog="dot")
        nx.draw(
            self, pos,
            with_labels=True, arrows=True
        )
        plt.savefig(f'./results/graph_{self.name}.png')
        plt.close()
    
    def plot_all_scenarios_png(self):
        """ It prints in a png file all the scenarios.
        It is usefull to observe how they differ.
        """
        for leaf in self.leaves:    
            y = self.get_history_node(leaf)
            for obs in range(self.dim_observations):
                plt.plot(y[obs, :], label=f'obs {obs}')
            plt.legend()
            plt.ylabel(f'History scenario {leaf}')
            plt.savefig(f'./results/scenario_{leaf}.png')
            plt.close()
