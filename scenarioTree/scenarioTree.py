# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

def prod(val):  
    res = 1 
    for ele in val:  
        res *= ele  
    return res   


class ScenarioTree(nx.DiGraph):
    def __init__(self, name, branching_factors, dim_observations, initial_value, stoch_model):
        nx.DiGraph.__init__(self)
        self.starting_node = 0
        self.dim_observations = dim_observations
        self.stoch_model = stoch_model
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

        self.n_scenarios = prod(self.branching_factors)

        count = 1
        last_added_nodes = [self.starting_node]
        n_nodes_per_level = 1
        for i in range(self.depth):
            next_level = []
            n_nodes_per_level *= self.branching_factors[i]
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
        return self.leaves

    def get_history_node(self, n):  
        ris = self.nodes[n]['obs'].reshape((1, self.dim_observations))
        if n == 0:
            return ris
        while n != self.starting_node:
            n = list(self.predecessors(n))[0]
            ris = np.vstack(
                (self.nodes[n]['obs'].reshape((1, self.dim_observations)), ris)
            )
        return ris

    def set_scenario_chain(self, simulation_data):
        """
        Set the scenario simulation_data in all the nodes.
        Use it only in the perfect information case
        """
        for t in range(len(self.branching_factors)):
            self.nodes[t]['obs'] = simulation_data[:, t]

    def print_matrix_form_on_file(self, name_details=""):
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
        for leaf in self.leaves:    
            y = self.get_history_node(leaf)
            for obs in range(self.dim_observations):
                plt.plot(y[obs, :], label=f'obs {obs}')
            plt.legend()
            plt.ylabel(f'History scenario {leaf}')
            plt.savefig(f'./results/scenario_{leaf}.png')
            plt.close()
