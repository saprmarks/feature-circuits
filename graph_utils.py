from collections import defaultdict
from copy import deepcopy
import random

class WeightedDAG:
    """
    A class for DAGs with edge and node weights and no parallel edges
    """

    # nodes : a list of nodes
    # edges : a dict of the form {node1: {node2: weight, ...}, ...}
    # we assume no parallel edges
    # if weight is None, treat the edge weight as unknown
    def __init__(self, nodes={}, edges={}):
        self._nodes = nodes
        self._edges = defaultdict(dict, edges)
        for node in edges.keys():
            if node not in nodes:
                raise ValueError(f"Node {node} in edges but not in nodes")
            
    @property
    def nodes(self):
        return list(self._nodes.keys())
    
    @property
    def edges(self):
        return [(n1, n2) for n1 in self._edges for n2 in self._edges[n1]]
    
    def has_node(self, node):
        return node in self._nodes
    
    def has_edge(self, e):
        node1, node2 = e
        return node2 in self._edges[node1]
    
    def node_weight(self, node):
        return self._nodes[node]
    
    def edge_weight(self, e):
        node1, node2 = e
        return self._edges[node1][node2]

    def add_node(self, node, weight=None):
        self._nodes[node] = weight

    def remove_node(self, node):
        del self._nodes[node]
    
    # add an edge from node1 to node2 with weight, or update the weight if the edge already exists
    def add_edge(self, node1, node2, weight=None):
        if node1 not in self._nodes:
            raise ValueError(f"Node {node1} not in nodes")
        if node2 not in self._nodes:
            raise ValueError(f"Node {node2} not in nodes")
        self._edges[node1][node2] = weight

    def remove_edge(self, e):
        node1, node2 = e
        del self._edges[node1][node2]
    
    def get_children(self, node):
        return list(self._edges[node].keys())

    def get_parents(self, node):
        return [nd for nd in self.nodes if node in self.get_children(nd)]
    
    def topological_sort(self, reverse=False):
        """
        Returns a topological sort of the nodes in the graph
        """
        visited = set()
        stack = []
        def dfs(node):
            visited.add(node)
            for neighbor in self.get_children(node):
                if neighbor not in visited:
                    dfs(neighbor)
            stack.append(node)
        for node in self.nodes:
            if node not in visited:
                dfs(node)
        return stack if reverse else stack[::-1]
    
    def random(n, p=0.5, weight_range=(0,1)):
        """
        Returns a random DAG with n nodes where each edge is present with probability p and has a random weight in weight_range
        """
        dag = WeightedDAG()
        nodes = list(range(n))
        for node in nodes:
            dag.add_node(node, random.uniform(*weight_range))

        # add random edges
        random.shuffle(nodes)
        for idx, nu in enumerate(nodes): # upstream node
            for nd in nodes[idx+1:]:
                if random.random() < p:
                    dag.add_edge(nu, nd, random.uniform(*weight_range))
        return dag

    
def deduce_edge_weights(dag, oomphs):
    """
    Define the oomph of a path in a DAG as the product of the edge weights in the path.
    Given a DAG and specification of, for each pair of nodes, the sum of the oomphs of all paths between them,
    compute the weights of all edges.

    dag : a WeightedDAG
    oomphs : a dict of the form {(node1, node2): oomph, ...} where node1 should be downstream of node2
    """
    oomphs_explained = {k : 0 for k in oomphs.keys()}

    reverse_topo_order = dag.topological_sort(reverse=True)
    for idxu, nu in enumerate(reverse_topo_order): # upstream node
        for nd in reversed(reverse_topo_order[:idxu]): # downstream node
            if (nd, nu) not in oomphs:
                continue
            weight = oomphs[(nd, nu)] - oomphs_explained[(nd, nu)]
            dag.add_edge(nu, nd, weight)
            for ndd in dag.get_children(nd):
                oomphs_explained[(ndd, nu)] += weight * dag.edge_weight((nd, ndd))
    return dag
            


if __name__ == "__main__":

    random_dag = WeightedDAG.random(100, p=0.5, weight_range=(0, 1))
    topo_order = random_dag.topological_sort()
    # check that topo_order is a valid topological order
    for idxu, nu in enumerate(topo_order):
        for nd in topo_order[idxu+1:]:
            if random_dag.has_edge((nd, nu)):
                raise ValueError(f"Invalid topological order: {nu} is downstream of {nd}")


    # check that each node is a parent of its children
    for node in random_dag.nodes:
        for child in random_dag.get_children(node):
            if node not in random_dag.get_parents(child):
                raise ValueError(f"Node {node} is not a parent of {child}")

    # compute oomphs for random_dag
    oomphs = defaultdict(int)
    for idxu, nu in enumerate(topo_order):
        for idxd, nd in enumerate(topo_order[idxu+1:]):
            if random_dag.has_edge((nu, nd)):
                oomphs[(nu, nd)] += random_dag.edge_weight((nu, nd))
            for nm in topo_order[idxu:idxu+idxd+1]:
                if random_dag.has_edge((nm, nd)) and (nu, nm) in oomphs:
                    oomphs[(nu, nd)] += oomphs[(nu, nm)] * random_dag.edge_weight((nm, nd))
    deduced_dag = deduce_edge_weights(random_dag, oomphs)
    for n1 in random_dag.nodes:
        for n2 in random_dag.nodes:
            e = (n1, n2)
            if not random_dag.has_edge(e):
                continue
            if not random_dag.has_edge(e):
                raise ValueError(f"Edge {e} is missing in deduced_dag")
            if random_dag.edge_weight(e) != deduced_dag.edge_weight(e):
                raise ValueError(f"Edge {e} has weight {random_dag.edge_weight(e)} in random_dag but {deduced_dag.edge_weight(e)} in deduced_dag")
                


                


    
    
    