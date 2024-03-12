from collections import defaultdict
import random
import torch as t

class VectorWeightedDAG:
    """
    A class for DAGs with edge and node weights and no parallel edges.
    The node weights are vector-like objects and the edge weights are matrix-like.
    """

    # nodes : dict of nodes 
    # edges : a dict of the form {node1: {node2: weight, ...}, ...}
    # we assume no parallel edges
    # if weight is None, treat the edge weight as unknown
    def __init__(self, nodes=None, edges=None):
        if nodes is None:
            nodes = {}
        if edges is None:
            edges = {}
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
        del self._edges[node]
        for nd in self.get_parents(node):
            del self._edges[nd][node]
    
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
    
    def random(n, p=0.5, dim=10):
        """
        Returns a random DAG with n nodes where each edge is present with probability p and has a random weight in weight_range
        """
        dag = VectorWeightedDAG()
        nodes = list(range(n))
        for node in nodes:
            dag.add_node(node, t.randn(dim))

        # add random edges
        random.shuffle(nodes)
        for idx, nu in enumerate(nodes): # upstream node
            for nd in nodes[idx+1:]:
                if random.random() < p:
                    dag.add_edge(nu, nd, t.randn(dim, dim))
        return dag
    
    def copy(self):
        out = VectorWeightedDAG()
        out._nodes = self._nodes.copy()
        out._edges = self._edges.copy()
        return out

    
def deduce_edge_weights(dag, oomphs):
    """
    Define the oomph of a path in a DAG as the product of the edge weights in the path.
    Given a DAG and specification of, for each pair of nodes, the sum of the oomphs of all paths between them,
    compute the weights of all edges.

    dag : a WeightedDAG
    oomphs : a dict of the form {(node1, node2): oomph, ...} where node2 should be downstream of node1
    """
    oomphs_explained = defaultdict(lambda:0)
    topo_order = dag.topological_sort()
    for idxd, nd in enumerate(topo_order):
        for idxu, nu in reversed(list(enumerate(topo_order[:idxd]))):
            if (nu, nd) in dag.edges:
                dag.add_edge(nu, nd, oomphs[(nu, nd)] - oomphs_explained[(nu, nd)])
                oomphs_explained[(nu, nd)] = oomphs[(nu, nd)]
                for nuu in topo_order[:idxu]:
                    if (nuu, nu) in oomphs:
                        oomphs_explained[(nuu, nd)] += dag.edge_weight((nu, nd)) @ oomphs_explained[(nuu, nu)]
    return dag


if __name__ == "__main__":

    dag = VectorWeightedDAG(
        nodes= {i : None for i in range(5)},
        edges = {
            0 : {1 : t.tensor([[1.]]), 2 : t.tensor([[2.]])},
            1 : {2 : t.tensor([[3.]])},
            2 : {3 : t.tensor([[4.]]), 4 : t.tensor([[5.]])},
            3 : {4 : t.tensor([[6.]])},
        }
    )
    topo_order = dag.topological_sort()

    oomphs = defaultdict(float)
    for idxu, nu in enumerate(topo_order):
        for idxd, nd in enumerate(topo_order[idxu+1:]):
            if dag.has_edge((nu, nd)):
                oomphs[(nu, nd)] += dag.edge_weight((nu, nd))
            for nm in topo_order[idxu:idxu+idxd+1]:
                if dag.has_edge((nm, nd)) and (nu, nm) in oomphs:
                    oomphs[(nu, nd)] += dag.edge_weight((nm, nd)) @ oomphs[(nu, nm)]

    cleared_dag = VectorWeightedDAG(
        nodes=dag._nodes,
        edges={
            n : {m : None for m in dag.get_children(n)} for n in dag.nodes
        }
    )
    deduced_dag = deduce_edge_weights(cleared_dag, oomphs)
    for n, m in dag.edges:
        if not deduced_dag.has_edge((n, m)):
            raise ValueError(f"Edge {(n, m)} is missing in deduced_dag")
        if dag.edge_weight((n, m)) != deduced_dag.edge_weight((n, m)):
            raise ValueError(f"Edge {(n, m)} has weight {dag.edge_weight((n, m))} in dag but {deduced_dag.edge_weight((n, m))} in deduced_dag")
    for n, m in deduced_dag.edges:
        if not dag.has_edge((n, m)):
            raise ValueError(f"Edge {(n, m)} is missing in dag")
        if dag.edge_weight((n, m)) != deduced_dag.edge_weight((n, m)):
            raise ValueError(f"Edge {(n, m)} has weight {dag.edge_weight((n, m))} in dag but {deduced_dag.edge_weight((n, m))} in deduced_dag")

    # test random dag
    dag = VectorWeightedDAG.random(10, p=0.1)
    topo_order = dag.topological_sort()

    # check that topo_order is a valid topological order
    for idxu, nu in enumerate(topo_order):
        for nd in topo_order[idxu+1:]:
            if dag.has_edge((nd, nu)):
                raise ValueError(f"Invalid topological order: {nu} is downstream of {nd}")


    # check that each node is a parent of its children
    for node in dag.nodes:
        for child in dag.get_children(node):
            if node not in dag.get_parents(child):
                raise ValueError(f"Node {node} is not a parent of {child}")

    oomphs = defaultdict(float)
    for idxu, nu in enumerate(topo_order):
        for idxd, nd in enumerate(topo_order[idxu+1:]):
            if dag.has_edge((nu, nd)):
                oomphs[(nu, nd)] += dag.edge_weight((nu, nd))
            for nm in topo_order[idxu:idxu+idxd+1]:
                if dag.has_edge((nm, nd)) and (nu, nm) in oomphs:
                    oomphs[(nu, nd)] += dag.edge_weight((nm, nd)) @ oomphs[(nu, nm)]

    cleared_dag = VectorWeightedDAG(
        nodes=dag._nodes,
        edges={
            n : {m : None for m in dag.get_children(n)} for n in dag.nodes
        }
    )
    deduced_dag = deduce_edge_weights(cleared_dag, oomphs)
    # check that computed edge weights are correct
    for n, m in dag.edges:
        if not deduced_dag.has_edge((n, m)):
            raise ValueError(f"Edge {(n, m)} is missing in deduced_dag")
        if ((dag.edge_weight((n, m)) - deduced_dag.edge_weight((n, m))).abs() > 1e-3).any():
            print(dag._edges)
            raise ValueError(f"Edge {(n, m)} has weight {dag.edge_weight((n, m))} in dag but {deduced_dag.edge_weight((n, m))} in deduced_dag")
    for n, m in deduced_dag.edges:
        if not dag.has_edge((n, m)):
            raise ValueError(f"Edge {(n, m)} is missing in dag")
        if ((dag.edge_weight((n, m)) - deduced_dag.edge_weight((n, m))).abs() > 1e-3).any():
            raise ValueError(f"Edge {(n, m)} has weight {dag.edge_weight((n, m))} in dag but {deduced_dag.edge_weight((n, m))} in deduced_dag")
    # check that the deduced oomphs to the root are correct
    # root = topo_order[-1]
    # for n in topo_order[:-1]:
    #     oomphs_out = []
    #     for m in dag.get_children(n):
    #         if m == root:
    #             oomphs_out.append(deduced_dag.edge_weight((n, m)))
    #         else:
    #             oomphs_out.append(deduced_dag.edge_weight((n, m)) * oomphs[(m, root)])
    #     if abs(sum(oomphs_out) - oomphs[(n, root)]) > 1e-4:
    #         raise ValueError(f"Sum of oomphs out of {n} is {sum(oomphs_out)} but should be {oomphs[(n, root)]}")
                


                


    
    
    