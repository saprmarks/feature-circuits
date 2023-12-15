# %%
def get_submodule_by_path(model, path):
    parts = path.split('.')
    submodule = model
    for part in parts:
        submodule = getattr(submodule, part)
    return submodule

class Node:
    def __init__(self, model, node):
        if type(node) == str:
            submodule_name, feature = Node._from_str(node)
            submodule = get_submodule_by_path(model, submodule_name)
        elif type(node) == tuple:
            submodule, feature = node
        else:
            submodule, feat = node, None
        self.submodule = submodule
        self.feature = feature
        

    def _from_str(str):
        if '/' in str:
            submodule_name, feature = str.split('/')
            if feature == '*':
                feature = None
            else:
                feature = int(feature)
        else:
            submodule_name, feature = str, None
        return submodule_name, feature


class Subnetwork:
    # nodes is a list of strings specifying features
    def __init__(self, nodes=[]):
        self.whitelist = {}
        for node in nodes:
            self.add_node(node)

    def add_node(self, node):
        submodule, feature = node.submodule, node.feature
        if submodule not in self.whitelist:
            self.whitelist[submodule] = set()
        if feature is None:
            self.whitelist[submodule] = None
        else:
            self.whitelist[submodule].add(feature)
            
    def __contains__(self, node):
        if node.submodule not in self.whitelist:
            return False
        if self.whitelist[node.submodule] is None:
            return True
        return node.feature in self.whitelist[node.submodule]

# %%
