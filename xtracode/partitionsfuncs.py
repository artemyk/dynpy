def partitions(set_):
    if not set_:
        yield []
        return
    for i in xrange(2**len(set_)/2):
        parts = [set(), set()]
        for item in set_:
            parts[i&1].add(item)
            i >>= 1
        for b in partitions(parts[1]):
            yield [parts[0]]+b
            
def prettyPartition(partition, node_labels=None): 
    s ="/".join([",".join(node_labels[node] for node in module) if node_labels is not None else "".join(str(node+1) for node in module) for module in partition])
    return s
    # return "/".join(sorted("".join(map(str,[(node_labels[n]  if node_labels is not None else n+1) for n in nodes])) for nodes in m))
    
def prettyPartition(partition, node_labels=None): 
    s ="/".join([",".join(node_labels[node] for node in module) if node_labels is not None else "".join(str(node+1) for node in module) for module in partition])
    return s
    # return "/".join(sorted("".join(map(str,[(node_labels[n]  if node_labels is not None else n+1) for n in nodes])) for nodes in m))
