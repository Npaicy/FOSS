from config import Config
config = Config()
def min_swap_steps(x, y):
    if len(x) != len(y):
        return -1  

    n = len(x)
    visited = [False] * n  
    swaps = 0

    for i in range(n):
        if x[i] != y[i] and not visited[i]:
            j = i
            cycle_size = 0

            while not visited[j]:
                visited[j] = True
                j = x.index(y[j])  
                cycle_size += 1

            if cycle_size > 0:
                swaps += cycle_size - 1

    return swaps
def diff_steps(x,y):
    if len(x) != len(y):
        return -1 
    modify = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            modify += 1
    return modify
def min_steps(base, modify):
    swap = min_swap_steps(base['join order'], modify['join order'])
    diff = diff_steps(base['join operator'], modify['join operator'])
    return swap + diff
def get_label(ref,cur):
    ratio = (ref - cur) / ref
    label = 0
    for l, p in enumerate(config.splitpoint):
        if ratio >= p:
            label = config.classNum - l
            break
    return label

def get_median(L1, L2):
    sorted_l1_indices = sorted(range(len(L1)), key=lambda i: L1[i])
    sorted_l1 = [L1[i] for i in sorted_l1_indices]
    sorted_l2 = [L2[i] for i in sorted_l1_indices]
    
    length = len(sorted_l1)
    median_index = length // 2
    
    median_value_l1 = sorted_l1[median_index]
    median_value_l2 = sorted_l2[median_index]
    
    return median_value_l1, median_value_l2
def swap_dict_items(data, key1, key2):
    if key1 not in data or key2 not in data:
        raise KeyError("Index Error")
    items = list(data.items())
    index1 = next(i for i, (k, v) in enumerate(items) if k == key1)
    index2 = next(i for i, (k, v) in enumerate(items) if k == key2)
    items[index1], items[index2] = items[index2], items[index1]
    new_data = dict(items)
    return new_data
