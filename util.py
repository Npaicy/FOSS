from config import Config
config = Config()
def min_swap_steps(x, y):
    if len(x) != len(y):
        return -1  # 序列长度不一致，无法进行交换操作

    n = len(x)
    visited = [False] * n  # 标记已访问的元素
    swaps = 0

    for i in range(n):
        if x[i] != y[i] and not visited[i]:
            j = i
            cycle_size = 0

            while not visited[j]:
                visited[j] = True
                j = x.index(y[j])  # 找到y[j]在x中的索引
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
    swap = min_swap_steps(base['join order'],modify['join order'])
    diff = diff_steps(base['join operator'],modify['join operator'])
    return swap + diff
def get_label(ref,cur):
    ratio = (ref - cur) / ref
    if  ratio >= config.alpha and ratio < config.beta:
        label = 1
    elif ratio >= config.beta:
        label = 2
    elif ratio < config.alpha:
        label = 0
    return label