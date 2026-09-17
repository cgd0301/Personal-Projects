import rubik

def shortest_path(start, end):
    """
    Using 2-way BFS, finds the shortest path from start_position to
    end_position. Returns a list of moves. 

    You can use the rubik.quarter_twists move set.
    Each move can be applied using rubik.perm_apply
    """
    level_s = {start: 0}
    level_e = {end: 0}
    parent_s = {start: (None, None)}
    parent_e = {end: (None, None)}
    frontier_s = [start]
    frontier_e = [end]
    level = 1
    while not set(level_s.keys()) & set(level_e.keys()) and (len(frontier_e) and len(frontier_s)):
        next_s = []
        for u in frontier_s:
            for v in adj(u):
                if v[0] not in level_s.keys():
                    level_s[v[0]] = level
                    parent_s[v[0]] = (u, v[1])
                    next_s.append(v[0])
        frontier_s = next_s

        next_e = []
        for u in frontier_e:
            for v in adj(u):
                if v[0] not in level_e.keys():
                    level_e[v[0]] = level
                    parent_e[v[0]] = (u, v[1])
                    next_e.append(v[0])
        frontier_e = next_e

        level += 1

    cross = (set(level_s.keys()) & set(level_e.keys())).pop()
    tmp_ptr_s = cross
    tmp_ptr_e = cross
    shortest_path = []

    while parent_s[tmp_ptr_s][0] is not None:
        shortest_path.insert(0, parent_s[tmp_ptr_s][1])
        tmp_ptr_s = parent_s[tmp_ptr_s][0]

    while parent_e[tmp_ptr_e][0] is not None:
        shortest_path.append(rubik.perm_inverse(parent_e[tmp_ptr_e][1]))
        tmp_ptr_e = parent_e[tmp_ptr_e][0]

    print(shortest_path)
    return shortest_path

def adj(u):
    adj = []
    for twist in rubik.quarter_twists:
        adj.append((rubik.perm_apply(twist, u), twist))
    return adj
