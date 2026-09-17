Problem 6-2 (a):
    def DFS(G, adj):
        parents = {}
        order = []
        for every v_i in G:
            if v_i is not in parents:
                parents[v_i] = None
                DFS_visit(G, v_i, adj)
                order.append(v_i)
        return order.reverse()

    def DFS_visit(G, v_i, adj):
        for v in adj[v_i]:
            if v is not in parents:
                parents[v] = v_i
                DFS_visit(G, v, adj)
                order.append(v)

Problem 6-2 (b):
    def DFS(G, adj):
        parents = {}
        order = []
        for every v_i in G:
            if v_i is installed:
                continue
            if v_i is not in parents:
                parents[v_i] = None
                DFS_visit(G, v_i, adj)
                order.append(v_i)
        return order.reverse()

    def DFS_visit(G, v_i, adj):
        for v in adj[v_i]:
            if v is installed:
                continue
            if v is not in parents:
                parents[v] = v_i
                DFS_visit(G, v, adj)
                order.append(v)


