import imagematrix

class ResizeableImage(imagematrix.ImageMatrix):
    def best_seam(self):
        energy_map = [[0 for _ in range(self.height)] for _ in range(self.width)]
        path = dict()
        ans = list()

        if self.height == 0: 
            return []
        for i in range(self.width):
            energy_map[i][0] = self.energy(i, 0)
            path[(i, 0)] = None

        for j in range(1, self.height):
            for k in range(self.width):
                arr = [energy_map[k][j - 1]]
                start = k

                if k - 1 >= 0:
                    arr.insert(0, energy_map[k - 1][j - 1])
                    start = k - 1
                if k + 1 < self.width:
                    arr.append(energy_map[k + 1][j - 1])

                min_val = min(arr)
                index = arr.index(min_val)
                energy_map[k][j] = min_val + self.energy(k, j)
                path[(k, j)] = (start + index, j - 1)

        min_path_index = 0
        min_path_val = energy_map[0][self.height - 1]
        for l in range(self.width):
            if energy_map[l][self.height - 1] < min_path_val:
                min_path_val = energy_map[l][self.height - 1]
                min_path_index = l

        start = (min_path_index, self.height - 1)
        while start is not None:
            ans.append(start)
            start = path[start]
        ans.reverse()
        return ans
        


    def remove_best_seam(self):
        self.remove_seam(self.best_seam())

