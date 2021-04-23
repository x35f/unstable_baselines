import math
import numpy as np

class MaxHeap(object):
    def __init__ (self):
        raise NotImplementedError

class SumTree(object):
    def __init__ (self, max_size):
        self.max_size = max_size
        self.tree_depth = math.ceil(math.log2(max_size + 1)) - 1
        self.tree_size = 2**(self.tree_depth+1) - 1
        self.curr = 0
        self.size = 0

        self.value = np.zeros((self.tree_size, ))
        self.data = np.ndarray((self.max_size, ), dtype=object)
    
    def update(self, idx, new_value):
        idx = int(idx + 2**self.tree_depth-1)
        diff = new_value - self.value[idx]
        self.value[idx] = new_value
        while (idx-1) // 2 >= 0:
            father = (idx-1) // 2
            self.value[father] += diff
            idx = father
    
    def add(self, new_value, new_data):
        idx = self.curr
        self.data[self.curr] = new_data
        self.update(idx, new_value)
        self.curr = (self.curr+1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def find(self, target, normalize=True):
        if normalize:
            target *= self.value[0]
        return self._find_helper(target, 0)

    def _find_helper(self, target, index):
        if 2*index+1 > self.tree_size-1:
            return index- 2**self.tree_depth + 1, self.value[index], self.data[index - 2**self.tree_depth + 1]

        left_value = self.value[2*index+1]
        if target <= left_value:
            return self._find_helper(target, 2*index+1)
        else:
            return self._find_helper(target-left_value, 2*index+2)
            
    def __str__(self):
        res = []
        for i in range(self.tree_depth + 1):
            res.append("depth {}:\t".format(i)+str(self.value[2**i-1:2**(i+1)-1]))
        return "\n".join(res)

    @property
    def total(self):
        return self.value[0]
    
    @property
    def max(self):
        if self.size == 0:
            return 0
        start = 2**self.tree_depth + 1
        return np.max(self.value[start:start+self.size])
        
if __name__ == "__main__":
    st = SumTree(16)
    for i in range(0):
        st.add(i, i)
    print(st)
    res = st.find(0.5, True)
    print(res)
    print(st.max)