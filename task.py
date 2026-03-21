import numpy as np

class CoordinateSystem:
    def __init__(self, basis):
        self.basis = np.array(basis).T
    def is_valid(self):
        return np.linalg.det(self.basis) != 0
    def transfer(self, v):
        return np.linalg.solve(self.basis, v)

if __name__ == "__main__":
    v_origin = np.array([1, 1])
    new_basis = [[1, 0], [1, 1]]
    my_sys = CoordinateSystem(new_basis)
    if my_sys.is_valid():
        new_v = my_sys.transfer(v_origin)
        print("转移后的结果:", new_v)