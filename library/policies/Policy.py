import numpy as np


class Policy:

    def __init__(self):
        self.no_bias = None
        self.basis = None

    # def eq(self, obj1=None, obj2=None):
    #     n1 = np.asarray(obj1).size
    #     n2 = np.asarray(obj2).size
    #     if n1 == n2:
    #         areEq = obj1 == obj2
    #     else:
    #         if n1 == 1:
    #             areEq = np.zeros((obj2.shape, obj2.shape))
    #             for i in np.arange(1, n2 + 1).reshape(-1):
    #                 areEq[i] = obj2(i) == obj1
    #         else:
    #             if n2 == 1:
    #                 areEq = np.zeros((obj1.shape, obj1.shape))
    #                 for i in np.arange(1, n1 + 1).reshape(-1):
    #                     areEq[i] = obj1(i) == obj2
    #             else:
    #                 raise Exception('Matrix dimensions must agree.')
    #
    #     return areEq

    def get_basis(self, States=None):
        if self.no_bias:
            phi = self.basis(States)
        else:
            phi = np.concatenate((np.ones((1, States.shape[1])), self.basis(States)), axis=0)

        return phi
