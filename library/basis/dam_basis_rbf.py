import numpy as np
from library.basis.basis_rrbf import basis_rrbf
    
def dam_basis_rbf(state=None):
    centers = np.array([[-20, 50, 120, 190]])
    width = 60
    if state is None:
        phi = basis_rrbf(centers, width, 0)
    else:
        assert(state.shape[0] == 1)
        phi = basis_rrbf(centers, width, 0, state)

    return phi