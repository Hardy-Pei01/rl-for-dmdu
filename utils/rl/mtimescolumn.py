import numpy as np


def mtimescolumn(A=None, B=None):
    # MTIMESCOLUMN Multiplies each column of a matrix A by each row of a matrix
    # B to obtain many 2d matrices. These matrices are then vectorized in a 2d
    # matrix (one vectorization per column).
    # It is equivalent to the following loop:
    # >> for i = 1 : D
    # >>     tmp = A(:,i) * B(:,i)';
    # >>     C(:,i) = tmp(:);
    # >> end
    #
    #    INPUT
    #     - A : [N x D] matrix
    #     - B : [M x D] matrix
    #
    #    OUTPUT
    #     - C : [N*M x D] matrix

    N = A.shape[0]
    M = B.shape[0]
    C = (np.transpose(A[:, :, None], (0, 2, 1)) * np.transpose(B[:, :, None], (2, 0, 1))).reshape((N * M, -1))
    return C
