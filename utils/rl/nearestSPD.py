import numpy as np
from scipy import linalg


def nearestSPD(A):
    # nearestSPD - the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
    # usage: Ahat = nearestSPD(A)

    # From Higham: "The nearest symmetric positive semidefinite matrix in the
    # Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
    # where H is the symmetric polar factor of B=(A + A')/2."

    # http://www.sciencedirect.com/science/article/pii/0024379588902236

    # arguments: (input)
    #  A - square matrix, which will be converted to the nearest Symmetric
    #    Positive Definite Matrix.

    # Arguments: (output)
    #  Ahat - The matrix chosen as the nearest SPD matrix to A.

    # test for a square matrix A
    r, c = A.shape
    if r != c:
        raise Exception('A must be a square matrix.')
    elif (r == 1) and (A <= 0):
        # A was scalar and non-positive, so just return eps
        Ahat = np.spacing(1)
        return Ahat

    # symmetrize A into B
    B = (A + A.T) / 2
    # Compute the symmetric polar factor of B. Call it H.
    # Clearly H is itself SPD.
    U, sdiag, VH = linalg.svd(B)
    Sigma = np.zeros((B.shape[0], B.shape[1]))
    np.fill_diagonal(Sigma, sdiag)
    V = VH.T.conj()

    H = np.matmul(np.matmul(V, Sigma), V.T)
    # get Ahat in the above formula
    Ahat = (B + H) / 2
    # ensure symmetry
    Ahat = (Ahat + Ahat.T) / 2
    # test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
    k = 0
    while k < 1000.0:
        k = k + 1
        try:
            linalg.cholesky(Ahat)
            break
        except:
            # Ahat failed the chol test. It must have been just a hair off,
            # due to floating point trash, so it is simplest now just to
            # tweak by adding a tiny multiple of an identity matrix.
            mineig = np.amin(linalg.eigvals(Ahat)).real
            Ahat = Ahat + (- mineig * k ** 2 + np.spacing(mineig)) * np.identity(A.shape[0])
        # print(f"nearestSPD: {k}")
    return Ahat
