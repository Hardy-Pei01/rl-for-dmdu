    
def vec2mat(vec = None,n_rows = None): 
    # VEC2MAT Transforms a vector into a matrix. Elements are put into the
    # matrix column by column.

    #    INPUT
    #     - vec    : the vector
    #     - n_rows : number of rows of the matrix

    #    OUTPUT
    #     - mat    : the matrix

    # =========================================================================
    # EXAMPLE
    # vec = [1 2 3 4 5 6 7 8 9 10 11 12]
    # vec2mat(vec,2)
    #   [ 1     3     5     7     9    11
    #     2     4     6     8    10    12 ]
    
    n_col = (len(vec)) // n_rows
    mat = vec.reshape(n_rows, n_col)
    return mat