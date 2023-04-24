import numpy as np
    
def normalize_data(p=None, minp=None, maxp=None): 
    # NORMALIZE_DATA Normalizes data points by pn = (p - minp) / (maxp - minp).
    # If MINP == min(P) and MAXP == max(P), the points are normalized in [0,1].

    #    INPUT
    #     - p    : [N x D] matrix, where N is the number of points and D is the
    #              dimensionality of a point
    #     - minp : (optional) [1 x D] vector of the minimum feasible value the
    #              points can assume (min(p) by default)
    #     - maxp : (optional) [1 x D] vector of the maximum feasible value the
    #              points canassume (max(p) by default)

    #    OUTPUT
    #     - pn   : [N x D] matrix of normalized points
    
    if minp is None and maxp is None:
        minp = np.amin(p, 0)
        maxp = np.amax(p, 0)
    
    # checkmin = bsxfun(@ge,p,minp);
    # checkmin = min(checkmin(:));
    # checkmax = bsxfun(@le,p,maxp);
    # checkmax = min(checkmax(:));
    # if ~checkmin || ~checkmax
    #     warning('There are points out of the normalizing bounds.')
    # end

    if len(p) == 0:
        return np.array(p)
    else:
        pn = (p - minp) * (1.0 / (maxp - minp))
        return pn