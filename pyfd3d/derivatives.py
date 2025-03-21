import numpy as np
import scipy.sparse as sp
import numpy as np
try:
    import cupy as cp
    from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def createDws(w, f, dL, N):
    '''
        s = 'x' or 'y': x derivative or y derivative
        f = 'b' or 'f'
        catches exceptions if s and f are misspecified
    '''
    M = np.prod(N);
  

    sign = 1 if f == 'f' else -1;
    dw = None; #just an initialization
    indices = np.reshape(np.arange(M), N, order = 'F');
    if(w == 'x'):
        ind_adj = np.roll(indices, -sign, axis = 0)
        dw = dL[0]
    elif(w == 'y'):
        ind_adj = np.roll(indices, -sign, axis = 1)
        dw = dL[1];
    elif(w == 'z'):
        ind_adj = np.roll(indices, -sign, axis = 2)
        dw = dL[-1]
        
    # we could use flatten here since the indices are already in 'F' order
    indices_flatten = np.reshape(indices, (M, ), order = 'F')
    indices_adj_flatten = np.reshape(ind_adj, (M, ), order = 'F')
    # on_inds = np.hstack((indices.flatten(), indices.flatten()))
    # off_inds = np.concatenate((indices.flatten(), ind_adj.flatten()), axis = 0);
    on_inds = np.hstack((indices_flatten, indices_flatten));
    off_inds = np.concatenate((indices_flatten, indices_adj_flatten), axis = 0);

    all_inds = np.concatenate((np.expand_dims(on_inds, axis =1 ), np.expand_dims(off_inds, axis = 1)), axis = 1)

    data = np.concatenate((-sign*np.ones((M)), sign*np.ones((M))), axis = 0)
    Dws = sp.csc_matrix((data, (all_inds[:,0], all_inds[:,1])), shape = (M,M));

    return (1/dw)*Dws;

def createDws_gpu(w, f, dL, N):
    '''
    GPU-accelerated version of createDws using CuPy
    '''
    if not GPU_AVAILABLE:
        raise ImportError("CuPy is not available. Cannot use GPU acceleration.")
        
    M = int(cp.prod(N).item())
    sign = 1 if f == 'f' else -1
    
    # Create indices on CPU first, then transfer to GPU
    indices_cpu = np.reshape(np.arange(M), N, order='F')
    indices = cp.asarray(indices_cpu)
    
    if w == 'x':
        ind_adj = cp.roll(indices, -sign, axis=0)
        dw = dL[0]
    elif w == 'y':
        ind_adj = cp.roll(indices, -sign, axis=1)
        dw = dL[1]
    elif w == 'z':
        ind_adj = cp.roll(indices, -sign, axis=2)
        dw = dL[-1]
    
    indices_flatten = cp.reshape(indices, (M,), order='F')
    indices_adj_flatten = cp.reshape(ind_adj, (M,), order='F')
    
    on_inds = cp.hstack((indices_flatten, indices_flatten))
    off_inds = cp.concatenate((indices_flatten, indices_adj_flatten), axis=0)
    
    all_inds = cp.concatenate((cp.expand_dims(on_inds, axis=1), cp.expand_dims(off_inds, axis=1)), axis=1)
    
    data = cp.concatenate((-sign * cp.ones(M), sign * cp.ones(M)), axis=0)
    Dws = cp_csc_matrix((data, (all_inds[:,0], all_inds[:,1])), shape=(M, M))
    
    return (1/dw) * Dws