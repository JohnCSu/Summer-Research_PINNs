import scipy.io
import numpy as np

def get_data(data_dir='/content/drive/MyDrive/HFM/Cylinder2D.mat'):
    data = scipy.io.loadmat(data_dir)
    t_star = data['t_star'] # T x 1
    x_star = data['x_star'] # N x 1
    y_star = data['y_star'] # N x 1

    T = t_star.shape[0]
    N = x_star.shape[0]

    U_star = data['U_star'] # N x T
    V_star = data['V_star'] # N x T
    P_star = data['P_star'] # N x T
    C_star = data['C_star'] # N x T

    np.random.seed(42)

    # Rearrange Data 
    T_star = np.tile(t_star, (1,N)).T # N x T
    X_star = np.tile(x_star, (1,T)) # N x T
    Y_star = np.tile(y_star, (1,T)) # N x T

    t = T_star.flatten()[:,None] # NT x 1
    x = X_star.flatten()[:,None] # NT x 1
    y = Y_star.flatten()[:,None] # NT x 1
    u = U_star.flatten()[:,None] # NT x 1
    v = V_star.flatten()[:,None] # NT x 1
    p = P_star.flatten()[:,None] # NT x 1
    c = C_star.flatten()[:,None] # NT x 1

    ######################################################################
    ######################## Training Data ###############################
    ######################################################################

    T_data = T # int(sys.argv[1])
    N_data = N # int(sys.argv[2])
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_data, replace=False)
    t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    u_data = U_star[:, idx_t][idx_x,:].flatten()[:,None]
    v_data = V_star[:, idx_t][idx_x,:].flatten()[:,None]
    c_data = C_star[:, idx_t][idx_x,:].flatten()[:,None]
        
    T_eqns = T
    N_eqns = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_eqns, replace=False)
    t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None]

    # Training Data on velocity (inlet)
    t_inlet = t[x == x.min()][:,None]
    x_inlet = x[x == x.min()][:,None]
    y_inlet = y[x == x.min()][:,None]
    u_inlet = u[x == x.min()][:,None]
    v_inlet = v[x == x.min()][:,None]

    return t_data,x_data,y_data,u_data,v_data,c_data


