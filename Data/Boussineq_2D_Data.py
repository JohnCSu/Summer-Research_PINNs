import numpy as np
import pickle

def get_boussineq_2D(file_path:str):
    with open(file_path,'rb') as f:
        data = pickle.load(f)

    data.keys()

    coords = data['coords']
    x_pre ,y_pre = coords[:,0], coords[:,-1] # Nx1
    t_pre = data['time'] #Tx1
    
    #Reshape coords and t to be NxT then flatten
    # x,y are Nx1 and t is Tx1
    x = np.stack([x_pre]*len(t_pre),axis = -1)
    y = np.stack([y_pre]*len(t_pre),axis = -1)
    t = np.stack([t_pre]*len(x_pre),axis = 0)

    #Field Data is in shape NxT already
    u = data['u_velocity']
    v = data['y_velocity']
    temp = data['temperature']
    p = data['pressure']

    # Flatten data into one long as array
    x_data = x.flatten()
    y_data = y.flatten()
    t_data = t.flatten()

    u_data = u.flatten()
    v_data = v.flatten()
    temp_data = temp.flatten()
    p_data = p.flatten()
    
    data = np.stack([x_data,y_data,t_data,u_data,v_data,temp_data,p_data],axis = -1)
    np.random.seed(42)
    np.random.shuffle(data)
    # raise ValueError('OI SHUFFLE THE DATA YOU MONG')
    # Put all arrays together into one array
    return data
    # return x_data,y_data,t_data,u_data,v_data,temp_data,p_data