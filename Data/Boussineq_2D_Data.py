import numpy as np
import pickle

def get_boussineq_2D(file_path:str,shuffle=True):
    with open(file_path,'rb') as f:
        data = pickle.load(f)

    data.keys()

    coords = data['coords']
    x_pre ,y_pre = coords[:,0][:,np.newaxis] , coords[:,-1][:,np.newaxis] # Nx1
    t_pre = data['time'][:,np.newaxis] #Tx1
    
    #Reshape coords and t to be NxT then flatten
    # x,y are Nx1 and t is Tx1
    T = t_pre.shape[0]
    N = x_pre.shape[0]

    x= np.tile(x_pre,(1,T))
    y= np.tile(y_pre,(1,T))
    t= np.tile(t_pre,(1,N)).T
    #Field Data is in shape NxT already
    u = data['u_velocity']
    v = data['y_velocity']
    temp = data['temperature']
    p = data['pressure']
    # return x_pre,y_pre,t_pre,u,v,temp
    # Flatten data into one long as array
    x_data = x.flatten()
    y_data = y.flatten()
    t_data = t.flatten()

    u_data = u.flatten()
    v_data = v.flatten()
    temp_data = temp.flatten()
    p_data = p.flatten()
    
    data = np.stack([x_data,y_data,t_data,u_data,v_data,temp_data,p_data],axis = -1)
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(data)
    # raise ValueError('OI SHUFFLE THE DATA YOU MONG')
    # Put all arrays together into one array
    return data
    # return x_data,y_data,t_data,u_data,v_data,temp_data,p_data

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # x,y,t,u,v,temp = get_boussineq_2D('Data/DHC_2D_Unsteady.dat',shuffle=False)


    # for i in range(0,40401,1000):
    #     plt.plot(t,v[i])
    #     plt.title('v velocitys of 1000 nodes over time')
    # plt.show()
    # print(u[:,0].shape)

    # v_mag = np.sqrt(v[:,-1]**2 + u[:,-1]**2)
    # cs = plt.tricontourf(x.squeeze(),y.squeeze(),temp[:,0])
    # plt.colorbar(cs)
    # plt.show()
    
    
    
    data =  get_boussineq_2D('Data/DHC_2D_Unsteady.dat',shuffle=False)
    for i in range(0,100,5):
        s0 = 240*i
        s1 = 240*(i+1)
        t = data[0:240,2]
        u = data[s0:s1,3]
        plt.plot(t,u)
    plt.show()
    


