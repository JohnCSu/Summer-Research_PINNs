from Utilities import *
from Data.HFM_training_data import *
import torch
from torch.utils.data import DataLoader,Dataset
from functorch import jacrev,vmap,vjp
import numpy as np
from torch import tensor
'''
Script to recreate HFM
'''

def aux_wrap(net):
    def aux_net(xyt):
        result = net(xyt)
        return result,result
    return aux_net
def get_conc_derivs(net,xyt,hes_vecs):
    '''
    Get the all the derivatives for the equations for Navier Stokes and Concentration Equations
    Assumes the NN Model maps: (x,y,t) --> (u,v,p,c)

    Uses Functorch a JAX like interface to get derivatives
    '''
    aux_net = aux_wrap(net)
    #Crazy One Functional liner gives returns (Jacobian,Hessian function, Network Evaluation) for input xyt = (x,y,t)
    jac,partial_hess,out = vjp(jacrev(aux_net,has_aux = True),xyt,has_aux = True)

    u,v,p,c = out

    #Jacobian:
    ## x    y    t
    #u ux  uy   ut
    #v vx  vy   vt
    #p px  py   pt
    #c cx  cy   ct
    ux,uy,ut = jac[0,:]
    vx,vy,vt = jac[1,:]
    px,py    = jac[2,0:2]
    cx,cy,ct = jac[3,:]
    
    #The partial hess allows us to get the derivatives of specific elements in the Jacobian
    #Much more effecient than calulating entire hessian
    d2 = vmap(partial_hess)(hes_vecs)[0]

    #Second Order Terms
    uxx,uyy = d2[[0,1],[0,1]]
    vxx,vyy = d2[[2,3],[0,1]]
    cxx,cyy = d2[[4,5],[0,1]]

    return (p,px,py), (c,cx,cy,ct,cxx,cyy),(u,ux,uy,ut,uxx,uyy),(v,vx,vy,vt,vxx,vyy) 
    
# Code for one epoch
def one_epoch(train_loader,net,optimizer,Re = 100, Pec = 100, device = 'cpu',weights = 5*[1.]):
    vecs = torch.zeros((6,4,3))
    #These tensors allow us to effeciently get 2nd order derivatives without
    #Computing the entire Hessian
    #uxx uyy
    vecs[0,0,0] = 1
    vecs[1,0,1] = 1

    #vxx vyy
    vecs[2,1,0] = 1
    vecs[3,1,1] = 1

    #Cxx, Cyy
    vecs[4,3,0] = 1
    vecs[5,3,1] = 1
    vecs = vecs.to(device)

    #Variables to keep track of total loss and individual losses (unweighted)
    running_loss = 0.
    indi_loss = torch.zeros(5)

    for data in train_loader:
        optimizer.zero_grad()
        
        xyt,c_data = data[0].to(device),data[1].to(device)
        (p,px,py), (c,cx,cy,ct,cxx,cyy),(u,ux,uy,ut,uxx,uyy),(v,vx,vy,vt,vxx,vyy) = vmap(get_conc_derivs,(None,0,None))(net,xyt,vecs)
        
        #Navier stokes eq
        e1 = ut + u*ux+v*uy + px - 1/Re*(uxx + uyy)
        e2 = vt + vx*u +vy*v + py - 1/Re*(vxx+vyy)
        #Incompress
        e3 = ux+vy
        #Concentration eq
        e4 = ct + (u*cx + v*cy) - 1/Pec*(cxx+cyy)
        # Data Fitting
        e5 = c-c_data 
        
        eq_s = [e1,e2,e3,e4,e5]

        #Sum of MSE of each loss
        total_loss = sum([w*e.pow(2).mean() for w,e in zip(weights,eq_s)])
        
        #Get Gradients then update weights
        total_loss.backward()
        optimizer.step()
        
        #Loss Tracking
        running_loss += total_loss
        indi_loss += tensor([e.pow(2).mean() for e in eq_s])
        
    return running_loss ,indi_loss 

class TensorData(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        data = self.data[idx,:]
        xyt = data[0:3]
        c = data[-1]
        return xyt,c
from datetime import datetime
import csv
import time

if __name__ == '__main__':
    #Set Network Up
    net = HFM_Net(3,4,10,200)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # net.non_lin = torch.tanh
    net = net.to(device)
    
    #Data Wrangling see HFM_training data for how data works (from HFM paper)
    t_data,x_data,y_data,u_data,v_data,c_data = get_data('Cylinder2D.mat')
    data = np.stack([x_data,y_data,t_data,c_data], axis = 1)
    new_data = tensor(data,dtype = torch.float32).squeeze()

    #DataLoader for Easy shuffling of data. Only Use 2_000_000 points
    train_set = TensorData(new_data[:2000,:])
    train_loader = DataLoader(train_set,batch_size = 1_000,shuffle = True,num_workers =2)

    optimizer = torch.optim.Adam(net.parameters(),lr = 1e-3)

    #Tracking Loss
    best_loss = float('inf')
    currentDateAndTime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    epoch_losses = []

    MAX_HOUR_DURATION = 40
    MAX_EPOCHS = 1
    
    print(f'\n\n Starting Run at time {currentDateAndTime} using device {device} \n\n')

    with open(f'Run_{currentDateAndTime}.csv','w',newline = '') as f:
    
        #Text file to track losses
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Epoch','Total Loss','NS_u','NS_y','Incompressibility','Conc_eq','Conc_data'])
        
        timer_start = time.perf_counter()
        for i in range(MAX_EPOCHS):
            #One epoch training
            loss,indi_loss = one_epoch(train_loader,net,optimizer,Re = 100, Pec = 100, device = device)
            
            #Record Losses and elapsed time
            epoch_losses.append( [i,float(loss)] + list(map(float,indi_loss)) )
            csv_writer.writerow(epoch_losses[-1])
            elapsed_time = (time.perf_counter() - timer_start)/3600 
            
            if i % 2  == 0:

                print(f'Time (H) {elapsed_time:.3f} Epoch {i} loss {loss:3f} Equation Losses \t {indi_loss}') 
                
                if loss < best_loss:
                    torch.save(net.to('cpu').state_dict(),f'HFM_Cylinder2D_run_{currentDateAndTime}.pth')
                    best_loss = loss
                    net.to(device)
                    print('Saved!')
            if elapsed_time > MAX_HOUR_DURATION:
                break


