import torch.nn as nn
import numpy as np
import torch
import scipy.io
import os
import time

from functorch import jacrev,vmap

class HFM_Net(nn.Module):
    def __init__(self,num_inputs = 4, num_outputs = 4,hidden_layers = 10, hidden_width = 50) -> None:
        super().__init__()
        self.non_lin = torch.sin
        self.input = nn.Linear(num_inputs,hidden_width)
        self.hidden = nn.ModuleList([nn.Linear(hidden_width,hidden_width) for _ in range(hidden_layers)])
        self.head = nn.Linear(hidden_width,num_outputs )
    def forward(self,x):
        
        out = self.non_lin(self.input(x))

        for h_layer in self.hidden:
            out = self.non_lin(h_layer(out))

        return self.head(out)


############################### Derivatives ###################################
from functorch import vmap,vjp,jacrev,hessian, jacfwd,jvp


def aux_wrap(f):
    def aux_func(x):
        result = f(x)
        return result,result  
    return aux_func

#Crazy functional programming
def get_derivatives(net,x,hessian_vectors):
    aux_func = aux_wrap(net)

    jacob_func = (jacrev(aux_func,has_aux= True))

    out_jacob, partial_hess, out = vjp(jacob_func,x,has_aux=True)
    # print(out_jacob.shape)
    # print(hessian_vectors.shape)
    out_xx = vmap(lambda x : partial_hess(x,create_graph= True))(hessian_vectors)
    # return out,out_jacob,out_xx
    return out, out_jacob, out_xx


#Wrapper got get_derivatives to allow use of V map
def batch_get_derivatives_wrapper(net,hessian_vectors):
    def batched_derivatives(x):
        return get_derivatives(net,x,hessian_vectors)
    return batched_derivatives

def batched_deriv(net,x,hessian_vectors):
    ''' x is batched '''
    get_derivs = batch_get_derivatives_wrapper(net,hessian_vectors)
    return vmap(get_derivs)(x)




############################### Equation ###################################



def navier_stokes(u,u_jac,u_xx,u_t,p_x,Re):
    #Calculate batch navier stokes residual. inputs should be size bxn, bxnxn for the case of var u_jac, and constant Re
    
    # Broadcast values to allow batch addition of scalars
    # so e.g. for batch 0, laplacian[0] is treated as a const so we can do tensor + int broadcasting
    laplacian = u_xx.sum(dim = -1).view(-1,1)
    
    # Equiv to matmul of batched u^T * u_jac. size -> [b,3] this gives the convection term
    # convection = torch.einsum('bi,bij -> bj',u,u_jac)
    # print(u.unsqueeze(dim= -1).shape)
    convection = torch.bmm(u_jac,u.unsqueeze(dim= -1)).squeeze(dim = -1) 
    Du = u_t  + convection 
    
    return Du + p_x - 1/Re*(laplacian)
def incompressibility(u_x):
    #Grad Op dot u == sum of spatial derivatives
    #Pass in the diagonals of the jacobian matrix for u e.g. torch.diagonal(ux,0,dim1 = 1, dim2 = -1)
    return u_x.sum(dim=-1)

def scalar_transport(c,c_x,c_xx,c_t,u,Pec):
    #Pass in the column for c derivative 
    # c,ct -> size [b,1], cx,cxx size [b,3], u size,[b,3] 
    laplacian = c_xx.sum(dim = -1).view(-1,1)

    return c_t + (u*c_x).sum(dim= -1) - 1/Pec*(laplacian)

