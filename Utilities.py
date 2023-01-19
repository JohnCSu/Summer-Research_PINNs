import torch.nn as nn
import numpy as np
import torch


from functorch import jacrev,vmap,make_functional

##Network Architectures

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

class Fourier_encoding(nn.Module):
    def __init__(self,num_inputs,num_freq) -> None:
        super().__init__()
        self.lin = nn.Linear(num_inputs,num_freq,bias =False)

    def forward(self,x):
        out = self.lin(x)
        return torch.concat((torch.sin(2*torch.pi*out),torch.cos(2*torch.pi*out)),dim = -1)


class Fourier_Net(nn.Module):
    def __init__(self,num_inputs = 4, num_outputs = 4,num_freq = 5,hidden_layers = 10, hidden_width = 50) -> None:
        super().__init__()
        self.non_lin = torch.sin
        self.input = Fourier_encoding(num_inputs,num_freq= num_freq)
        self.linear = nn.Linear(num_freq*2,hidden_width)
        self.hidden = nn.ModuleList([nn.Linear(hidden_width,hidden_width) for _ in range(hidden_layers-1)])
        self.head = nn.Linear(hidden_width,num_outputs )
    def forward(self,x):
        out = self.input(x)

        out = self.non_lin(self.linear(out))

        for h_layer in self.hidden:
            out = self.non_lin(h_layer(out))

        return self.head(out)


class Modified_Fourier_Net(nn.Module):
    def __init__(self,num_inputs = 4, num_outputs = 4,hidden_layers = 10, hidden_width = 50) -> None:
        super().__init__()
        self.non_lin = torch.sin

        self.fourier = nn.Linear(num_inputs,hidden_width)
        self.h1 = nn.Linear(num_inputs,hidden_width)

        self.hidden = nn.ModuleList([nn.Linear(hidden_width,hidden_width) for _ in range(hidden_layers-1)])
        self.head = nn.Linear(hidden_width,num_outputs )


    def forward(self,x):
        U,V = torch.sin(self.fourier(x)),torch.cos(self.fourier(x))
        out = self.non_lin(self.h1(x))

        for hidden in self.hidden:
            Z =self.non_lin(hidden(out))
            out = (1-Z)*U + (Z)*V
        
        return self.head(out)


############################### Derivatives ###################################
from functorch import vmap,vjp,jacrev

def aux_wrap(f):
    def aux_func(*x):
        result = f(*x)
        return result,result  
    return aux_func

#Crazy functional programming
def eff_hessian(net,x,hessian_vectors):
    '''
    Returns the full jacobian and Hessian vector porduct (i.e. subset of hessian matrix) of a non-batched tensor input
    '''
    aux_func = aux_wrap(net)

    jacob_func = (jacrev(aux_func,has_aux= True))

    out_jacob, partial_hess, out = vjp(jacob_func,x,has_aux=True)
    # print(out_jacob.shape)
    # print(hessian_vectors.shape)
    out_xx = vmap(lambda x : partial_hess(x,create_graph= True))(hessian_vectors)
    # return out,out_jacob,out_xx
    return out, out_jacob, out_xx


#Wrapper over get_derivatives to allow use of V map. We use the same Hessian Vectors across both
def batched_eff_hessian_wrapper(net,hessian_vectors):
    def batched_eff_hessian(x):
        return eff_hessian(net,x,hessian_vectors)
    return batched_eff_hessian

def batched_eff_hessian(net,x : torch.tensor ,hessian_vectors : torch.tensor):
    ''' 
    x is batched returns the func outpuy, full Jacobian and Hessian vector product given a number of vectors
    over batch
    
    Hessian vectors should be an n x # outputs x # inputs where n is the number of Jacobian elements to evaluate, 
    and ouputs x inputs are the number of 
     '''
    get_derivs = batched_eff_hessian_wrapper(net,hessian_vectors)
    return vmap(get_derivs)(x)

def batched_jacobian(net,x,return_func = True):
    if return_func:
        func = lambda x : jacrev(aux_wrap(net),has_aux= True)(x)
        return vmap(func)(x)
    
    return vmap(jacrev(net))(x) 




'''############################### Equation ###################################'''


def stream_function(fmodel,params,x,vec,device = 'cpu'):
        '''
        Calculate variables of interest from stream function form of Navier stokes
        Model Network: (x,y,t) --> (phi,p)
        

        phi is the stream function which the velocity are the derivatives of it
        u_x = d_phi/dy
        u_y = -d_phi/dx
        
        '''
        #Calulate stream function and required derivatives 
        #
        #Initial Jac Function gets u,v and p_x,p_y
        x = x.to(device)
        
        #Check whether network is normal OOP form or functional form
        if params is not None:
            jac = jacrev(fmodel,argnums=1)
            #Wrapped for vjp as vjp cant specify which argnums to differentiate
            f = lambda x : jac(params,x)
        else:
            f = jacrev(fmodel)
            

        jacobian = f(x)
        #This gives v
        vec_u,vec_v = vec
        #Hess gives derivs of v, we need second order derivs of v (third for stream function)
        
        #out is jacobian 2x3, we are predicting stream function and p so we get u,v,px,py from jacobian
        u,v,(px,py) = -jacobian[0,1], jacobian[0,0], jacobian[1,0:2]
        
        #derivative function of u,v (2nd order of stream function)
        v_deriv = lambda x : vjp(f,x)[1](vec_v)[0] #returns v derivative function 
        u_deriv = lambda x : vjp(f,x)[1](vec_u)[0] #returns u derivatives function
        
        #Get derivatives of u and v and the second derivative function
        (vx,vy,vt), d2v_f = vjp(v_deriv,x)    
        (ux,uy,ut), d2u_f = vjp(u_deriv,x)
    
        #Get Second order Derivatives of velocities
        vxx,vyy = d2v_f(torch.tensor([1,0,0]).to(device))[0][0] ,d2v_f(torch.tensor([0,1,0]).to(device))[0][1]
        # print(v_xx,v_yy,'\n\n')
        uxx,uyy = d2u_f(torch.tensor([1,0,0]).to(device))[0][0] , d2u_f(torch.tensor([0,1,0]).to(device))[0][1]         
        # print(u_x + v_y,'\n\n')

        return (px,py), (u,ux,uy,ut,uxx,uyy), (v,vx,vy,vt,vxx,vyy)



def navier_stokes_stream_2D(net,params,x,Re,vec,device = 'cpu'):
    # x,u_data,v_data,mask = ( d.to(device) for d in data )
    # phi,p = net(x)

    (px,py), (u,ux,uy,ut,uxx,uyy), (v,vx,vy,vt,vxx,vyy) = stream_function(net,params,x,vec,device)
    e1 = ut + u*ux+v*uy + px - 1/Re*(uxx + uyy)
    e2 = vt + vx*u +vy*v + py - 1/Re*(vxx+vyy)

    return e1,e2,u,v








# def NTK_weighting(func,net,params,x,*args,return_weights = True):
#     '''
#     Returns the NTK weighting across a batched data x, Func should be a function that takes functional version of your network, params,x
#     and then any other args.

#     func should take the order of (net,params,x,*args) as arguments. This is useful if the NTK weighings are not neccesarily the direct
#     output of the network e.g the residual of PINN networks.  

#     return_weights = True to return the NTK weighting otherwise set to false to return the computed trace of each output.
#     setting to false is useful if the entire batch cannot fit all at once on the GPU. mini_batch_NTK_weighting is recommended

#     So we jac1 returns a tuple with len == no. of outputs of the function. each element contains the batched jacobian of 
#     the each output wrt the parameters

#     Each j is a tuple of len = to the # of layers of the network
#     Each j contains a tensors of size [b,*li] where b is the batch size and *li is the shape of the layers i.e. params
    
#     As we only need the trace, this is corresponding to a simple dot product with itself so we can just square everything
#     And then sum all together to get K_ii = Tr(J@J.T) and then easily calulate the weigthings (hopefully)
#     '''

#     jac1 = vmap(jacrev(func,argnums = 1), (None,None, 0) +(None,)*len(args) )(net,params, x,*args)
    
#     trace_NTK = torch.tensor([ sum((l.pow(2).sum() for l in j)) for j in jac1])
    
#     return torch.tensor([sum(trace_NTK)/Kii for Kii in trace_NTK]) if return_weights else trace_NTK 

# def mini_batch_NTK_weighting(trainloader,func,net,*args):
#     '''
#     For 
#     '''
#     fmodel, params = make_functional(net,disable_autograd_tracking=True)

#     for i,x in enumerate(trainloader):
#         if i ==0:
#             mini_batch_traces = NTK_weighting(func,fmodel,params,x,*args,return_weights=False)        
#         else:
#             mini_batch_traces += NTK_weighting(func,fmodel,params,x,*args,return_weights=False)
    
#     return torch.tensor([sum(mini_batch_traces)/Kii for Kii in mini_batch_traces])
        
            


if __name__ == '__main__':

    '''
    The following is random testing of code
    '''
    net = Modified_Fourier_Net(3,2,5,100)
    # print(net.state_dict)
    # data = torch.tensor([1.,2.,3.])
    data = torch.rand((100,3))
    from torch.utils.data import DataLoader
    dl = DataLoader(data,batch_size= 10)
    # data.shape
    # data = torch.stack((data,data,data),dim = 0 )
    print(data.shape)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    vec_v = torch.zeros((2,3)).to(device)
    vec_v[0,0] = 1

    vec_u = torch.zeros((2,3)).to(device)
    vec_u[0,1] = -1
    vec = [vec_u.to(device),vec_v.to(device)]
    
    fmodel, params = make_functional(net,disable_autograd_tracking=True)
