from functorch import vmap,vjp,jacrev,hessian,jvp   
import torch

def aux_wrap(f):
    def aux_func(x):
        result = f(x)
        return result,result  
    return aux_func

#Crazy functional programming
def get_derivatives(net,x):
    aux_func = aux_wrap(net)

    jacob_func = (jacrev(aux_func,has_aux= True))


    out_jacob, partial_hess, out = vjp(jacob_func,x,has_aux=True)
    
    #Only want derivatives along diagonal i.e. u_xx,v_yy etc

    base = torch.zeros( (x.shape[-1],x.shape[-1],x.shape[-1]))
    mask = [range(x.shape[-1]) for i in range(x.shape[-1])]
    base[mask] = 1


    out_xx = vmap(lambda x : partial_hess(x,create_graph= True))(base)
    # return out,out_jacob,out_xx
    return out, out_jacob, out_xx


#Wrapper got get_derivatives to allow use of V map
def batch_get_derivatives_wrapper(net):
    def batched_derivatives(x):
        return get_derivatives(net,x)
    return batched_derivatives

def batched_deriv(net,x):
    ''' x is batched '''
    get_derivs = batch_get_derivatives_wrapper(net)
    # get_derivs = lambda x : get_derivatives(net,x)
    return vmap(get_derivs)(x)
