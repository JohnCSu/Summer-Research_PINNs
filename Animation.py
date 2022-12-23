from matplotlib import animation
from Utilities import *
from torch import tensor
import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader

net_state_dict = torch.load('HFM_Test.pth',map_location= 'cpu')

net = HFM_Net(3,3,10,30)

net.load_state_dict(net_state_dict)



x = torch.linspace(-2.5,7.5,50)
y = torch.linspace(-2.5,2.5,50)
t = torch.linspace(0,15,60)
grid = torch.meshgrid([x,y,t])

flat_grids = [torch.flatten(i) for i in grid]

data = torch.stack( flat_grids ,dim = 1)

h_vec =torch.stack([torch.zeros((3,3)) for i in range(4)])

h_vec[0,0,0] = 1
h_vec[1,0,1] = 1
h_vec[2,1,0] = 1
h_vec[3,1,1] = 1 



out,out_jac,out_p_hess = batched_deriv(net,data,h_vec)
u,v,p = [out[:,i] for i in range(3)]
u_jac = out_jac[:,0:2,0:2]


uy = u_jac[:,0,1]
vx = u_jac[:,1,0]

vorticity = vx-uy



u_grid = u.reshape(grid[0].shape).detach()
vort = vorticity.reshape(grid[0].shape).detach()


x_grid,y_grid = grid[0][:,:,0],grid[1][:,:,0]
import numpy as np
# plt.figure(figsize=(10,2))




fig,ax = plt.subplots()
fig.set_figwidth(12.5)
fig.set_figheight(5)

level_min,level_max = torch.min(u_grid),torch.max(u_grid)

level_min2,level_max2 = torch.min(vort),torch.max(vort)


# CS = ax.contourf(x_grid,y_grid,u_grid[:,:,-0], levels = np.linspace(level_min, level_max,10) )
CS = ax.contourf(x_grid,y_grid,vort[:,:,0], levels = np.linspace(level_min2, level_max2,10) )

circ = plt.Circle((0,0),0.5)

fig.colorbar(CS)
def animateGraph(i):
    ax.clear()
    ax.set_title(f'Vorticity at Time {t[i]:.2f}')
    CS = ax.contourf(x_grid,y_grid,vort[:,:,i], levels = np.linspace(level_min2, level_max2,10) )
    ax.add_patch(circ)

anim = animation.FuncAnimation(fig,animateGraph,60,interval=50)
anim.save('Vorticity.gif',writer= 'Pillow')
plt.show()
