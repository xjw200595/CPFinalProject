import numpy as np
import matplotlib.pyplot as plt
from myfun import RK4, RK4_AD, Energy, ODE


from scipy.integrate import odeint
from pylab import *


    

g = 9.82
mu = 0.3
m = 0.02
R = 0.02
alpha = 0.3
I1 = 131/350*m*R**2
I3 = 0.4*m*R**2   
    
t_end = 8.0
time = np.linspace(0.0, t_end, 20000)

theta = 0.1
dot_theta = 0.0
dot_phi = 0.0
omega = 155.0
vx = 0.0
vy = 0.0
x_0 = np.array([theta, dot_theta, dot_phi, omega, vx, vy])  


sigma = 0.01
h = 0.00001

#time, y = RK4_AD(ODE, 0.0, t_end, x_0, sigma, h)
#y = RK4_modify(ODE, x_0, time)
y = odeint(ODE,x_0,time)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
ax1.plot(time,y[:,0],label=r"$\theta$")
ax1.plot([0,8],[0.5*np.pi,0.5*np.pi])
ax1.plot([0,8],[np.pi,np.pi])
ax1.set_ylim(0,np.pi+0.1)
ax1.legend(loc=2)

E = Energy(y[:,0], y[:,1], y[:,2], y[:,3])
ax2.plot(time, E, label="Energy")
ax2.set_ylim(0,np.max(E)*1.1)
ax2.legend()

ax3.plot(time, y[:,3], label=r"$\omega$")
ax3.legend()

ax4.plot(time, y[:,4], label=r"$v_x$")
ax4.plot(time, y[:,5], label=r"$v_y$")
ax4.legend()