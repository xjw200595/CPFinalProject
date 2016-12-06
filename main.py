import numpy as np
import matplotlib.pyplot as plt
from myfun import n_into_x, x_into_n, mod
from myfun import vec_r_n, vec_v_TCP_x, vec_F_N_x
from myfun import ddot_phi, ddot_psi, ddot_theta, ddot_z
from myfun import RK1, RK2, RK4, VL


R = 0.0025
m = 0.015
a = 0.0005
I = 0.4*m*R*R
I3 = 0.8*I
mu = 0.5
g = 9.8


def ODE(x,t):
    ddz = ddot_z(x[0],x[1],x[2],x[3],x[4],x[5])
    
    F_N_x = vec_F_N_x(ddz)
    F_N_n = x_into_n(x[0],x[2],x[4],F_N_x)
    FN = mod(F_N_n)

    
    u_CP_x = np.array([x[7],x[9],0])
    u_CP_n = x_into_n(x[0],x[2],x[4],u_CP_x)
    
    v_TCP_x = vec_v_TCP_x(x[0],x[1],x[2],x[3],x[4],x[5])
    v_TCP_n = x_into_n(x[0],x[2],x[4],v_TCP_x)
    
    v_TP_n = u_CP_n+v_TCP_n
    v_TP = mod(v_TP_n)
    
    F_f_n = -mu*FN/v_TP*v_TP_n
    F_f_x = n_into_x(x[0],x[2],x[4],F_f_n)    
    
    F_T_n = F_N_n + F_f_n
    
    r_n = vec_r_n(x[0],x[1],x[2],x[3],x[4],x[5])
    
    N = np.cross(r_n,F_T_n)
    
    dx0 = x[1]      #phi
    dx1 = ddot_phi(x[0],x[1],x[2],x[3],x[4],x[5],N)
    dx2 = x[3]      #psi
    dx3 = ddot_psi(x[0],x[1],x[2],x[3],x[4],x[5],N)
    dx4 = x[5]      #theta
    dx5 = ddot_theta(x[0],x[1],x[2],x[3],x[4],x[5],N)
    dx6 = x[7]      #rx
    dx7 = F_f_x[0]/m
    dx8 = x[9]      #ry
    dx9 = F_f_x[1]/m
    print np.sign(dx6*dx7)
    return np.array([dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9])
    



phi_0 = 0.0
dot_phi_0 = 0.0
psi_0 = 0.0
dot_psi_0 = 100.0
theta_0 = 0.1
dot_theta_0 = 0.0
rx_0 = 0.0
dot_rx_0 = 0.0
ry_0 = 0.0
dot_ry_0 = 0.0
###############  x[0]    x[1]      x[2]    x[3]       x[4]      x[5]      x[6]   x[7]     x[8]     x[9]
x_0 = np.array([phi_0, dot_phi_0, psi_0, dot_psi_0, theta_0, dot_theta_0, rx_0, dot_rx_0, ry_0, dot_ry_0])
t_start = 0.0
t_end = 1.0
TimeNumber = 2000*(t_end-t_start)
[TT, XX] = RK4(ODE, t_start, t_end, x_0, int(TimeNumber)) 


fig, ((ax1)) = plt.subplots(nrows=1, ncols=1)
ax1.plot(TT,XX[:,4])
#ax1.set_ylim(0,1)