import numpy as np
import matplotlib.pyplot as plt
from myfun import n_into_x, x_into_n, mod
from myfun import vec_r_n, vec_v_TCP_x, vec_F_N_x
#from myfun import ddot_phi, ddot_psi, ddot_theta, ddot_z
from myfun import RK1, RK2, RK4, VL

R = 0.0025
m = 0.015
a = 0.0005
I = 0.4*m*R*R
I3 = I
mu = 0.3
g = 9.8

phi = 0
dot_phi = 0
psi = 0
dot_psi = 100
theta = 0.1
dot_theta = 0
x = 0
dot_x = 0
y = 0
dot_y = 0

def vec_v_TP_x(dot_x, dot_y, phi, dot_phi, psi, dot_psi, theta, dot_theta):
    vx = dot_x + (R*dot_psi+a*dot_phi)*np.sin(theta)*np.cos(phi) + dot_theta*np.sin(phi)*(a*np.cos(theta)-R)
    vy = dot_y + (R*dot_psi+a*dot_phi)*np.sin(theta)*np.sin(phi) + dot_theta*np.cos(phi)*(R-a*np.cos(theta))
    return np.array([vx, vy, 0])

def vec_C_n(phi, psi, theta, v_TP_x):
    left_n = np.array([0,-R*np.sin(theta),a-R*np.cos(theta)])
    right_x = np.array([0,0,1])-mu*v_TP_x/mod(v_TP_x)
    right_n = x_into_n(phi, psi, theta, right_x)
    C_n = np.cross(left_n,right_n)
    return C_n
    
def f_ddot_theta(dot_phi, dot_psi, dot_theta, Cn):
    upper1 = I3*dot_phi*dot_psi*np.sin(theta)   
    upper2 = (I3-I)*dot_phi**2*np.sin(theta)*np.cos(theta)
    upper3 = - m*g*Cn
    upper4 = - m*a*dot_theta**2*np.sin(theta)*Cn
    bottom = m*a*np.sin(theta)*Cn-I
    Solution = (upper1 + upper2 + upper3 + upper4)/bottom
    return Solution
    
def f_ddz(theta, dot_theta, ddot_theta):
    ddz = a*(dot_theta**2 + ddot_theta*np.sin(theta))
    return ddz
    
def vec_N(ddz, C_n):
    N = m*(g+ddz)*C_n
    return N
    
def f_ddot_phi2(phi,dot_phi,psi,dot_psi,theta,dot_theta,N):
    return (N[1]-(2*I-I3)*dot_theta*dot_phi*np.cos(theta)+I3*dot_theta*dot_psi)/(I*np.sin(theta))
    
def f_ddot_psi2(phi,dot_phi,psi,dot_psi,theta,dot_theta,N):
    return N[2]/I3-f_ddot_phi2(phi,dot_phi,psi,dot_psi,theta,dot_theta,N)*np.cos(theta)+dot_phi*dot_theta*np.sin(theta)
    
def f_ddot_theta2(phi,dot_phi,psi,dot_psi,theta,dot_theta,N):
    return (N[0]-I3*dot_phi*dot_psi*np.sin(theta)-(I3-I)*dot_phi**2*np.sin(theta)*np.cos(theta))
    
    
v_TP_x = vec_v_TP_x(dot_x, dot_y, phi, dot_phi, psi, dot_psi, theta, dot_theta)
C_n = vec_C_n(phi, psi, theta, v_TP_x)
ddot_theta = f_ddot_theta(dot_phi, dot_psi, dot_theta, C_n[0])
ddz = f_ddz(theta, dot_theta, ddot_theta)
N = vec_N(ddz, C_n)


ddot_phi2 = (phi,dot_phi,psi,dot_psi,theta,dot_theta,N)
ddot_psi2 = (phi,dot_phi,psi,dot_psi,theta,dot_theta,N)
ddot_theta2 = (phi,dot_phi,psi,dot_psi,theta,dot_theta,N)


