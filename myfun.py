import numpy as np
#nn np n3


'''
R = 0.0025
m = 0.02
DensityRatio = 2.0
a = 3.0/8.0*R*(DensityRatio-1.0)/(DensityRatio+1.0)
I = 2.0*m*R*R/5.0
I3 = 0.8*I
mu = 0.3
g = 9.8
'''

R = 0.0025
m = 0.015
a = 0.0005
I = 0.4*m*R*R
I3 = 0.8*I
mu = 0.3
g = 9.8

def n_into_x(phi,psi,theta,A):
    x = A[0]*np.cos(phi) - A[1]*np.cos(theta)*np.sin(phi) + A[2]*np.sin(theta)*np.sin(phi)
    y = A[0]*np.sin(phi) + A[1]*np.cos(theta)*np.cos(phi) - A[2]*np.sin(theta)*np.cos(phi)
    z = A[1]*np.sin(theta) + A[2]*np.cos(theta)
    return np.array([x, y, z])

def x_into_n(phi,psi,theta, A):
    n = A[0]*np.cos(phi) + A[1]*np.sin(phi)
    p = -A[0]*np.cos(theta)*np.sin(phi) + A[1]*np.cos(theta)*np.cos(phi) + A[2]*np.sin(theta)
    t = A[0]*np.sin(theta)*np.sin(phi) - A[1]*np.sin(theta)*np.cos(phi) + A[2]*np.cos(theta)
    return np.array([n, p, t])
    
def mod(A):
    x2 = A[0]**2
    y2 = A[1]**2
    z2 = A[2]**2
    S = np.sqrt(x2+y2+z2)
    return S

'''
def vec_omega_n(phi,dot_phi,psi,dot_psi,theta,dot_theta):
    omega_nn = dot_theta
    omega_np = dot_phi*np.sin(theta)
    omega_nt = dot_psi+dot_phi*np.cos(theta)
    return np.array([omega_nn, omega_np, omega_nt])
    
def vec_alpha_n(phi,dot_phi,psi,dot_psi,theta,dot_theta):
    alpha_nn = dot_theta
    alpha_np = dot_phi*np.sin(theta)
    alpha_nt = dot_phi*np.cos(theta)
    return np.array([alpha_nn, alpha_np, alpha_nt])
'''
def vec_r_n(phi,dot_phi,psi,dot_psi,theta,dot_theta):
    r_nn = 0
    r_np = -R*np.sin(theta)
    r_nt = a-R*np.cos(theta)
    return np.array([r_nn, r_np, r_nt])
'''    
def vec_v_TC_n(phi,dot_phi,psi,dot_psi,theta,dot_theta):
    v_TC_nn = (R*dot_psi+a*dot_phi)*np.sin(theta)
    v_TC_np = dot_theta*(R*np.cos(theta)-a)
    v_TC_nt = -dot_theta*R*np.sin(theta)
    return np.array([v_TC_nn, v_TC_np, v_TC_nt])
'''
#x,y,z
def vec_v_TCP_x(phi,dot_phi,psi,dot_psi,theta,dot_theta):
    v_TCP_nx = (R*dot_psi+a*dot_phi)*np.sin(theta)*np.cos(phi)+dot_theta*np.sin(phi)*(a*np.cos(theta)-R)
    v_TCP_ny = (R*dot_psi+a*dot_phi)*np.sin(theta)*np.sin(phi)+dot_theta*np.cos(phi)*(R-a*np.cos(theta))
    v_TCP_nz = 0
    return np.array([v_TCP_nx, v_TCP_ny, v_TCP_nz])
    
def vec_F_N_x(ddz):
    F_N_z = m*g+m*ddz
    return np.array([0,0,F_N_z])
    

def ddot_phi(phi,dot_phi,psi,dot_psi,theta,dot_theta,N):
    return (N[1]-(2*I-I3)*dot_theta*dot_phi*np.cos(theta)+I3*dot_theta*dot_psi)/(I*np.sin(theta))
    
def ddot_psi(phi,dot_phi,psi,dot_psi,theta,dot_theta,N):
    return N[2]/I3-ddot_phi(phi,dot_phi,psi,dot_psi,theta,dot_theta,N)*np.cos(theta)+dot_phi*dot_theta*np.sin(theta)
    
def ddot_theta(phi,dot_phi,psi,dot_psi,theta,dot_theta,N):
    return (N[0]-I3*dot_phi*dot_psi*np.sin(theta)-(I3-I)*dot_phi**2*np.sin(theta)*np.cos(theta))
    
def ddot_z(phi,dot_phi,psi,dot_psi,theta,dot_theta):
    return a*(dot_theta**2*np.cos(theta)+0)
    
    
    
    
###############################################################################
#RK4
def RK4(f, t0, t1, x0, N):
    h = (t1-t0)/float(N)
    t = t0 + np.arange(N+1)*h
    x = np.zeros([N+1,len(x0)])
    x[0] = x0
    for i in range(N):
        k1 = h*f(x[i], t[i])
        k2 = h*f(x[i]+0.5*k1, t[i]+0.5*h)
        k3 = h*f(x[i]+0.5*k2, t[i]+0.5*h)
        k4 = h*f(x[i]+k3, t[i]+h)
        x[i+1] = x[i] + (k1+2*k2+2*k3+k4)/float(6)
    return t, x
    
    
###############################################################################
#RK2
def RK2(f, t0, t1, x0, N):
    h = (t1-t0)/float(N)
    t = t0 + np.arange(N+1)*h
    x = np.zeros([N+1,len(x0)])
    x[0] = x0
    for i in range(N):
        k1 = h*f(x[i],t[i])
        k2 = h*f(x[i]+0.5*k1, t[i]+0.5*h)
        x[i+1] = x[i] + k2
    return t, x
    
###############################################################################
#Euler Forward
def RK1(f, t0, t1, x0, N):
    h = (t1-t0)/float(N)
    t = t0 + np.arange(N+1)*h
    x = np.zeros([N+1,len(x0)])
    x[0] = x0
    for i in range(N):
        k1 = h*f(x[i],t[i])
        x[i+1] = x[i] + k1
    return t, x
    
    
###############################################################################
#Verlet    
def VL(f, t0, t1, x0, N):
    h = (t1-t0)/float(N)
    t = t0 + np.arange(N+1)*h
    x = np.zeros([N+1,len(x0)/2])
    v = np.zeros([N+1,len(x0)/2])
    x[0] = [x0[0],x0[2],x0[4],x0[6],x0[8]]
    v[0] = [x0[1],x0[3],x0[5],x0[7],x0[9]]
    v_temp = v[0] + 0.5*h*f(x[0],t[0])
    for i in range(N):
        x[i+1] = x[i] + h*v_temp
        k = h*f(x[i+1],t+h)
        v[i+1] = v_temp + 0.5*k
        v_temp = v_temp + k
    XX = np.zeros([N+1,len(x0)])
    XX[:,0] = x[:,0]
    XX[:,2] = x[:,1]
    XX[:,1] = v[:,0]
    XX[:,3] = v[:,1]
    return t, XX
