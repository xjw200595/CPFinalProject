import numpy as np
#nn np n3

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
#One step of RK4
def RK4_1step(f,x,t,h):
    k1 = h*f(x,t)
    k2 = h*f(x+0.5*k1, t+0.5+h)
    k3 = h*f(x+0.5*k2, t+0.5*h)
    k4 = h*f(x+k3,t+h)
    x = x+(k1+2*k2+2*k3+k4)/float(6)
    t = t + h
    return t, x 

###############################################################################
#RK4 with adaptive
def RK4_AD(f, t0, t_end, x0, sigma, h):    
    T = []
    T.append(t0)   
    XX = []
    XX.append(x0)
    t = t0    
    while t < t_end-t0:
        rho = 0
        while rho < 1 or rho == 1:    
            x1 = XX[-1]
            x2 = XX[-1]           
            t1 = T[-1]
            t2 = T[-1]
            h1 = h
            h2 = h*2
            tmid, xmid = RK4_1step(f,x1,t1,h1)
            t1, x1 = RK4_1step(f,xmid,tmid,h1)
            t2, x2 = RK4_1step(f,x2,t2,h2)
            
            rho = h*sigma*30/np.sqrt(((x1-x2)**2).sum())   
            if rho<1:
                h = h*rho**0.25
        t = t + h2  
        print(t)
        T.append(t)
        XX.append(x1)            
        h = min([h*rho**0.25,2*h,(t_end-t)/2])    
    return T, np.array(XX)
    
def ddtheta(theta, dot_theta, dot_phi, omega, vx, vy, gn):
    solution = np.sin(theta)/I1*(I1*dot_phi**2*np.cos(theta) - I3*omega*dot_phi - R*alpha*gn) + R*mu*gn*vx/I1*(1-alpha*np.cos(theta))
    return solution

def ddphi(theta, dot_theta, dot_phi, omega, vx, vy, gn):
    solution = (I3*dot_theta*omega - 2*I1*dot_theta*dot_phi*np.cos(theta) - mu*gn*vy*R*(alpha-np.cos(theta)))/(I1*np.sin(theta))
    return solution
        
def domega(theta, dot_theta, dot_phi, omega, vx, vy, gn):
    solution = -mu*gn*vy*R*np.sin(theta)/I3
    return solution
    
def dvx(theta, dot_theta, dot_phi, omega, vx, vy, gn):
    part1 = R*np.sin(theta)/I1*(dot_phi*omega*(I3*(1-alpha*np.cos(theta))-I1) + gn*R*alpha*(1-alpha*np.cos(theta)) - I1*alpha*(dot_theta**2+dot_phi**2*np.sin(theta)**2))
    part2 = -mu*gn*vx/m/I1*(I1 + m*R**2*(1-alpha*np.cos(theta))**2) + dot_phi*vy
    solution = part1 + part2
    return solution
    
def dvy(theta, dot_theta, dot_phi, omega, vx, vy, gn):
    part1 = -mu*gn*vy/(m*I1*I3)*(I1*I3 + m*R**2*I3*(alpha-np.cos(theta))**2 + m*R**2*I1*np.sin(theta)**2)
    part2 = omega*dot_theta*R/I1*(I3*(alpha-np.cos(theta)) + I1*np.cos(theta)) - dot_phi*vx
    solution = part1 + part2
    return solution

def f_gn(theta, dot_theta, dot_phi, omega, vx, vy):
    upper = m*g*I1 + m*R*alpha*(np.cos(theta)*(I1*dot_phi**2*np.sin(theta)**2 + I1*dot_theta**2) - I3*dot_phi*omega*np.sin(theta)**2)
    bottom = I1 + m*R**2*alpha**2*np.sin(theta)**2 - m*R**2*alpha*np.sin(theta)*(1-alpha*np.cos(theta))*mu*vx
    solution = float(upper)/bottom
    return solution

def Energy(theta, dot_theta, dot_phi, omega):
    part1 = 0.5*(I1*dot_phi**2*np.sin(theta)**2 + I1*dot_theta**2 + I3*omega**2)
    part2 = m*g*R*(1-np.cos(theta))
    part3 = 0.5*m*R**2*((alpha-np.cos(theta))**2*(dot_theta**2+dot_phi**2*np.sin(theta)**2) + np.sin(theta)**2*(dot_theta**2 + omega**2 + 2*omega*dot_phi*(alpha-np.cos(theta))))
    solution = part1 + part2 + part3
    return solution

def ODE(x ,t):
    gn = f_gn(x[0], x[1], x[2], x[3], x[4], x[5])
    dx0 = x[1]
    dx1 = ddtheta(x[0], x[1], x[2], x[3], x[4], x[5], gn)
    dx2 = ddphi(x[0], x[1], x[2], x[3], x[4], x[5], gn)
    dx3 = domega(x[0], x[1], x[2], x[3], x[4], x[5], gn)
    dx4 = dvx(x[0], x[1], x[2], x[3], x[4], x[5], gn)
    dx5 = dvy(x[0], x[1], x[2], x[3], x[4], x[5], gn)
    return np.array([dx0, dx1, dx2, dx3, dx4, dx5])
