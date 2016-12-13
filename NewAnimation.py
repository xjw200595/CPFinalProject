import numpy as np
import matplotlib.pyplot as plt

theta = np.load("theta.npy")
phi = np.load("phi.npy")
x_position = np.load("x_position.npy")
y_position = np.load("y_position.npy")

def trans(x,y,z):
    X = y- np.sqrt(2)/4.0*x
    Y = z- np.sqrt(2)/4.0*x
    return X, Y

N = 20
for i in np.arange(0,N,10):
    fig, ((ax1)) = plt.subplots(nrows=1, ncols=1)
    
    angle = np.linspace(0, 2*np.pi, 1000)
    angle1 = np.linspace(-0.5*np.pi, 0.5*np.pi, 1000)
    angle2 = np.linspace(0.5*np.pi, 1.5*np.pi, 1000)
    l1 = np.linspace(0, 0, 1000)
    l2 = np.linspace(-1.5, 1.5, 1000)
    ax1.plot(l1, l2, 'b--', lw=1)#y
    ax1.plot(l2, l1, 'b--', lw=1)#z
    l3 = np.linspace(-0.375*np.sqrt(2), 0.375*np.sqrt(2), 1000)
    ax1.plot(l3, l3, 'b--', lw=1)#x
    point_X, point_Y = trans(np.sin(theta[i])*np.cos(phi[i]), np.sin(theta[i])*np.sin(phi[i]), np.cos(theta[i]))
    X = np.linspace(-point_X, point_X, 1000)
    Y = np.linspace(-point_Y, point_Y, 1000)
    ax1.plot(X, Y, 'r', lw=2)#point
    #print(i)
    ax1.plot(np.cos(angle), np.sin(angle), 'k', lw=1)#x circle
    ax1.plot(trans(np.cos(angle1), np.sin(angle1), 0)[0], trans(np.cos(angle1), np.sin(angle1), 0)[1], 'k', lw=1)#z circle
    ax1.plot(trans(np.cos(angle2), np.sin(angle2), 0)[0], trans(np.cos(angle2), np.sin(angle2), 0)[1], 'k--', lw=1)
    ax1.plot(trans(np.cos(angle1), 0, np.sin(angle1))[0], trans(np.cos(angle1), 0, np.sin(angle1))[1], 'k', lw=1)#y circle
    ax1.plot(trans(np.cos(angle2), 0, np.sin(angle2))[0], trans(np.cos(angle2), 0, np.sin(angle2))[1], 'k--', lw=1)
    ax1.plot(trans(np.sin(theta[i])*np.cos(angle1), np.sin(theta[i])*np.sin(angle1), np.cos(theta[i]))[0], trans(np.sin(theta[i])*np.cos(angle1), np.sin(theta[i])*np.sin(angle1), np.cos(theta[i]))[1], 'g', lw=1)#small circle
    ax1.plot(trans(np.sin(theta[i])*np.cos(angle2), np.sin(theta[i])*np.sin(angle2), np.cos(theta[i]))[0], trans(np.sin(theta[i])*np.cos(angle2), np.sin(theta[i])*np.sin(angle2), np.cos(theta[i]))[1], 'g--', lw=1)
    
    string = "figure_test/image_{0:06d}.png".format(i)#第一个0是第一个format的变量
    #冒号之后的东西是format的形式。d是整数，04是至少4位，0开头。
    fig.savefig(string)
    plt.close(fig)