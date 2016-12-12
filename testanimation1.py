import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def trans(x,y,z):
    X = y- np.sqrt(2)/4.0*x
    Y = z- np.sqrt(2)/4.0*x
    return X, Y

theta = np.load("theta.npy")

phi = np.load("phi.npy")

fig, ((ax)) = plt.subplots(nrows=1, ncols=1)
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_aspect('equal')


line1, = ax.plot([], [], 'b--', lw=2)
line2,= ax.plot([], [], 'b--', lw=2)
line3,= ax.plot([], [], 'b--', lw=2)
line4,= ax.plot([], [], 'r.', lw=2)
line5,= ax.plot([], [], 'k', lw=1)
line6,= ax.plot([], [], 'k', lw=1)
line62,= ax.plot([], [], 'k--', lw=1)
line7,= ax.plot([], [], 'k', lw=1)
line72,= ax.plot([], [], 'k--', lw=1)
line8,= ax.plot([], [], 'g', lw=1)
line82,= ax.plot([], [], 'g--', lw=1)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    line5.set_data([], [])
    line6.set_data([], [])
    line7.set_data([], [])
    line8.set_data([], [])
    line62.set_data([], [])
    line72.set_data([], [])
    line82.set_data([], [])
    return line1, line2, line3, line4, line5, line6, line7, line62, line72, line8, line82

def animate(i):
    angle = np.linspace(0, 2*np.pi, 1000)
    angle1 = np.linspace(-0.5*np.pi, 0.5*np.pi, 1000)
    angle2 = np.linspace(0.5*np.pi, 1.5*np.pi, 1000)
    l1 = np.linspace(0, 0, 1000)
    l2 = np.linspace(-1.5, 1.5, 1000)
    line1.set_data(l1, l2)
    line2.set_data(l2, l1)
    l3 = np.linspace(-0.375*np.sqrt(2), 0.375*np.sqrt(2), 1000)
    line3.set_data(l3, l3)
    point_X, point_Y = trans(np.sin(theta[i])*np.cos(phi[i]), np.sin(theta[i])*np.sin(phi[i]), np.cos(theta[i]))
    X = np.linspace(-point_X, point_X, 1000)
    Y = np.linspace(-point_Y, point_Y, 1000)
    line4.set_data(X, Y)
    #print(i)
    line5.set_data(np.cos(angle), np.sin(angle))
    line6.set_data(trans(np.cos(angle1), np.sin(angle1), 0)[0], trans(np.cos(angle1), np.sin(angle1), 0)[1])
    line62.set_data(trans(np.cos(angle2), np.sin(angle2), 0)[0], trans(np.cos(angle2), np.sin(angle2), 0)[1])
    line7.set_data(trans(np.cos(angle1), 0, np.sin(angle1))[0], trans(np.cos(angle1), 0, np.sin(angle1))[1])
    line72.set_data(trans(np.cos(angle2), 0, np.sin(angle2))[0], trans(np.cos(angle2), 0, np.sin(angle2))[1])
    line8.set_data(trans(np.sin(theta[i])*np.cos(angle1), np.sin(theta[i])*np.sin(angle1), np.cos(theta[i]))[0], trans(np.sin(theta[i])*np.cos(angle1), np.sin(theta[i])*np.sin(angle1), np.cos(theta[i]))[1])
    line82.set_data(trans(np.sin(theta[i])*np.cos(angle2), np.sin(theta[i])*np.sin(angle2), np.cos(theta[i]))[0], trans(np.sin(theta[i])*np.cos(angle2), np.sin(theta[i])*np.sin(angle2), np.cos(theta[i]))[1])
    return line1, line2, line3, line4, line5, line6, line7, line62, line72, line8, line82

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=20000, interval=0.001, blit=True)

