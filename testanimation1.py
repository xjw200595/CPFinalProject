import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

def trans(x,y,z):
    X = y- np.sqrt(2)/4.0*x
    Y = z- np.sqrt(2)/4.0*x
    return X, Y

theta = np.load("theta.npy")

phi = np.load("phi.npy")

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2)).set_aspect('equal')

line1, = ax.plot([], [], '--', lw=2)
'''
line2,= ax.plot([], [], '--', lw=2)
line3,= ax.plot([], [], '--', lw=2)
line4,= ax.plot([], [], '.', lw=2)
line5,= ax.plot([], [], lw=2)
line6,= ax.plot([], [], '.', lw=2)
line7,= ax.plot([], [], '.', lw=2)

 #initialization function: plot the background of each frame
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return line1, line2, line3, line4, line5, line6, line7

# animation function.  This is called sequentially
def animate(i):
    angle = np.linspace(0, 2*np.pi, 1000)
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
    line6.set_data(trans(np.cos(angle), np.sin(angle), 0)[0], trans(np.cos(angle), np.sin(angle), 0)[1])
    line7.set_data(trans(np.cos(angle), 0, np.sin(angle))[0], trans(np.cos(angle), 0, np.sin(angle))[1])
    
    return line1, line2, line3, line4, line5, line6, line7

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=20000, interval=0.001, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html


plt.show()
'''