'''
# 2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()

'''

# 3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


X = np.load('/Users/heebinyoo/Desktop/Xcoord790.npy')

Y = np.load('/Users/heebinyoo/Desktop/Ycoord790.npy')

Z = np.load('/Users/heebinyoo/Desktop/Zcoord790.npy')

XL = []
YL = []
ZL = []
x, y, z = 0, 0, 0

for p, q, r in zip(X, Y, Z):
    x = x + p
    y = y + q
    z = z + r
    XL.append(x)
    YL.append(y)
    ZL.append(z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xdata, ydata, zdata = [], [], []
ln = ax.scatter(xdata, ydata, zdata, 'ro')

def init():
    ax.set_xlim(-110, 20)
    ax.set_ylim(0, 90)
    ax.set_zlim(-10, 8000)
    return ln,

def update(frame):
    xdata.append(XL[frame])
    ydata.append(YL[frame])
    zdata.append(ZL[frame])
    ln._offsets3d = (xdata, ydata, zdata)
    return ln,

ani = FuncAnimation(fig, update, frames=790,
                    init_func=init, blit=True)
plt.show()
ani.save('/Users/heebinyoo/Desktop/animation.gif', fps=10)

