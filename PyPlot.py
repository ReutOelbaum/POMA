import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np


def plot_3d_given_param( a,b, c, d, e, f):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    #Z1 = np.sqrt(a*X**2 + b*Y**2)#*np.exp(-2*X**2 + -1*Y**2)

    Z2 = (a*(X-c)*(X-c)+b*(Y-d)*(Y-d)) *np.exp(-e* X ** 2 - f* Y ** 2)
    # Plot the surface.
    #surf = ax.plot_surface(X, Y, Z1, cmap='summer',
     #                      linewidth=0, antialiased=False)

    surf = ax.plot_surface(X, Y, Z2, cmap='Blues',
                           linewidth=0, antialiased=False)


    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


    #add indibudual dots
    #ax.scatter(2,3,1,marker='^', c='r') # plot the point (2,3,4) on the figure
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

real_param = [6.23, 5.11, 1, 1, 1, 1]
plot_3d_given_param(real_param[0],real_param[1],real_param[2],real_param[3],real_param[4],real_param[5])