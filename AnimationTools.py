import numpy as np
import matplotlib.pyplot as plt
import Integrand


def axis_setup(xvec_trajectory, pdf_trajectory):
    NeedToChangeYAxes = True
    NeedToChangeXAxes = True
    MinX = min([(min(a)) for a in xvec_trajectory])
    MaxX = max([(max(a)) for a in xvec_trajectory])
    MaxY = max([(max(a)) for a in pdf_trajectory])
    starting_minxgrid = np.floor(np.min(xvec_trajectory[0]))
    starting_maxxgrid = np.ceil(np.max(xvec_trajectory[0]))
    starting_maxygrid = np.ceil(np.max(pdf_trajectory[0]))
    diff = 3
    if abs(MaxY - starting_maxygrid) < diff:
        starting_maxygrid = MaxY
        NeedToChangeYAxes = False
    if (abs(MaxX - starting_maxxgrid) < diff) & (abs(MinX - starting_minxgrid) < diff):
        starting_maxxgrid = MaxX
        starting_minxgrid = MinX
        NeedToChangeXAxes = False
    return NeedToChangeXAxes, NeedToChangeYAxes, starting_minxgrid, starting_maxxgrid, starting_maxygrid


def update_y_axis(pdf_trajectory, step, starting_minxgrid,
                  starting_maxxgrid, starting_maxygrid, l):
    diff = 10
    My = np.ceil(np.max(pdf_trajectory))
    if (np.abs(My - starting_maxygrid)) > diff:
        l.set_ylim(0, My)
        starting_maxygrid = My
    if My > starting_maxygrid:
        l.set_ylim(0, My)
        starting_maxygrid = My

    return starting_minxgrid, starting_maxxgrid, starting_maxygrid


def update_x_axis(xvec_trajectory, step, starting_minxgrid,
                  starting_maxxgrid, starting_maxygrid, l):
    diff = 10
    mx = np.floor(np.min(xvec_trajectory[step]))
    Mx = np.ceil(np.max(xvec_trajectory[step]))
    if (mx < starting_minxgrid) & (Mx > starting_maxxgrid):
        l.set_xlim(mx, Mx)
        starting_minxgrid = mx
        starting_maxxgrid = Mx
    elif (starting_minxgrid - mx > diff) | (starting_maxxgrid - Mx > diff):
        l.set_xlim(mx, Mx)
        starting_minxgrid = mx
        starting_maxxgrid = Mx
    elif mx < starting_minxgrid:
        l.set_xlim(mx, starting_maxxgrid)
        starting_minxgrid = mx
    elif Mx > starting_maxxgrid:
        l.set_xlim(starting_minxgrid, Mx)
        starting_maxxgrid = Mx
    return starting_minxgrid, starting_maxxgrid, starting_maxygrid


def update_animation(step, pdf_trajectory, l, xvec_trajectory, im,  NeedToChangeXAxes, NeedToChangeYAxes, starting_minxgrid, starting_maxxgrid, starting_maxygrid):
    if step == 0:
        plt.xlim(starting_minxgrid, starting_maxxgrid)
        plt.ylim(0, starting_maxygrid)

    im.set_xdata(xvec_trajectory[step])
    im.set_ydata(pdf_trajectory[step])
    if NeedToChangeXAxes: update_x_axis(xvec_trajectory, step, starting_minxgrid,
                                        starting_maxxgrid, starting_maxygrid, l)

    if NeedToChangeYAxes: update_y_axis(pdf_trajectory[step], step, starting_minxgrid,
                                        starting_maxxgrid, starting_maxygrid, l)
    return im


def update_animation_integrand(step, G_history, l, xvec_trajectory, pdf_trajectory, im, NeedToChangeXAxes, NeedToChangeYAxes, starting_minxgrid, starting_maxxgrid, starting_maxygrid):
    if step == 0:
        plt.xlim(starting_minxgrid, starting_maxxgrid)
        plt.ylim(0, starting_maxygrid)

    NeedToChangeYAxes = True
    integrand = Integrand.calculateIntegrand(G_history[step], pdf_trajectory[step])

    Y = np.zeros([np.size(xvec_trajectory[step]), np.size(integrand, 1)])
    for i in range(np.size(integrand, 1)):
        Y[i, :] = xvec_trajectory[step]
    im.set_xdata(Y)
    im.set_ydata(integrand)

    if NeedToChangeXAxes: update_x_axis(xvec_trajectory, step, starting_minxgrid,
                  starting_maxxgrid, starting_maxygrid, l)

    if NeedToChangeYAxes: update_y_axis(integrand, step, starting_minxgrid,
                                        starting_maxxgrid, starting_maxygrid, l)
    return im
