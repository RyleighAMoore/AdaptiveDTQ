# code to compute the PDF of the solution of the SDE:
#
# dX_t = X_t*(4-X_t^2) dt + dW_t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Integrand
import AnimationTools
import GMatrix
import XGrid
import QuadRules
import setupMethods as setup
import Functions as fun


machEps = np.finfo(float).eps

# visualization parameters ###################################################
animate = True
finalGraph = True

# tolerance parameters
epsilonTolerance = -10
minSizeGAndStillRemoveValsFromG = 100
minMaxOfPhatAndStillRemoveValsFromG = 0.001

# simulation parameters
autoCorrectInitialGrid = True
RandomXvec = True  # if autoCorrectInitialGrid is True this has no effect.

RemoveFromG = True  # Also want AddToG to be true if true
IncGridDensity = True
DecGridDensity = True
AddToG = True

T = 1.0  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
k = h ** s
xMin = -2
xMax = 2
##############################################################################

pdf_trajectory = []
xvec_trajectory = []
epsilonArray = []
G_history = []
steepnessArr = []
kvec_trajectory = []
diff = []

a = init + fun.driftfun(init)
b = np.abs(fun.difffun(init)) * np.sqrt(h)

if not autoCorrectInitialGrid: #Assuming initial grid is sufficient to capture the pdf.
    xvec, phat, G = setup.setupNonCorrectedGrid(xMin, xMax, k, a, b, h)

if autoCorrectInitialGrid:
    xvec, k, phat = setup.correctInitialGrid(xMin, xMax, a, b, k)
    G = GMatrix.computeG(xvec, xvec, h)


epsilonArray.append(Integrand.computeEpsilon(G, phat))
pdf_trajectory.append(phat)  # solution after one time step from above
xvec_trajectory.append(xvec)
G_history.append(G)

countSteps = 0
while countSteps < numsteps - 1:  # since one time step is computed above
    countSteps, G, AddToG, pdf_trajectory, xvec_trajectory, IncGridDensity, G_history, epsilonTolerance, epsilonArray, init, kvec_trajectory, k, h = XGrid.stepForwardInTime(countSteps, G, AddToG, pdf_trajectory, xvec_trajectory,IncGridDensity, G_history, epsilonTolerance, epsilonArray, init, kvec_trajectory, k, h)

    # Animate the PDF
if animate:
    f1 = plt.figure()
    l = f1.add_subplot(1, 1, 1)
    im, = l.plot([], [], 'r', markersize=1)
    NeedToChangeXAxes, NeedToChangeYAxes, starting_minxgrid, starting_maxxgrid, starting_maxygrid = AnimationTools.axis_setup(
        xvec_trajectory, pdf_trajectory)
    anim = animation.FuncAnimation(f1, AnimationTools.update_animation, len(xvec_trajectory),
                                   fargs=(
                                       pdf_trajectory, l, xvec_trajectory, im, NeedToChangeXAxes, NeedToChangeYAxes,
                                       starting_minxgrid, starting_maxxgrid, starting_maxygrid), interval=50,
                                   blit=False)
    plt.show()


if finalGraph:
    assert animate == True, 'The variable animate must be True'
    plt.plot(xvec_trajectory[-1], pdf_trajectory[-1], '.')
    plt.show()

