# Code to graph solutions together saved by simpleDTQ.py using pickle.

import numpy as np
import matplotlib.pyplot as plt
import pickle

courseSolution = open('CoarseSolution', 'rb')
pdf_trajectory = pickle.load(courseSolution)
xvals = open('CoarseX', 'rb')
xvec = pickle.load(xvals)

fineSolution = open('FineSolution', 'rb')
pdf_trajectoryfine = pickle.load(fineSolution)
xvalsfine = open('FineX', 'rb')
xvecfine = pickle.load(xvalsfine)

plt.figure()
plt.suptitle(r'Evolution for $f(x)=\tan(x), g(x)=1, k \approx 0.032$ vs. $k = 0.01$')
numPDF = np.size(pdf_trajectory, 1)
plt.subplot(2, 2, 1)
plt.title("t=0")
plt.plot(xvec, pdf_trajectory[:, 0])
plt.plot(xvecfine, pdf_trajectoryfine[:, 0])
plt.subplot(2, 2, 2)
plt.title("t=T/3")
plt.plot(xvec, pdf_trajectory[:, int(np.ceil(numPDF * (1 / 3)))])
plt.plot(xvecfine, pdf_trajectoryfine[:, int(np.ceil(numPDF * (1 / 3)))])
plt.subplot(2, 2, 3)
plt.title("t=2T/3")
plt.plot(xvec, pdf_trajectory[:, int(np.ceil(numPDF * (2 / 3)))])
plt.plot(xvecfine, pdf_trajectoryfine[:, int(np.ceil(numPDF * (2 / 3)))])
plt.subplot(2, 2, 4)
plt.title("t=T")
plt.plot(xvec, pdf_trajectory[:, int(np.ceil(numPDF - 1))])
plt.plot(xvecfine, pdf_trajectoryfine[:, int(np.ceil(numPDF - 1))])
plt.show()


