from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Functions as fun
import Integrand
import Operations2D
import XGrid
from mpl_toolkits.mplot3d import Axes3D
import QuadRules
from tqdm import tqdm, trange
import random
import UnorderedMesh as UM
from scipy.spatial import Delaunay
import MeshUpdates2D as MeshUp
import pickle
import multiprocessing as mp

def f(point):
    return UM.StepForward(Vertices,point,VerticesNum, GMat, Grids, pdf, pdfNew)
       

if __name__ == '__main__':   

    T = 0.01  # final time, code computes PDF of X_T
    s = 0.75  # the exponent in the relation k = h^s
    h = 0.01  # temporal step size
    init = 0  # initial condition X_0
    numsteps = int(np.ceil(T / h))
    
    assert numsteps > 0, 'The variable numsteps must be greater than 0'
    
    # define spatial grid
    kstep = h ** s
    kstep = 0.15
    xmin=-2
    xmax=2
    ymin=-2
    ymax=2
    h=0.01
    
    
    #def loopNewPDf(Px, Py, grid, kstep, h, interpPDF):
    #    val = 0
    #    for i in range(len(interpPDF)):
    #        val = val + kstep**2*fun.G(Px, Py, grid[i,0], grid[i,1], h)*interpPDF[i]
    #    return val
    
    mesh = UM.generateOrderedGridCenteredAtZero(-0.5, 0.5, -0.5, 0.5, kstep)      # ordered mesh  
    #x = np.arange(-0.2, 0.2, .01)
    #y = np.arange(-0.2, 0.2, .01)
    #xx, yy = np.meshgrid(x, y)
    #points = np.ones((np.size(xx),2))
    #points[:,0]= xx.ravel()
    #points[:,1]=yy.ravel()
    #tri1 = Delaunay(points).simplices
    #
    #
    #w = np.arange(0,len(x)-1,1)
    ##w=np.sort(np.concatenate((w,w)))
    #cSimplices = []
    #for j in range(len(x)-1):
    #    for i in w:
    #        #cSimplices.append([i+j*len(x),i+1+j*len(x),i+len(x)+j*len(x)])
    #        cSimplices.append([i+j*len(x),i+1+j*len(x),i+len(x)+1+j*len(x)])
    #
    #v = np.arange(len(x),len(x)*2-1,1)
    #for j in range(0,len(x)-1):
    #    for i in v:
    #        cSimplices.append([i+j*len(x),i+1+j*len(x),i-len(x)+j*len(x)])
    #                
    #
    #s = np.asarray(cSimplices, dtype='int32')
    #tri = Delaunay(points)
    #tri.simplices = s
    #mesh = points
    #mesh = UM.generateRandomPoints(-0.2,0.2,-0.2,0.2,200)  # unordered mesh
    
    #rad =  np.sqrt(mesh[:,0]**2 + mesh[:,1]**2)
    #for i in range(len(rad)-1,-1,-1):
    #    if rad[i] > 1.8:
    #        mesh = np.delete(mesh,i,0)
    
    #circle_r = 0.7
    ## center of the circle (x, y)
    #circle_x = 0
    #circle_y = 0
    #mesh2 = []
    #for r in range(100):
    #    # random angle
    #    alpha = 2 * math.pi * random.random()
    #    # random radius
    #    r = circle_r * math.sqrt(random.random())
    #    # calculating coordinates
    #    x = r * math.cos(alpha) + circle_x
    #    y = r * math.sin(alpha) + circle_y
    #    mesh2.append([x, y])
    
    #mesh2 = UM.generateOrderedGrid(-0.25, 0.25, -0.25, 0.25, 0.08)      # ordered mesh  
    #mesh = np.vstack((mesh,mesh2))
    
    #mesh = np.vstack((mesh,[0,0]))
    #mesh = np.vstack((mesh,[0.05,0.05]))
    #mesh = np.vstack((mesh,[-0.05,0.05]))
    #mesh = np.vstack((mesh,[0.05,-0.05]))
    #mesh = np.vstack((mesh,[-0.05,-0.05]))
    
    #pdf = np.zeros(len(mesh))
    #mesh2 = mesh2
    #mesh = np.vstack((mesh,mesh2))
    
    pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], 0.1, 0.1)
    #pdf = phat_rav
    #pdf = np.zeros(len(mesh))
    #pdf[-1]=10
    #pdf[1000]=10
    
    
    Meshes = []
    PdfTraj = []
    PdfTraj.append(np.copy(pdf))
    Meshes.append(np.copy(mesh))
    GMat = []
    Grids = []
    Vertices = []
    VerticesNum = []
    #
    tri = Delaunay(mesh, incremental=True)
    #tri.simplices=s
    #order2 = []
    for point in trange(len(mesh)):
        grid = UM.makeOrderedGridAroundPoint([mesh[point,0],mesh[point,1]],kstep, 2*max(xmax-xmin, ymax-ymin), xmin,xmax,ymin,ymax)
        Grids.append(np.copy(grid))
        Vertices.append([])
        VerticesNum.append([])
        for currGridPoint in range(len(grid)):
            vertices, indices = UM.getVerticesForPoint([grid[currGridPoint,0], grid[currGridPoint,1]], mesh, tri) # Points that make up triangle
            Vertices[point].append(np.copy(vertices))
            VerticesNum[point].append(np.copy(indices))
        gRow = MeshUp.generateGRow([mesh[point,0], mesh[point,1]], grid, kstep, h)
        GMat.append(np.copy(gRow))
    
    
       
    pdf = np.copy(PdfTraj[-1])
    adjustGrid = True
    for i in trange(1):
        if (i >= 1) and adjustGrid:
            #GridAdjustments  
            mesh, GMat, Grids, Vertices, VerticesNum, pdf, tri = MeshUp.PerformGridUpdates(GMat, mesh, Grids, Vertices, VerticesNum, pdf, tri, kstep, h, xmin, xmax, ymin, ymax)
        pdfNew = np.copy(pdf)                   
        print("stepping forward...")
        #for point in range(len(mesh)):
        
        pdfNew = mp.Pool(2).map(f,list(range(len(mesh)))) 
        
        PdfTraj.append(np.copy(pdfNew))
        Meshes.append(np.copy(mesh))
        pdf=pdfNew
        print('Length of mesh = ', len(mesh))
        
    ##Use
    #fig = plt.figure()
    #ax = Axes3D(fig)
    #ax.scatter(grid[:,0], grid[:,1], np.asarray(interpPdf), c='r', marker='.')
    #            
    #Use to check triangularization
    #plt.plot(mesh[:,0], mesh[:,1], '.k')
    #plt.plot(vertices[0,0], vertices[0,1], '.g', markersize=14)
    #plt.plot(vertices[1,0], vertices[1,1], '.g',  markersize=14)
    #plt.plot(vertices[2,0], vertices[2,1], '.g',  markersize=14)
    #plt.plot(Px, Py, '.r', markersize=14)
    #plt.show()
    ##
    #fig = plt.figure()
    #ax = Axes3D(fig)
    #ax.scatter(grid[:,0], grid[:,1], interpPdf, c='r', marker='.')
    
    fig = plt.figure()
    ax = Axes3D(fig)
    index = 100
    ax.scatter(Meshes[index][:,0], Meshes[index][:,1], PdfTraj[index], c='r', marker='.')
    #    
    #    
    def update_graph(num):
        graph.set_data (Meshes[num][:,0], Meshes[num][:,1])
        graph.set_3d_properties(PdfTraj[num])
        title.set_text('3D Test, time={}'.format(num))
        return title, graph
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')
        
    graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker="o")
    ax.set_zlim(0, np.max(PdfTraj[2]))
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj),
                                             interval=1000, blit=False)
    
    plt.show()



#pickle.dump(PdfTraj, open( "PDF.p", "wb" ) )
#pickle.dump(Meshes, open( "Meshes.p", "wb" ) )

#gg = 0
##inds = np.asarray(list(range(0, np.size(x1)*np.size(x2))))
##phat_rav = np.ravel(PdfTraj[gg])
#
#
#si = int(np.sqrt(len(Meshes[gg][:,0])))
##inds_unrav = np.unravel_index(inds, (si, si))
#
#pdfgrid = np.zeros((si,si))
#for i in range(si):
#    for j in range(si):
#        pdfgrid[i,j]=PdfTraj[0][i+j]
