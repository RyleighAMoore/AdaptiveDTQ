import numpy as np
import Functions as fun
import UnorderedMesh as UM

def getICMesh(radius, stepSize, h):
    meshSpacing = stepSize #DM.separationDistance(mesh)*2
    grid = UM.generateOrderedGridCenteredAtZero(-radius*2, radius*2, -radius*2, radius*2, meshSpacing , includeOrigin=True)
    # noise = np.random.normal(0,1, size = (len(grid),2))
    
    # noise = np.random.uniform(-meshSpacing, meshSpacing,size = (len(grid),2))
    
    # shake = 0
    # noise = -meshSpacing*shake +(meshSpacing*shake - - meshSpacing*shake)/(np.max(noise)-np.min(noise))*(noise-np.min(noise))
    # noiseGrid = grid+noise
    
    x,y = grid.T
    X = []
    Y = []
    for point in range(len(grid)):
        if np.sqrt(x[point]**2 + y[point]**2) < radius:
            X.append(x[point])
            Y.append(y[point])
    
    newGrid = np.vstack((X,Y))
    x,y = newGrid

    return newGrid.T


if __name__ == "__main__":
    newGrid = getICMesh(0.5, 0.1, 0.01)