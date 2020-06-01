# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:03:32 2020

@author: Rylei
"""
import pickle

pkl_file= open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/ICMesh1.p", "wb" ) 
pickle.dump(mesh, pkl_file)
pkl_file.close()


# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/ICMesh1.p", "rb" ) 
# mesh = pickle.load(pkl_file)

timestr = time.strftime("%Y%m%d-%H%M%S")
#
# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PdfTrajBimodal-"+ timestr+".p", "wb" ) 
# pkl_file2 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/MeshesBimodal-"+ timestr+".p", "wb" ) 

# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PdfTrajLQTwoHillLongFullSplit.p", "wb" ) 
# pkl_file2 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/MeshesLQTwoHillLongFullSplit1.p", "wb" ) 

# #    
# pickle.dump(PdfTraj, pkl_file)
# pickle.dump(Meshes, pkl_file2)
# pkl_file.close()
# pkl_file2.close()

#pickle_in = open("C:/Users/Rylei/SyderProjects/SimpleDTQGit/PickledData/PDF.p","rb")
#PdfTraj = pickle.load(pickle_in)
#
#pickle_in = open("C:/Users/Rylei/SyderProjects/SimpleDTQGit/PickledData/Meshes.p","rb")
#Meshes = pickle.load(pickle_in)
#
# #
# import pickle
# pickle_in = open("C:/Users/Rylei/SyderProjects/SimpleDTQGit/PickledData/PdfTraj-20200205-191728.p","rb")
# PdfTraj2 = pickle.load(pickle_in)

# pickle_in = open("C:/Users/Rylei/SyderProjects/SimpleDTQGit/PickledData/Meshes-20200205-191728.p","rb")
# Meshes2 = pickle.load(pickle_in)



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

# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PDFTrajDenseMovingHill.p", "wb" ) 
# pkl_file2 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/MeshDenseMovingHill.p", "wb" ) 

# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PdfTrajV.p", "wb" ) 
# pkl_file2 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/MeshesV.p", "wb" ) 

# #    
# pickle.dump(PdfTraj, pkl_file)
# pickle.dump(Meshes, pkl_file2)
# pkl_file.close()
# pkl_file2.close()

# Generate data...


# plt.scatter(Meshes[1][:,0], Meshes[1][:,1],c=np.log(PdfTraj[1]))
# plt.show()
