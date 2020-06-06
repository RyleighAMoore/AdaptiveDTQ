2D Method

The main code for the 2D method is located at 2DTQ-UnorderedMeshWithMeshUpdates.py.
The drift and diffustion for the code is set in the Function.py file. You can
select from the options given or define your own. 
The current code has mesh updates where we add and remove points at the boundary.


Untested/Needs to be implemented List:
    1. Using diffusion that depends on (x,y) is currently untested with this version of the code. 

 
    2. Updates to change the density of the mesh interior are not working currently 
        and are turned off right now in the code.
        
    3. h should be 0.01 right now. May require code adjustments to use different values.
    
    4. Currently using candidate samples of 7*np.random.normal(0, 1, (num_vars, n)) for Leja points.
        Not sure if this is best. Need to test sqrt(2)*np.random.normal(0, 1, (num_vars, n)), etc.
        
        
        
Other good files to know:
    Function.py - Change drift and diffusion functions here
    
    LejaQuadrature.py - Called from 2DTQ-UnorderedMeshWithMeshUpdates.py to step the solution forward in time.
    
    MeshUpdates2D.py - Contains the mesh update procedures
    
    pyopoly1/QuadratureRules.py - Contains the Leja quadrature rules
    
    QuadraticFit.py - Contains the code for the Quadratic fit method to divide out gaussian from integrand.
    
    pyopoly1/LejaPoints.py - Methods to generate Leja Points. Uses code in pyopoly1/LejaUtilities.py
    
    

WARNING:
The files for the 1D method are mixed into this code as well. I should move them at some point to be separate.
For example, setupMethods.py and QuadRules.py are for the 1D method.
    
    
    
    

 



