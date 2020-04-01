# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:37:13 2020

@author: Ryleigh Moore
"""
from __future__ import print_function
from ortools.linear_solver import pywraplp
import numpy as np
# Instantiate a Glop solver, naming it Linear.
solver = pywraplp.Solver('Linear', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

# Create the two variables and let them take on any non-negative value.
ts1 = solver.NumVar(0, solver.infinity(), 'ts1')
ts2 = solver.NumVar(0, solver.infinity(), 'ts2')

# Variables
tE = 168
h1 = 12
h2 = 12
LE = 1*10**5
vp = 300
vs = 8
#tA = tE-ts1-2*h
tAmin = 20 # Not sure about this value
Np0 = 1

# Constraint g1: -ts1 > TAmin + h1 + h2 - tE
constraintg1 = solver.Constraint(tAmin + h1 + h2 - tE, solver.infinity())
constraintg1.SetCoefficient(ts1, -1)
constraintg1.SetCoefficient(ts2, 0)


# Constraint g2: ts1 + h1 < ts2 - > h1 < ts2-ts1
constraintg1 = solver.Constraint(h1, solver.infinity())
constraintg1.SetCoefficient(ts1, -1)
constraintg1.SetCoefficient(ts2, 1)

# Constraint g3: ts1 - ts2 < 0
constraintg1 = solver.Constraint(-solver.infinity(), 0)
constraintg1.SetCoefficient(ts1, 1)
constraintg1.SetCoefficient(ts2, -1)

# Constraint g2: ts2 + h2 < tE ->  ts2 < tE - h2
constraintg1 = solver.Constraint(tE-h2, solver.infinity())
constraintg1.SetCoefficient(ts1, 0)
constraintg1.SetCoefficient(ts2, 1)



alphaE = (tE*0.69)/(np.log((LE*vp)/(Np0)))
BetaE = alphaE/tE

alphaD = 

(tE/(2))

# Objective function 
objective = solver.Objective()
objective.SetCoefficient(x, 3)
objective.SetCoefficient(y, 4)
objective.SetMaximization()



