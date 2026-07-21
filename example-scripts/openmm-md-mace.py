#!/usr/bin/env python
from __future__ import division, print_function
import os, sys
import numpy as np
from openmm.app import *
from openmm import *
from openmmml import MLPotential
import openmm.unit as u
from openmmplumed import *

##### Simulation system ##### 
# Create a system in openmm
prmtop_file = f'*.prmtop'
crd_file = f'*.rst'
prmtop = AmberPrmtopFile(prmtop_file)
inpcrd = AmberInpcrdFile(crd_file)

# load the model and create system  
model_file = '*.model'
potential = MLPotential('mace', modelPath=model_file)
system = potential.createSystem(prmtop.topology)

# Set different force groups
#for i, f in enumerate(system.getForces()):
#    print(f.getForceGroup())
#    f.setForceGroup(i)


##### Add constraints to the ML water molecules #####
d_OH = 0.0978882  #OPC3 geometry
d_HH = 0.1598507

# get water IDs 
start_id = 2  # atomID of first water atom 
num_molecules = 70
water_molecules = [
    [i, i + 1, i + 2]
    for i in range(start_id, start_id + num_molecules * 3, 3)]
water_molecules = [[x - 1 for x in sublist] for sublist in water_molecules]  # Following treatment is because OpenMM starts from 0 index

added_constraints = 0
for water in water_molecules:
    O_idx, H1_idx, H2_idx = water

    system.addConstraint(O_idx, H1_idx, d_OH)
    system.addConstraint(O_idx, H2_idx, d_OH)
    system.addConstraint(H1_idx, H2_idx, d_HH)
    added_constraints += 3

print(f'Added {added_constraints} rigid constraints to ML water molecules.')


##### Set up integrator and simulation #####
integrator = LangevinIntegrator(300.0*u.kelvin, 2.0/u.picosecond,
                    0.002*u.picosecond)
integrator.setConstraintTolerance(1e-05)
platform = Platform.getPlatformByName("CUDA")
properties = {'Precision': 'mixed'}
simulation = Simulation(prmtop.topology, system, integrator, platform, properties)

# Set the positions and box vectors
simulation.context.setPositions(inpcrd.positions)
if inpcrd.boxVectors is not None: 
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

#simulation.context.setPeriodicBoxVectors(*parm.box_vectors)

# Print different force groups
for i, f in enumerate(system.getForces()):
    state = simulation.context.getState(getEnergy=True, groups={i})
    print(f.getName(), state.getPotentialEnergy())

# Minimization
#simulation.minimizeEnergy(maxIterations=10)

# WarmUp 
simulation.context.setVelocitiesToTemperature(300.0*u.kelvin)
#print('Warming up the system...')
#T = 5
#mdsteps = 60
#for i in range(60):
#  simulation.step(int(mdsteps/60) )
#  temperature = (T+(i*T))*u.kelvin 
#  integrator.setTemperature(temperature)

# Set velocities
#simulation.context.setVelocities(parm.velocities)

# The mdout file
rep = StateDataReporter("mdout", 5000,
    step=True,
    time=True,
    potentialEnergy=True,
    kineticEnergy=True,
    totalEnergy=True,
    temperature=True,
    volume=True,
    density=True)

simulation.reporters.append(rep)

# The mdinfo file
progress_rep = StateDataReporter('mdinfo', 5000, step=True, speed=True, progress=True, totalSteps=1000000, remainingTime=True)
simulation.reporters.append(progress_rep)

# The trajectory
simulation.reporters.append(DCDReporter("Zn_openmm-mace.dcd", 250))

# The checkpoint for restart
restrt_reporter = CheckpointReporter("md_md.chk", 5000)
simulation.reporters.append(restrt_reporter)

# Run 2 ns (time step = 2 fs)
simulation.step(1000000)

final_state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
with open('final_state.xml', 'w') as f:
    f.write(XmlSerializer.serialize(final_state))

quit()



