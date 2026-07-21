#!/usr/bin/env python
from __future__ import division, print_function
import os, sys
from openmm.app import *
from openmm import *
from openmmml import MLPotential
import openmm.unit as u
import numpy as np


##### Simulation system ##### 
# Create system in openmm  
prmtop_file = f'*.prmtop'
crd_file = f'*.inpcrd' 
prmtop = AmberPrmtopFile(prmtop_file)
inpcrd = AmberInpcrdFile(crd_file)

mm_system = prmtop.createSystem(nonbondedMethod=PME,
                nonbondedCutoff=10.0*u.angstrom,
                constraints=None, rigidWater=True,
                removeCMMotion=True)

# define the ML atoms (metal site atom numbers using 0 indexing)  
ml_atoms = [2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2115, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2181, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2467, 4709, 4710, 4711, 4712, 4713, 4714, 4715] 

# load the model and create mixed system  
model_file = '*.model'
potential = MLPotential('mace', modelPath=model_file)
system = potential.createMixedSystem(prmtop.topology, mm_system, ml_atoms)


##### Set link atoms as virtual sites #####
# Add link atoms at the ML/MM boundary (between CB and CA atoms of metal site) 
link_atoms = [2116, 2182, 2468]  # AtomID in pdb file 
cb_atoms = [2103, 2169, 2457]
ca_atoms = [2101, 2167, 2455]
link_atoms = [i-1 for i in link_atoms]  # Following treatment is because OpenMM starts from 0 index
cb_atoms = [i-1 for i in cb_atoms]
ca_atoms = [i-1 for i in ca_atoms]

for i in link_atoms:
    system.setParticleMass(i, 0.0)

w_ca = 1.09 / 1.526  # Ratio between H-C and C-C bond lengths
w_cb = 1.0 - w_ca

for i in range(len(link_atoms)):
    link_atom_idx = link_atoms[i]
    cb_atom_idx = cb_atoms[i]
    ca_atom_idx = ca_atoms[i]

    vs = TwoParticleAverageSite(cb_atom_idx, ca_atom_idx, w_cb, w_ca)
    system.setVirtualSite(link_atom_idx, vs)

# Check about the mass of the link atoms
for i in range(len(link_atoms)):
    link_atom_idx = link_atoms[i]
    mass = system.getParticleMass(link_atom_idx)
    if mass > 0.0*u.dalton:
        print(f"Warning: Link Atom mass is {mass}, fixing it to 0!")
        system.setParticleMass(link_atom_idx, 0.0)


##### Add constraints to the ML water molecules ##### 
d_OH = 0.09572  # TIP3P geometry
d_HH = 0.15136 

ml_water_molecules = [
    [4711, 4712, 4713],  # Atom IDs in PDB file of first water bound to Zn2+ 
    [4714, 4715, 4716]   
]
ml_water_molecules = [[x - 1 for x in sublist] for sublist in ml_water_molecules]   # Following treatment is because OpenMM starts from 0 index

added_constraints = 0
for water in ml_water_molecules:
    O_idx, H1_idx, H2_idx = water

    system.addConstraint(O_idx, H1_idx, d_OH)
    system.addConstraint(O_idx, H2_idx, d_OH)
    system.addConstraint(H1_idx, H2_idx, d_HH)
    added_constraints += 3

print(f'Added {added_constraints} rigid constraints to ML water molecules.')

# Apply constraints to all explicit bonds within the ML region using equilibrium distances from the MM force field (HarmonicBondForce). This prevents double-counting between the ML potential and MM bonded terms and avoids integration instability for stiff intra-ML bonds.
# Locate the HarmonicBondForce in the *original* MM system
ml_atoms_0idx = set(ml_atoms)
harm_bond_force = None
for force in mm_system.getForces():
    if isinstance(force, HarmonicBondForce):
        harm_bond_force = force
        break
if harm_bond_force is None:
    raise RuntimeError("HarmonicBondForce not found in mm_system – cannot set ML bond constraints.")

# Build a set of already-constrained pairs so we don't add duplicates
existing_constraint_pairs = set()
for ci in range(system.getNumConstraints()):
    p1, p2, _ = system.getConstraintParameters(ci)
    existing_constraint_pairs.add(tuple(sorted((p1, p2))))

ml_bond_constraints_added = 0
for bi in range(harm_bond_force.getNumBonds()):
    p1, p2, r0, k = harm_bond_force.getBondParameters(bi)
    if p1 in ml_atoms_0idx and p2 in ml_atoms_0idx:  # Only constrain bonds where BOTH atoms are inside the ML region
        pair = tuple(sorted((p1, p2)))
        if pair not in existing_constraint_pairs:
            r0_nm = r0.value_in_unit(u.nanometer)
            system.addConstraint(p1, p2, r0_nm)
            existing_constraint_pairs.add(pair)
            ml_bond_constraints_added += 1

print(f'Added {ml_bond_constraints_added} bond constraints for explicit bonds in the ML region using force-field equilibrium distances.')

num_constraints = system.getNumConstraints()
pairs = set()
for i in range(num_constraints):
    p1, p2, dist = system.getConstraintParameters(i)
    pair = tuple(sorted((p1, p2)))
    if pair in pairs:
        print(f"Duplicate constraint found between atoms {p1} and {p2}")
    pairs.add(pair)


##### Set different force groups #####
for i, f in enumerate(system.getForces()):
    print(f.getForceGroup())
    f.setForceGroup(i)


##### Set up integrator and simulation #####
integrator = LangevinIntegrator(300.0*u.kelvin, 2.0/u.picosecond,
                    0.002*u.picosecond)
integrator.setConstraintTolerance(1e-05)
platform = Platform.getPlatformByName("CUDA")
properties = {'Precision': 'mixed'}
simulation = Simulation(prmtop.topology, system, integrator, platform, properties)

# Set the positions and box vectors
simulation.context.setPositions(inpcrd.positions)
simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
simulation.context.computeVirtualSites()

# Print different force groups
for i, f in enumerate(system.getForces()):
    state = simulation.context.getState(getEnergy=True, groups={i})
    print(f.getName(), state.getPotentialEnergy())


##### Simulate #####
# Minimization
simulation.minimizeEnergy(maxIterations=1000)
print("Minimization of 1000 steps finished.")

# Warm up 
simulation.context.setVelocitiesToTemperature(300.0*u.kelvin)

# The mdout file
rep = StateDataReporter("mdout", 500,
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
progress_rep = StateDataReporter('mdinfo', 500, step=True, speed=True, progress=True, totalSteps=1000000, remainingTime=True)
simulation.reporters.append(progress_rep)

# The trajectory
simulation.reporters.append(DCDReporter("Zn_openmm-mace.dcd", 250))

# The checkpoint for restart
restrt_reporter = CheckpointReporter("md_md.chk", 500)
simulation.reporters.append(restrt_reporter)

# Run 2 ns (time step = 2 fs) 
simulation.step(1000000)

final_state = simulation.context.getState(getPositions=True,
     getVelocities=True, enforcePeriodicBox=True)
restrt_reporter.report(simulation, final_state)

# Print different force groups
for i, f in enumerate(system.getForces()):
    state = simulation.context.getState(getEnergy=True, groups={i})
    print(f.getName(), state.getPotentialEnergy())

quit()
