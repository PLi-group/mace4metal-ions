#!/usr/bin/env python
"""
This script used the following conversion factors:
Hartree to eV: 27.2113825435
Bohr to A: 0.529177210903
Hartree/Bohr to eV/A: 51.42208619083232

Which have slightly difference from the most recent suggested values (version 2022) from:
    https://physics.nist.gov/cuu/Constants/index.html
Hartree to eV: 27.211386245981
Bohr to A: 0.529177210544
Hartree/Bohr to eV/A: 51.42206751119799  (based on the two numbers above)

But the difference is very small, with the ratio of the above
    value / below value as:
    Hartree to eV: 0.9999998639363329
    Bohr to A: 1.0000000006784115
    Hartree/Bohr to eV/A: 1.000000363261052

These differences will change the results in a very minimal manner,
and could be the reason that one gets slightly different
values from us and across different software.

For future calculations, one is suggested to use the most recent
suggested values from:
    https://physics.nist.gov/cuu/Constants/index.html
"""
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from tqdm import tqdm
from ase.io import write
import os

def read_orca_engrad(filename):
    """
    Parse an ORCA .engrad file.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Remove whitespace and clean data
    lines = [l.strip() for l in lines]
    
    # 1. Get number of atoms
    # The first block in ORCA engrad format is usually the number of atoms
    # Find the line below "# Number of atoms"
    try:
        idx_natoms = lines.index("# Number of atoms")
        n_atoms = int(lines[idx_natoms + 2])
    except ValueError:
        raise ValueError("Could not find '# Number of atoms' in the file")

    # 2. Get energy
    # Find "# The current total energy in Eh"
    try:
        idx_energy = lines.index("# The current total energy in Eh")
        energy_hartree = float(lines[idx_energy + 2])
        energy_ev = energy_hartree * 27.2113825435  # Convert Hartree to eV
    except ValueError:
        raise ValueError("Could not find energy information")

    # 3. Get gradient and convert to force
    # Force = - Gradient
    # Gradient unit is usually Hartree/Bohr
    try:
        idx_grad = lines.index("# The current gradient in Eh/bohr")
        # Gradient data starts from the following lines, total 3 * n_atoms lines
        grad_lines = lines[idx_grad + 2 : idx_grad + 2 + 3 * n_atoms]
        gradient_flat = np.array([float(x) for x in grad_lines])
        
        # Convert to Force (eV/Angstrom)
        # 1 Hartree/Bohr = 51.42208619083232 eV/Angstrom
        H_Bohr_to_eV_A = 51.42208619083232
        forces_flat = -gradient_flat * H_Bohr_to_eV_A
        forces = forces_flat.reshape(n_atoms, 3)
    except ValueError:
        raise ValueError("Could not find gradient information")

    # 4. Get coordinates and atomic numbers
    # Find "# The atomic numbers and current coordinates in Bohr"
    try:
        idx_coords = lines.index("# The atomic numbers and current coordinates in Bohr")
        coord_lines = lines[idx_coords + 2 : idx_coords + 2 + n_atoms]
        
        numbers = []
        positions_bohr = []
        
        for line in coord_lines:
            parts = line.split()
            numbers.append(int(parts[0]))
            positions_bohr.append([float(x) for x in parts[1:4]])
            
        # Convert Bohr to Angstrom (1 Bohr = 0.529177210903 Angstrom)
        Bohr_to_A = 0.529177210903
        positions = np.array(positions_bohr) * Bohr_to_A
        
    except ValueError:
        raise ValueError("Could not find coordinate information")

    # 5. Construct ASE Atoms object
    atoms = Atoms(numbers=numbers, positions=positions)
    
    # Attach energy and forces
    #calc = SinglePointCalculator(atoms, energy=energy_ev, forces=forces)
    #atoms.calc = calc
    
    # Also write to info/arrays for convenient saving
    atoms.info['energy'] = energy_ev
    atoms.arrays['forces'] = forces
    
    return atoms

# --- Configuration Area ---
# Path to the folder containing your .engrad files
input_dir = "Your input file folder"
# Final generated MACE training file
output_file = "Your output xyz file"

def main():
    # Get all .engrad files
    if not os.path.exists(input_dir):
        print(f"Error: Folder '{input_dir}' does not exist.")
        return

    files = [f for f in os.listdir(input_dir) if f.endswith('.engrad')]
    files.sort() # Sort to ensure consistent order
    
    if len(files) == 0:
        print("No .engrad files found, please check the path.")
        return

    print(f"Preparing to process {len(files)} files...")
    
    all_atoms = []
    success_count = 0
    fail_count = 0

    # Use tqdm to show progress bar
    for filename in tqdm(files, desc="Processing"):
        filepath = os.path.join(input_dir, filename)
        
        try:
            # 1. Call custom function to parse
            atoms = read_orca_engrad(filepath)
            
            # 2. Add extra metadata (optional)
            # E.g., record which file this structure came from for debugging
            atoms.info['config_name'] = filename
            
            # 3. Add to list
            all_atoms.append(atoms)
            success_count += 1
            
        except Exception as e:
            # If a file is corrupted, print error but do not stop program
            # print(f"\nSkipping file {filename}: {e}")
            fail_count += 1

    # --- Save Results ---
    if len(all_atoms) > 0:
        print(f"\nWriting to {output_file} ...")
        # format='extxyz' is required to save info and arrays
        write(output_file, all_atoms, format='extxyz')
        print("------------------------------------------------")
        print(f"Processing complete!")
        print(f"Success: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Output file: {output_file}")
        
        # Verify the first structure data
        print("\n[Verifying first structure data]:")
        print(f"Energy: {all_atoms[0].info['energy']:.4f} eV")
        print(f"Max Force: {np.max(np.abs(all_atoms[0].arrays['forces'])):.4f} eV/A")
    else:
        print("No valid data generated.")

if __name__ == "__main__":
    main()


