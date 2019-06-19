#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from qiskit import Aer

# lib from Qiskit Aqua
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import COBYLA

# lib from Qiskit Chemistry

from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock

# Define a molecule
driver = PySCFDriver(atom='C 0.86380 1.07246 1.16831; H 0.76957 0.07016 1.64057; H 1.93983 1.32622 1.04881; H 0.37285 1.83372 1.81325; H 0.37294 1.05973 0.17061', unit=UnitsType.ANGSTROM,
                          charge=0, spin=0, basis='sto-3g')

molecule = driver.run()

# Prepare qubit Hamiltonian

freeze_list = [0]
remove_list = [-3, -2]
map_type = 'bravyi_kitaev'

h1 = molecule.one_body_integrals
h2 = molecule.two_body_integrals
nuclear_repulsion_energy = molecule.nuclear_repulsion_energy

num_particles = molecule.num_alpha + molecule.num_beta
num_spin_orbitals = molecule.num_orbitals * 2
print("HF energy: {}".format(molecule.hf_energy-molecule.nuclear_repulsion_energy))
print("# of electrons: {}".format(num_particles))
print("# of spin orbitals: {}".format(num_spin_orbitals))

remove_list = [ x % molecule.num_orbitals for x in remove_list ]
freeze_list = [ x % molecule.num_orbitals for x in freeze_list ]

remove_list = [ x - len(freeze_list) for x in remove_list ]
remove_list += [ x + molecule.num_orbitals - len(freeze_list) for x in remove_list ]
freeze_list += [ x + molecule.num_orbitals for x in freeze_list ]

# Preparing the fermionic operator
energy_shift = 0.0
qubit_reduction = (map_type == 'parity')

ferOp = FermionicOperator(h1=h1, h2=h2)
if len(freeze_list) > 0:
    ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)

if len(remove_list) > 0:
    ferOp = ferOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)

qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)
qubitOp = qubitOp.two_qubit_reduced_operator(num_particles) if qubit_reduction else qubitOp
qubitOp.chop(10**-10)

print(qubitOp.print_operators())
print(qubitOp)

# using exact eigensolver to get the smallest eigenvalue

exact_eigensolver = ExactEigensolver(qubitOp, k=1)
ret = exact_eigensolver.run()
print("The computed energy is: {:.12f}".format(ret['eigvals'][0].real))
print("The total ground state energy is: {:.12f}".format(ret['eigvals'][0].real + energy_shift + nuclear_repulsion_energy))

from qiskit import IBMQ
#from qiskit.providers.ibmq import least_busy
IBMQ.load_accounts()
#large_enough_devices = IBMQ.backends(filters=lambda x: x.configuration().n_qubits > 4 and not x.configuration().simulator)
#backend = least_busy(large_enough_devices)

backend = Aer.get_backend('statevector_simulator')

# setup COBYLA optimizer
max_eval = 200
cobyla = COBYLA(maxiter=max_eval)

# setup HartreeFock state
HF_state = HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type, qubit_reduction)

# setup UCCSD variational form
var_form = UCCSD(qubitOp.num_qubits, depth=1, num_orbitals=num_spin_orbitals, num_particles=num_particles,active_occupied=[0], active_unoccupied=[0,1], initial_state=HF_state, qubit_mapping=map_type, two_qubit_reduction=qubit_reduction, num_time_slices=1)

# setup VQE
vqe = VQE(qubitOp, var_form, cobyla, 'matrix')
quantum_instance = QuantumInstance(backend=backend)

# Run algorithms and retrieve the results

results = vqe.run(quantum_instance)
print("The computed ground state energy is: {:.12f}".format(results['eigvals'][0]))
print("Final energy: {:.12f}".format(results['eigvals'][0] + energy_shift + nuclear_repulsion_energy))
print("Parameters: {}".format(results['opt_params']))
