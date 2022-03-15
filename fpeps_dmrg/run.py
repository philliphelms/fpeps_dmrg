from quimb.tensor.fermion.block_interface import set_options
from pyblock3.algebra.fermion_symmetry import Z2, U1
from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D
from fpeps_dmrg.dmrg import *
from sys import argv

# System settings
Lx, Ly = int(argv[1]), int(argv[2])
Ne = int(argv[3])
D = int(argv[4])
u = float(argv[5])
peps = None
d = 4
chi = 2*D**2 #128
eig_step_size=1.
t = 1
random_initial_particle_dist = False
balance_bonds = False
equalize_norms = False

# Other settings
su_tau_dict = {0.5: 100,
               0.1: 200,
               0.05: 200,
               0.01: 400}
save_su_peps_loc=f'./saved_states/lowchi_Lx{Lx}_Ly{Ly}_Ne{Ne}_D{D}_t{t}_u{u}_chi{chi}_su'
save_peps_loc=f'./saved_states/lowchi_Lx{Lx}_Ly{Ly}_Ne{Ne}_D{D}_t{t}_u{u}_chi{chi}_dmrg'

# Set up som stuff
symmetry = 'u1'
symmetry_class = U1
set_options(symmetry=symmetry, use_cpp=True)

if RANK == 0:
    # Create a random PEPS
    if peps == None:
        peps = get_u1_init_fermionic_peps(Lx, Ly, Ne, D, d, symmetry_class, random_particle_dist=random_initial_particle_dist)
        su_ham = Hubbard2D(t, u, Lx, Ly)
        peps = run_simple_update(peps, su_ham, 
                                 D=D, chi=chi, 
                                 tau_dict=su_tau_dict,
                                 return_energy=False,
                                 energy_interval=100000)

        # Save SU PEPS for future use
        write_ftn_to_disc(peps, save_su_peps_loc, provided_filename=True)

    # Get the Sumops Hubbard Hamiltonian
    H = get_hubbard_op(Lx, Ly, t, u, 
                       symmetry=symmetry_class, 
                       flat=True,
                       max_distance=1.)

    # Run a DMRG style optimization
    E, peps = dmrg(H, peps, chi=chi, 
                   symmetry=symmetry_class, 
                   write_to_disc=True,
                   save_peps_loc=save_peps_loc,
                   eig_step_size=eig_step_size,
                   eig_maxiter=10,
                   eig_tol=1e-4,
                   eig_backend='lobpcg')

    # End Calculation
    for complete_rank in range(1, SIZE):
        COMM.send('finished', dest=complete_rank)
else:
    worker_execution()
