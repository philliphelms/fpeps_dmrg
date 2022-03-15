from quimb.tensor.fermion.fermion_core import FermionTensor, FermionTensorNetwork, FTNLinearOperator, _launch_fermion_expression
from quimb.tensor.fermion.block_interface import set_options
from quimb.tensor.tensor_core import tensor_contract
from quimb.tensor.fermion.fermion_2d import FPEPS, FermionTensorNetwork2D
from quimb.tensor.fermion.block_gen import rand_all_blocks as rand
from quimb.linalg.base_linalg import eig, eigh
import quimb.tensor as qtn
import numpy as np
import itertools
import shutil
import atexit
import pickle
import time
import uuid
import os
from pyblock3.algebra.fermion import eye
from pyblock3.algebra.fermion_symmetry import Z2, U1
from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D, SimpleUpdate, LocalHam2D
from pyblock3.algebra.fermion_ops import (ParticleNumber, H1, 
                                          ParticleNumberAlpha,
                                          ParticleNumberBeta,
                                          creation, annihilation, 
                                          onsite_U, bonded_vaccum)
np.set_printoptions(linewidth=1000, precision=5)

#####################################################################################
# MPI STUFF
#####################################################################################

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Make a temporary directory to write intermediate files to
RANK = MPI.COMM_WORLD.Get_rank()
if RANK == 0:
    TMPDIR = os.environ.get('TMPDIR','.')
    if TMPDIR[-1] == '/':
        TMPDIR = TMPDIR[:-1]
    RANDTMPDIR = TMPDIR + '/' + str(uuid.uuid4()) + '/'
    print(f'Saving temporary files in: {RANDTMPDIR}')
    os.mkdir(RANDTMPDIR)

def create_rand_tmpdir():
    # This funciton is poorly named... it just returns the already created
    # temporary directory for this calculation
    return RANDTMPDIR

def rand_fname():
    return str(uuid.uuid4())

def clear_tmpdir(tmpdir):
    try:
        shutil.rmtree(tmpdir)
    except OSError as e:
        pass

if RANK == 0:
    atexit.register(clear_tmpdir, RANDTMPDIR)

def parallelized_looped_function(func, iterate_over, args, kwargs):
    """
    When a function must be called many times for a set of parameters
    then this implements a parallelized loop controlled by the rank 0 process.
    
    Args:
        func: Function
            The function that will be called. 
        iterate_over: iterable
            The argument of the function that will be iterated over.
        args: list
            A list of arguments to be supplied to the function
        kwargs: dict
            A dictrionary of arguments to be supplied to the function
            at each call

    Returns:
        results: list
            This is a list of the results of each function call stored in 
            a list with the same ordering as was supplied in 'iterate_over'
    """
    # Figure out which items are done by which worker
    min_per_worker = len(iterate_over) // SIZE

    per_worker = [min_per_worker for _ in range(SIZE)]
    for i in range(len(iterate_over) - min_per_worker * SIZE):
        per_worker[SIZE-1-i] += 1

    randomly_permuted_tasks = np.random.permutation(len(iterate_over))
    worker_ranges = []
    for worker in range(SIZE):
        start = sum(per_worker[:worker])
        end = sum(per_worker[:worker+1])
        tasks = [randomly_permuted_tasks[ind] for ind in range(start, end)]
        worker_ranges.append(tasks)

    # Container for all the results
    worker_results = [None for _ in range(SIZE)]

    # Loop over all the processes (backwards so zero starts last
    for worker in reversed(range(SIZE)):

        # Collect all info needed for workers
        worker_iterate_over = [iterate_over[i] for i in worker_ranges[worker]]
        worker_info = [func, worker_iterate_over, args, kwargs]

        # Send to worker
        if worker != 0:
            COMM.send(worker_info, dest=worker)

        # Do task with this worker
        else:
            worker_results[0] = [None for _ in worker_ranges[worker]]
            for func_call in range(len(worker_iterate_over)):
                result = func(worker_iterate_over[func_call], 
                              *args, **kwargs)
                worker_results[0][func_call] = result

    # Collect all the results
    for worker in range(1, SIZE):
        worker_results[worker] = COMM.recv(source=worker)

    results = [None for _ in range(len(iterate_over))]
    for worker in range(SIZE):
        worker_ind = 0
        for i in worker_ranges[worker]:
            results[i] = worker_results[worker][worker_ind]
            worker_ind += 1

    # Return the results
    return results

def worker_execution():
    """
    All but the rank 0 process should initially be called
    with this function. It is an infinite loop that continuously 
    checks if an assignment has been given to this process. 
    Once an assignment is recieved, it is executed and sends
    the results back to the rank 0 process. 
    """
    # Create an infinite loop
    while True:

        # Loop to see if this process has a message
        # (helps keep processor usage low so other workers
        #  can use this process until it is needed)
        while not COMM.Iprobe(source=0):
            time.sleep(0.01)

        # Recieve the assignments from RANK 0
        assignment = COMM.recv()

        # End execution if received message 'finished'
        if assignment == 'finished': 
            break

        # Otherwise, call function
        function = assignment[0]
        iterate_over = assignment[1]
        args = assignment[2]
        kwargs = assignment[3]
        results = [None for _ in range(len(iterate_over))]
        for func_call in range(len(iterate_over)):
            results[func_call] = function(iterate_over[func_call], 
                                          *args, **kwargs)

        # Send the results back to the rank 0 process
        COMM.send(results, dest=0)

#####################################################################################
# READ/WRITE FTN FUNCS
#####################################################################################

def delete_ftn_from_disc(fname):
    """
    Simple wrapper that removes a file from disc. 
    Args:
        fname: str
            A string indicating the file to be removed
    """
    try:
        os.remove(fname)
    except:
        pass

def remove_env_from_disc(benv):
    """
    Simple wrapper that removes all files associated
    with a boundary environment from disc. 
    Args:
        benv: dict
            This is the dictionary holding the boundary environment
            tensor networks. Each entry in the dictionary should be a 
            dictionary. We check if the key 'tn' is in that dictionary 
            and if so, remove that file from disc.
    """
    for key in benv:
        if 'tn' in benv[key]:
            delete_ftn_from_disc(benv[key]['tn'])

def load_ftn_from_disc(fname, delete_file=False):
    """
    If a fermionic tensor network has been written to disc
    this function loads it back as a fermionic tensor network
    of the same class. if 'delete_file' is True, then the
    supplied file will also be removed from disc.
    """

    # Get the data
    if type(fname) != str:
        data = fname
    else:
        # Open up the file
        with open(fname, 'rb') as f:
            data = pickle.load(f)

    # Set up a dummy fermionic tensor network
    tn = FermionTensorNetwork([])

    # Put the tensors into the ftn
    tensors = [None,] * data['ntensors']
    for i in range(data['ntensors']):

        # Get the tensor
        ten_info = data['tensors'][i]
        ten = ten_info['tensor']
        ten = FermionTensor(ten.data, inds=ten.inds, tags=ten.tags)

        # Get/set tensor info
        tid, site = ten_info['fermion_info']
        ten.fermion_owner = None
        ten._avoid_phase = False

        # Add the required phase
        ten.phase = ten_info['phase']

        # Add to tensor list
        tensors[site] = (tid, ten)

    # Add tensors to the tn
    for (tid, ten) in tensors:
        tn.add_tensor(ten, tid=tid, virtual=True)

    # Get addition attributes needed
    tn_info = data['tn_info']

    # Set all attributes in the ftn
    extra_props = dict()
    for props in tn_info:
        extra_props[props[1:]] = tn_info[props]

    # Convert it to the correct type of fermionic tensor network
    tn = tn.view_as_(data['class'], **extra_props)

    # Remove file (if desired)
    if delete_file:
        delete_ftn_from_disc(fname)

    # Return resulting tn
    return tn

def write_ftn_to_disc(tn, tmpdir, provided_filename=False):
    """
    This function takes a fermionic tensor network that is supplied 'tn'
    and saves it as a random filename inside of the supplied directory
    'tmpdir'. If 'provided_filename' is True, then it will assume that 
    'tmpdir' includes a previously assigned filename and will overwrite
    that file
    """

    # Create a generic dictionary to hold all information
    data = dict()

    # Save which type of tn this is
    data['class'] = type(tn)

    # Add information relevant to the tensors
    data['tn_info'] = dict()
    for e in tn._EXTRA_PROPS:
        data['tn_info'][e] = getattr(tn, e)

    # Add the tensors themselves
    data['tensors'] = []
    ntensors = 0
    for ten in tn.tensors:
        ten_info = dict()
        ten_info['fermion_info'] = ten.get_fermion_info()
        ten_info['phase'] = ten.phase
        ten_info['tensor'] = ten
        data['tensors'].append(ten_info)
        ntensors += 1
    data['ntensors'] = ntensors

    # If tmpdir is None, then return the dictionary
    if tmpdir is None:
        return data

    # Write fermionic tensor network to disc
    else:
        # Create a temporary file
        if provided_filename:
            fname = tmpdir
            print('saving to ', fname)
        else:
            if tmpdir[-1] != '/': 
                tmpdir = tmpdir + '/'
            fname = tmpdir + rand_fname()

        # Write to a file
        with open(fname, 'wb') as f:
            pickle.dump(data, f)

        # Return the filename
        return fname

#####################################################################################
# SIMPLE UPDATE FUNCTIONS (FOR INITIAL GUESS)
#####################################################################################
def run_simple_update(peps, ham, D=None, chi=None, tau_dict={0.1: 100, 0.01: 100}, energy_interval=np.inf, return_energy=True):
    """
    This is a wrapper on quimb's simple update functionality

    Args:
        peps: FermionicPEPS object
            The initial guess to be supplied to the simple update algorithm
        ham: Fermionic LocalHam2D object
            The Hamiltonian to be used, example available at 
            quimb.tensor.fermion.fermion_2d_tebd.Hubbard2D

    kwargs:
        D: int
            The maximum bond dimension
        chi: int
            The maximum boundary bond dimension
        tau_dict: dictionary
            A dictionary with keys referring to the step size tau
            and entries being the number of simple update steps to be
            taken with that value of tau
        energy_interval: int
            How frequently the energy should be evaluated (default is 
            for the energy to never be evaluated)
        return_energy: bool
            Whether or not to return the final simple update energy

    Returns:
        state: FermionicPEPS object
            The resulting peps state
    """
    # Get a simple update object
    su = SimpleUpdate(peps, ham, D=D, chi=chi)

    # Run the simple update evolution
    taus = sorted(tau_dict.keys())[::-1]
    for tau in taus:
        niters = tau_dict[tau]
        for step in range(niters):
            # Print progress

            # Do the energy sweep
            su.sweep(tau)
            
            # Compute the energy
            if (step+1) % energy_interval == 0:
                Ei = su.get_state().compute_local_expectation(ham.terms,
                                                              max_bond=chi,
                                                              normalized=True)
                print(f'SU Step {step+1}/{niters} = {Ei}')
            else:
                print(f'SU step {step+1}/{niters}')
    
    # Compute final energy
    if return_energy:
        Ei = su.get_state().compute_local_expectation(ham.terms,
                                                      max_bond=chi,
                                                      normalized=True)
        return Ei, su.get_state()

    else:
        return su.get_state()

def get_u1_init_fermionic_peps(Nx, Ny, Ne, D, d, symmetry, random_particle_dist=False):
    """
    Generate an initial fermionic peps (product state) with a specified
    particle number (Ne)

    Args:
        Nx: int
            System size in x dimension
        Ny: int
            System size in y dimension
        Ne: int
            Number of electrons in the system
        D: int
            The maximum bond dimension (not actually used...)
        d: int
            The physical bond dimension (also doesnt seem to be used...)
        symmetry: U1
            U1 is the only currently supported argument

    Kwargs:
        random_particle_dist: bool
            If true, then instead of attempting to evenly distribute the 
            electrons in the initial product state, we randomly place them
            so some sites might have two electrons while others have zero

    Returns:
        peps: fermionicPEPS
            A fermionic peps object with a product state with the specified 
            particle number of electrons
            
    """
    if symmetry != U1:
        raise ValueError('Only works for U1 symmetry')

    # Generate a simple state as template
    peps = qtn.PEPS.rand(Nx, Ny, bond_dim=1, phys_dim=1)

    # Create a fermionic tensor network to store the fpeps
    fpeps = FermionTensorNetwork([])

    # Helper variables to create peps
    ind_to_pattern_map = dict()
    inv_pattern = {"+":"-", "-":"+"}

    # Needed operators
    cre = creation('sum') # a^{\dagger}_{alpha} + a^{\dagger}_{beta}
    cre_alpha = creation('a')
    cre_beta = creation('b')
    cre_double = np.tensordot(cre_alpha, cre_beta, axes=([-1,],[0,])) # a^{\dagger}_{alpha}a^{\dagger}_{beta}

    # Function to get pattern for tensor inds
    def get_pattern(inds):
        """
        make sure patterns match in input tensors, eg,

        --->A--->B--->
         i    j    k
        pattern for A_ij = +-
        pattern for B_jk = +-
        the pattern of j index must be reversed in two operands
        """
        pattern = ''
        for ix in inds[:-1]:
            if ix in ind_to_pattern_map:
                ipattern = inv_pattern[ind_to_pattern_map[ix]]
            else:
                nmin = pattern.count("-")
                ipattern = "-" if nmin*2<len(pattern) else "+"
                ind_to_pattern_map[ix] = ipattern
            pattern += ipattern
        pattern += "+" # assuming last index is the physical index
        return pattern

    # Assign number of particles for each site
    nelec = dict()
    sites = get_sites(Nx, Ny)
    if random_particle_dist:
        # Initiate all sites with zero electrons
        for site in sites:
            nelec[site] = 0
        # Randomly put in all the electrons
        for electron_i in range(Ne):
            filled = False
            while not filled:
                random_site_ind = np.random.randint(len(sites))
                site = sites[random_site_ind]
                if nelec[site] < 2:
                    nelec[site] += 1
                    filled = True
    else:
        sites = [sites[_] for _ in np.random.permutation(Nx*Ny)]
        if Ne == Nx*Ny:
            # Half filling
            for site in sites:
                nelec[site] = 1
        elif Ne < Nx*Ny:
            # Less than half filling
            for siteind in range(Ne):
                nelec[sites[siteind]] = 1
            for siteind in range(Ne, Nx*Ny):
                nelec[sites[siteind]] = 0
        elif Ne > Nx*Ny:
            # More than half filling
            for siteind in range(Nx*Ny):
                if siteind < Ne-Nx*Ny:
                    nelec[sites[siteind]] = 2
                else:
                    nelec[sites[siteind]] = 1

    # Create the product state
    for ix, iy in itertools.product(range(peps.Lx), range(peps.Ly)):

        # Get information on the tensor
        T = peps[ix, iy]
        pattern = get_pattern(T.inds)

        # Create a vacuum tensor at that site
        vac = bonded_vaccum((1,)*(T.ndim-1), pattern=pattern)
        trans_order = list(range(1,T.ndim))+[0]

        # Add particles as needed
        if nelec[ix, iy] == 1: 
            # Create one particle on this site: superposition |+> + |->
            data = np.tensordot(cre, vac, axes=((1,), (-1,))).transpose(trans_order)
        elif nelec[ix, iy] == 2:
            # Create two particles on this site
            data = np.tensordot(cre_double, vac, axes=((1,), (-1,))).transpose(trans_order)
        elif nelec[ix, iy] == 0:
            # Leave site without particles
            data = vac

        # Put results into a fermion tensor/tn
        new_T = FermionTensor(data, inds=T.inds, tags=T.tags)
        fpeps.add_tensor(new_T, virtual=False)

    # Return resulting fermion peps
    fpeps.view_as_(FPEPS, like=peps)

    # Check how many electrons have been included
    print(f'Number of electrons in fpeps: {np.sum([i.data.dq for i in fpeps.tensors])}')

    # Return resulting peps
    return fpeps

def get_simple_init_fermionic_peps(Lx, Ly, D, phys_dim, symmetry):
    """
    Generate an initial fermionic peps (product state) with a specified
    particle number (Ne)

    Args:
        Lx: int
            System size in x dimension
        Ly: int
            System size in y dimension
        D: int
            The maximum bond dimension (not actually used...)
        phys_dim: int
            The physical bond dimension (also doesnt seem to be used...)
        symmetry: Z2 or U1
            The type of symmetry to be used in generating the initial peps

    Returns:
        peps: fermionicPEPS
            A fermionic peps object with a simple product state
    """

    cre = creation('sum', symmetry=symmetry, flat=True)

    # Get a reference peps (to give bonds, etc)
    peps = qtn.PEPS.rand(Lx, Ly, 1, phys_dim=1)

    # Create a fermionic tensor network to store the fpeps
    ftn = FermionTensorNetwork([])

    # Dictionary of indices
    ind_map = dict()

    # A dict to flip plus to minus and reverse
    inv_pattern = {"+":"-", "-":"+"}

    # Function to give '+' or '-' for each index in a tensor
    def get_pattern(inds):
        pattern = ""

        # Loop over all provided indices
        for ix in inds[:-1]:

            # If ind is already recorded, give it opposite sign
            if ix in ind_map:
                ipattern = inv_pattern[ind_map[ix]]

            # Otherwise, give it a label based on what is already in the tensor
            else:
                lenth = len(pattern)
                nmin = pattern.count("-")
                ipattern = "-" if nmin*2<lenth else "+"
                ind_map[ix] = ipattern
            pattern += ipattern
        pattern += "+"
        return pattern

    # Loop over all tensors in the peps
    for ix, iy in itertools.product(range(Lx), range(Ly)):

        # Get the original peps tensor
        T = peps[ix, iy]

        # Get '+' or '-' for each index in the tensor
        pattern = get_pattern(T.inds)

        # Create the initial tensor
        vac = bonded_vaccum((1,)*(T.ndim-1), pattern=pattern, symmetry=symmetry, flat=True)
        trans_order = list(range(1,T.ndim))+[0]
        shape = (4,) * T.ndim
        data = np.tensordot(cre, vac, axes=((1,), (-1,))).transpose(trans_order)
        new_T = FermionTensor(data, inds=T.inds, tags=T.tags)

        # Add it to the fermionic tensor network
        ftn.add_tensor(new_T, virtual=False)

    # Return a fermionic peps object
    ftn.view_as_(FPEPS, like=peps)
    return ftn

#####################################################################################
# TENSOR NETWORK FUNCTIONS
#####################################################################################
def af_parity(ix, iy):
    # Set parity with all sites alternating between even and odd parity
    return (ix + iy) % 2

def odd_parity(ix, iy):
    # Convert from tensor network to peps
    # Set parity with all sites having odd parity
    return 1

def even_parity(ix, iy):
    # Set parity with all sites having even parity
    return 0

def get_random_z2_fermionic_peps(Lx, Ly, D, phys_dim=4, seed=0, parity_func=af_parity):
    """
    Creates a random fermionic peps with Z2 symmetry

    Args:
        Lx: int
            The number of sites in the x direction
        Ly: int
            The number of sites in the y direction
        D: int
            The maximum bond dimension

    Kwargs
        phys_dim: int
            The physical bond dimension
        seed: int
            A seed to use for generating random numbers
        parity_func: function
            A function to indicate whether each site is spin up or down, 
            examples include 'af_parity', 'odd_parity', and 'even_parity'

    Returns:
        peps: fermionicPEPS
            A generated random fermionic peps with z2 symmetry
    """
    # Get a reference peps (to give bonds, etc)
    peps = qtn.PEPS.rand(Lx, Ly, D, phys_dim=phys_dim)

    # Create a fermionic tensor network to store the fpeps
    ftn = FermionTensorNetwork([])

    # Dictionary of indices
    ind_map = dict()

    # A dict to flip plus to minus and reverse
    inv_pattern = {"+":"-", "-":"+"}

    # Function to give '+' or '-' for each index in a tensor
    def get_pattern(inds):
        pattern = ""

        # Loop over all provided indices
        for ix in inds[:-1]:

            # If ind is already recorded, give it opposite sign
            if ix in ind_map:
                ipattern = inv_pattern[ind_map[ix]]

            # Otherwise, give it a label based on what is already in the tensor
            else:
                lenth = len(pattern)
                nmin = pattern.count("-")
                ipattern = "-" if nmin*2<lenth else "+"
                ind_map[ix] = ipattern
            pattern += ipattern
        pattern += "+"
        return pattern

    # Loop over all tensors in the peps
    for ix, iy in itertools.product(range(Lx), range(Ly)):

        # Specify parity of sites
        dq = af_parity(ix, iy) # alternating parity distribution

        # Get the original peps tensor
        T = peps[ix, iy]

        # Get '+' or '-' for each index in the tensor
        pattern = get_pattern(T.inds)

        # Size of the blocks
        blk_shape = (int(D/2),) * (T.ndim-1) + (int(phys_dim/2),)

        # Create a random sparse tensor
        data = rand(blk_shape, [(0,1)]*T.ndim, pattern=pattern, dq=dq, seed=seed)
        seed += 1

        # Normalize that tensor
        data = data / data.norm() ** .5

        # Convert it into a FermionTensor
        new_T = FermionTensor(data, inds=T.inds, tags=T.tags)

        # Add it to the TN object
        ftn.add_tensor(new_T, virtual=True)

    # Convert from tensor network to peps
    ftn.view_as_(FPEPS, like=peps)
    return ftn

def peps_total_parity(tn, supplied_norm=True):
    """
    Get the total parity for a supplied tensor network
    (if norm == True, then the peps will be extracted from the norm first,
     so it is only the total parity of the peps)
    """
    if supplied_norm:
        Lx = tn.Lx
        Ly = tn.Ly
        tn = FermionTensorNetwork(tn['KET'])
        tn = tn.view_as(FPEPS,
                        site_tag_id='I{},{}',
                        row_tag_id='ROW{}',
                        col_tag_id='COL{}',
                        site_ind_id='k{},{}',
                        Lx=Lx, Ly=Ly)
    return sum([T.data.parity for T in tn.tensors])

def unpack_swap_phase(applied_order, correct_order):
    '''
    Very importantly, in each pair of gates, ham1[opind].keys() has an order.
    And that's the order for the gates to be applied. In the boundary contraction,
    they might be appended to the network in the wrong order. If so, a parity flep of -1
    needs to be accounted for.
    '''
    if not isinstance(applied_order, list):
        applied_order = list(applied_order)
    if not isinstance(correct_order, list):
        correct_order = list(correct_order)
    if applied_order == correct_order[::-1]:
        return 1 # odd parity
    else:
        return 0 # even parity

def total_parities_after(T):
    '''
    returns the sum of parities for tensors ordered after
    the input tensor, eg:
    A B C D E F orderred from left to right:
    _total_parities_after(C) = P_D + P_E + P_F
    '''
    order = T.get_fermion_info()[1]
    parity = 0
    for Tx, site in T.fermion_owner[0].tensor_order.values():
        if site > order:
            parity += Tx.data.parity
    return parity

#####################################################################################
# OPERATOR FUNCTIONS
#####################################################################################
class OPTERM:
    """
    Simple class that holds a single term in a operator.
    """
    def __init__(self, sites, ops, prefactor):
        """
        Creates an OPTERM object to hold a single operator

        Args:
            sites: list of tuples
                A list of sites that the operators are applied to, i.e.
                [(x1, y1), (x2, y2), ...]
            ops: a list
                The operators corresponding to each site in 'sites'
            prefactor: float
                A value to multiply against each operator. This is helpful
                for reusing intermediates because prefactors on hamiltonian
                terms do not need to be absorbed into the operators themselves.
        """
        self.sites = sites
        self.nops = len(self.sites)
        self.ops = dict(zip(sites, ops))
        self.prefactor = prefactor
        
        # Figure out the operator tags
        self.optags = dict()
        for site in self.sites:
            for tag in list(self.ops[site].tags):
                if tag[:7] == 'OPLABEL':
                    self.optags[site] = tag
                    break

        # Store all ops as dicts (can convert to and from FermionTensors)
        for site in self.ops:
            self.ops[site] = write_ftn_to_disc(FermionTensorNetwork([self.ops[site]]), None)

    def get_op(self, site):
        # Get the operator (loaded from disc if needed)
        return load_ftn_from_disc(self.ops[site]).tensors[0]

    def copy(self):
        # Return a new OPTERM object with copies of the operator
        return OPTERM(self.sites,
                      [self.get_op(site).copy() for site in self.sites],
                      self.prefactor)

def get_sites(Lx, Ly):
    # Get a 2D list with all of the sites in the lattice
    sites = []
    for x in range(Lx):
        for y in range(Ly):
            sites.append((x, y))
    return sites

def get_hubbard_op(Lx, Ly, t, U, symmetry=None, flat=None, max_distance=1., mu=None, v=1):
    """
    Get a hubbard operator for use with the dmrg style optimization

    Args:
        Lx: int
            System size in the x direction
        Ly: int
            System size in the y direction
        t: float
            Hubbard parameter t
        U: float
            Hubbard parameter U
    
    Kwargs:
        symmetry: 
            Which symmetry style to use for the operators, i.e. U1, Z2, etc.
        flat: bool
            How the operators should be stored
        max_distance: float
            operator pairs acting on sites with linear distance
            greater than max_distance will not be included.
        mu: float
            Chemical potential (not currently included)

    Returns:
        ops: list
            A list of OPTERM objects representing all of the operators
            in the Hubbard model Hamiltonian
    """
    # Raise error if chemical potential is used
    if mu is not None:
        raise NotImplementedError('Chemical potential not supported in Hubbard model currently')

    # Get all sites in the system
    sites = get_sites(Lx, Ly)

    # Get needed operators
    c_up = creation(spin='a', symmetry=symmetry, flat=flat)
    c_dn = creation(spin='b', symmetry=symmetry, flat=flat)
    cd_up = annihilation(spin='a', symmetry=symmetry, flat=flat)
    cd_dn = annihilation(spin='b', symmetry=symmetry, flat=flat)
    n = ParticleNumber(symmetry=symmetry, flat=flat)
    nund = onsite_U(u=1, symmetry=symmetry, flat=flat)

    # Create a list for Hamiltonian terms
    ham_terms = []
    for i in range(len(sites)):

        x1, y1 = sites[i]

        # Add single site U term
        opi1 = FermionTensor(nund, 
                             inds=['b{},{}'.format(x1, y1), 
                                   'k{},{}'.format(x1, y1)],
                             tags=['I{},{}'.format(x1, y1), 
                                   'COL{}'.format(y1), 
                                   'ROW{}'.format(x1), 
                                   'OPS',f'OPLABEL0,{x1},{y1}'])
        ham_terms.append(OPTERM([(x1, y1)], 
                                [opi1], 
                                U))
                
        # Add remaining two-site terms
        for j in range(i+1, len(sites)):

            # Figure out the sites of interest and interaction strength
            x2, y2 = sites[j]
            r = np.sqrt((x1-x2)**2. + (y1-y2)**2.)

            # Add all needed terms
            if r <= max_distance:

                # Add term: t*c+_1,up*c_2,up
                opi1 = FermionTensor(cd_up, 
                                     inds=['b{},{}'.format(x1, y1), 
                                           'k{},{}'.format(x1, y1)],
                                     tags=['I{},{}'.format(x1, y1), 
                                           'COL{}'.format(y1), 
                                           'ROW{}'.format(x1), 
                                           'OPS',f'OPLABEL1,{x1},{y1}'])
                opi2 = FermionTensor(c_up,
                                     inds=['b{},{}'.format(x2, y2), 
                                           'k{},{}'.format(x2, y2)],
                                     tags=['I{},{}'.format(x2, y2), 
                                           'COL{}'.format(y2), 
                                           'ROW{}'.format(x2), 
                                           'OPS',f'OPLABEL2,{x2},{y2}'])
                ham_terms.append(OPTERM([(x1, y1), (x2, y2)], 
                                        [opi1, opi2], 
                                        -t/(r**v)))

                # Add term: t*c+_2,up*c_1,up
                opi1 = FermionTensor(cd_up,
                                     inds=['b{},{}'.format(x2, y2), 
                                           'k{},{}'.format(x2, y2)],
                                     tags=['I{},{}'.format(x2, y2), 
                                           'COL{}'.format(y2), 
                                           'ROW{}'.format(x2), 
                                           'OPS',f'OPLABEL1,{x2},{y2}'])
                opi2 = FermionTensor(c_up, 
                                     inds=['b{},{}'.format(x1, y1), 
                                           'k{},{}'.format(x1, y1)],
                                     tags=['I{},{}'.format(x1, y1), 
                                           'COL{}'.format(y1), 
                                           'ROW{}'.format(x1), 
                                           'OPS',f'OPLABEL2,{x1},{y1}'])
                ham_terms.append(OPTERM([(x2, y2), (x1, y1)], 
                                        [opi1, opi2], 
                                        -t/(r**v)))

                # Add term: t*c+_1,dn*c_2,dn
                opi1 = FermionTensor(cd_dn, 
                                     inds=['b{},{}'.format(x1, y1), 
                                           'k{},{}'.format(x1, y1)],
                                     tags=['I{},{}'.format(x1, y1), 
                                           'COL{}'.format(y1), 
                                           'ROW{}'.format(x1), 
                                           'OPS',f'OPLABEL3,{x1},{y1}'])
                opi2 = FermionTensor(c_dn,
                                     inds=['b{},{}'.format(x2, y2), 
                                           'k{},{}'.format(x2, y2)],
                                     tags=['I{},{}'.format(x2, y2), 
                                           'COL{}'.format(y2), 
                                           'ROW{}'.format(x2), 
                                           'OPS',f'OPLABEL4,{x2},{y2}'])
                ham_terms.append(OPTERM([(x1, y1), (x2, y2)], 
                                        [opi1, opi2], 
                                        -t/(r**v)))

                # Add term: t*c+_2,up*c_1,up
                opi1 = FermionTensor(cd_dn,
                                     inds=['b{},{}'.format(x2, y2), 
                                           'k{},{}'.format(x2, y2)],
                                     tags=['I{},{}'.format(x2, y2), 
                                           'COL{}'.format(y2), 
                                           'ROW{}'.format(x2), 
                                           'OPS',f'OPLABEL3,{x2},{y2}'])
                opi2 = FermionTensor(c_dn, 
                                     inds=['b{},{}'.format(x1, y1), 
                                           'k{},{}'.format(x1, y1)],
                                     tags=['I{},{}'.format(x1, y1), 
                                           'COL{}'.format(y1), 
                                           'ROW{}'.format(x1), 
                                           'OPS',f'OPLABEL4,{x1},{y1}'])
                ham_terms.append(OPTERM([(x2, y2), (x1, y1)], 
                                        [opi1, opi2], 
                                        -t/(r**v)))
                                    
    # Return results
    return ham_terms

def get_operator_sites(tn):
    """
    Return a list of sites (each as a tuple) where operators
    have been applied into a tensor or tensor network
    """
    return [(int(tag.split(',')[1]),int(tag.split(',')[2])) for tag in list(tn.tags) if tag[:7] == 'OPLABEL']

def filter_ops_by_distance(H, max_distance=1.+1e-10):
    """
    Removes any hamiltonian terms that act on sites further
    than max_distance from each other
    """
    Hnew = []
    
    for term_ind in range(len(H)):

        # Find the maximum distance between sites
        dis = 0.
        for site1 in H[term_ind].sites:
            for site2 in H[term_ind].sites:
                x1, y1 = site1
                x2, y2 = site2
                r = np.sqrt((x1-x2)**2+(y1-y2)**2)
                dis = max(r, dis)

        # Only keep terms if distance is sufficiently small
        if dis <= max_distance:
            Hnew.append(H[term_ind].copy())

    # Return the shortened Hamiltonian
    return Hnew

#####################################################################################
# ENVIRONMENT CONTRACTION FUNCTIONS
#####################################################################################

def insert_op_into_tn(x, y, tn, opi, 
                      applied_op_order, 
                      expected_op_order, 
                      peps_parity,
                      contract=True):
    """
    Insert an op into the tensor network after the ket tensor on the site specified
    (All are done wth inplace=True)

    Args:
        x: int
            The x position of the site where the operator will be applied
        y: int
            The y position of the site where the operator will be applied
        tn: FermionicTN
            The 2D fermionic tensor network with bra and ket peps (and possibly 
            other operators) included
        opi: FermionicTensor
            The operator that will be added into the fermionic tensor network
        applied_op_order: list
            A list which indicates the order in which operators have been applied
            in the supplied tensor network
        expected_op_order: list
            The order in which the operators are expected to have been applied, 
            i.e. whether we applied a_i*a+_(i+1) or a+_(i+1)*a_i
        peps_parity: int
            The total parity of the supplied peps

    kwargs:
        contract: bool
            If contract==True, then the ket site and the operator will be contracted with one another 
            (this is helpful for boundary contractions because you don't have to contract the 
            operator layer separately, which can cause some errors)

    returns
        tn: FermionicTN
            The resutling tensor network with the operator added into it
        applied_order: list
            The order in which operators have been applied to the tensor network

    """
    # Get the tensor
    T = tn[f'I{x},{y}','KET']
    
    # Save the order of operator application
    applied_op_order = applied_op_order + [(x, y)]

    # Update the parity (if needed)
    if opi.parity != 0:
        parity = peps_parity
        parity -= unpack_swap_phase(applied_op_order, expected_op_order)
        parity -= total_parities_after(T)
        parity = parity % 2
        if parity * opi.parity == 1:
            opi.data._global_flip()

    # Put the operator into the tensor network
    tid, site = T.get_fermion_info()
    fermion_space = tn.fermion_space
    fermion_space.insert(site + 1, opi, virtual=True)
    tn |= opi

    # Change the bra index
    tn[f'I{x},{y}','BRA'].reindex_({f'k{x},{y}':f'b{x},{y}'})

    # Contract between the two tensors if needed:
    if contract:
        tn.contract_between([f'I{x},{y}','KET'],[f'I{x},{y}','OPS'])
        tn.reindex_({f'b{x},{y}':f'k{x},{y}'})

    # Return resulting tn
    return tn, applied_op_order

def update_norm_left_boundary_environments(y, norm, benvs, chi=-1, tmpdir=None):
    """
    Perform a left boundary contraction of the norm fermionic tensor network, 
    contracting column 'y' into the boundary mps.

    Args:
        y: int
            The column to be contracted into the boundary mps
        norm: FermionicTN
            The norm tensor network being contracted
        benvs: dict
            The boundary environments passed around as a dictionary.
            This dictionary contains all of the environment tensor networks
            and is the object that is being updated when boundary contractions
            are performed

    Kwargs:
        chi: int
            The boundary bond dimension to be used
        tmpdir: str
            The location of the temporary directory where temporary
            files should be stored. The tensor networks in the benvs 
            are written to this directory if tmpdir != None

    Returns:
        benvs: dict
            The boundary environment dictionary, now with a key added that is:
            ('left', y, 'norm') which contains the fermionic tensor network with 
            the column y contracted into the boundary environment
    """
    # Tag for boundary environment
    first_column = norm.col_tag(0)

    # Use empty tensor network for boundary of left-most column
    if y == 0:
        tn = FermionTensorNetwork([])
        benvs['left', y, 'norm'] = {'tn': write_ftn_to_disc(tn, tmpdir),
                                    'boundary_tag': None}

    # Don't contract the first column
    elif y == 1:
        tn = norm.copy()
        benvs['left', y, 'norm'] = {'tn': write_ftn_to_disc(tn, tmpdir),
                                    'boundary_tag': first_column}

    # Do boundary contractions for remainder
    else:
        tn = load_ftn_from_disc(benvs['left', y-1, 'norm']['tn']).copy()
        tn.contract_boundary_from_left_(xrange = (0, norm.Lx-1),
                                        yrange = (y-2, y-1),
                                        max_bond = chi)
        benvs['left', y, 'norm'] = {'tn': write_ftn_to_disc(tn, tmpdir),
                                    'boundary_tag': first_column}

    # Return result
    return benvs

def update_norm_right_boundary_environments(y, norm, benvs, chi=-1, tmpdir=None):
    """
    Perform a right boundary contraction of the norm fermionic tensor network, 
    contracting column 'y' into the boundary mps.

    Args:
        y: int
            The column to be contracted into the boundary mps
        norm: FermionicTN
            The norm tensor network being contracted
        benvs: dict
            The boundary environments passed around as a dictionary.
            This dictionary contains all of the environment tensor networks
            and is the object that is being updated when boundary contractions
            are performed

    Kwargs:
        chi: int
            The boundary bond dimension to be used
        tmpdir: str
            The location of the temporary directory where temporary
            files should be stored. The tensor networks in the benvs 
            are written to this directory if tmpdir != None

    Returns:
        benvs: dict
            The boundary environment dictionary, now with a key added that is:
            ('right', y, 'norm') which contains the fermionic tensor network with 
            the column y contracted into the boundary environment
    """
    # Tag for boundary environment
    last_column = norm.col_tag(norm.Ly-1)

    # Use empty tensor for boundary environment of right-most column
    if y == norm.Ly-1:
        tn = FermionTensorNetwork([])
        benvs['right', y, 'norm'] = {'tn': write_ftn_to_disc(tn, tmpdir),
                                     'boundary_tag': None}

    # Don't contract the first column
    elif y == norm.Ly-2:
        tn = norm.copy()
        benvs['right', y, 'norm'] = {'tn': write_ftn_to_disc(tn, tmpdir),
                                     'boundary_tag': last_column}

    # Do boundary contractions for remainder
    else:
        tn = load_ftn_from_disc(benvs['right', y+1, 'norm']['tn']).copy()
        tn.contract_boundary_from_right_(xrange = (0, norm.Lx-1),
                                         yrange = (y+1, y+2),
                                         max_bond = chi)
        benvs['right', y, 'norm'] = {'tn': write_ftn_to_disc(tn, tmpdir),
                                     'boundary_tag': last_column}

    # Return result
    return benvs

def update_ham_term_left_boundary_environments(Hi, y, norm, benvs, 
                                               chi=-1,
                                               tmpdir=None,
                                               peps_parity=None):
    """
    Perform a left boundary contraction of the norm fermionic tensor network, 
    contracting column 'y' into the boundary mps, inserting Hamiltonian terms
    into the boundary contraction as necessary. Note that we try to reuse intermediates
    here so that boundary environments are only contracted when necessary, i.e. 
    when the same operator has not already been contracted into a boundary 
    environment

    Args:
        Hi: Single OPTERM
            This is one of the Hamiltonian terms in the lattice, for which
            we will be contracting the boundary environment
        y: int
            The column to be contracted into the boundary mps
        norm: FermionicTN
            The norm tensor network being contracted
        benvs: dict
            The boundary environments passed around as a dictionary.
            This dictionary contains all of the environment tensor networks
            and is the object that is being updated when boundary contractions
            are performed

    Kwargs:
        chi: int
            The boundary bond dimension to be used
        tmpdir: str
            The location of the temporary directory where temporary
            files should be stored. The tensor networks in the benvs 
            are written to this directory if tmpdir != None

    Returns:
        benvs: dict
            The boundary environment dictionary, now with a key added that is:
            ('left', y, Hi.optags) which contains the fermionic tensor network with 
            the column y contracted into the boundary environment and operators
            applied to sites in column y. Note that only the 'optags' corresponding 
            to contracted operators in Hi are included. 
    """

    # Read norm from disc
    norm = load_ftn_from_disc(norm)

    # Tag for boundary environment
    first_column = norm.col_tag(0)

    # Check if the operator needs to be included
    if any(_y < y for _x, _y in Hi.sites):

        # Get the previous environment key
        # CASE 1: All terms are completed
        if all(_y < y-1 for _x, _y in Hi.sites):
            prev_key = [Hi.optags[site] for site in Hi.sites]
            prev_key.sort()
            prev_key = ('left', y-1, tuple(prev_key))

        # CASE 2: Term started, incomplete
        elif any(_y < y-1 for _x, _y in Hi.sites) and \
             any(_y >= y-1 for _x, _y in Hi.sites):
            prev_key = [Hi.optags[site] for site in Hi.sites if site[1] < y-1]
            prev_key.sort()
            prev_key = ('left', y-1, tuple(prev_key))

        # CASE 3: Unstarted term, started here
        elif all(_y >= y-1 for _x, _y in Hi.sites):
            prev_key = ('left', y-1, 'norm')
            
        # Get the key for resulting boundary env
        next_key = [Hi.optags[site] for site in Hi.sites if site[1] < y]
        next_key.sort()
        next_key = ('left', y, tuple(next_key))

        # Only do this contraction if we haven't done it before
        if not next_key in benvs:

            # Get the previous tn
            if y == 1:
                # Special case if no boundaries have been contracted yet
                prev_tn = norm.copy()
            else:
                prev_tn = load_ftn_from_disc(benvs[prev_key]['tn']).copy()

            # Get the order of operators that have been applied
            applied_order = benvs['op_order'][prev_key] if prev_key in benvs['op_order'] else []

            # Add in operators
            for xi in range(norm.Lx):
                yi = y-1
                if (xi, yi) in Hi.sites:
                    prev_tn, applied_order = insert_op_into_tn(xi, yi, prev_tn, 
                                                               Hi.get_op((xi, yi)).copy(),
                                                               applied_order,
                                                               Hi.sites,
                                                               peps_parity)

            # Do the boundary contraction (unless first column)
            if y > 1:
                prev_tn.contract_boundary_from_left_(xrange = (0, norm.Lx-1),
                                                     yrange = (y-2, y-1),
                                                     max_bond = chi,
                                                     layer_tags = ('BRA', 'KET'))
                                                     #layer_tags = ('BRA', 'OPS', 'KET'))

            # Save the result
            prev_tn = write_ftn_to_disc(prev_tn, tmpdir)

            # Return result
            return prev_tn, next_key, first_column, applied_order

def update_ham_term_right_boundary_environments(Hi, y, norm, benvs,
                                                chi=-1,
                                                tmpdir=None,
                                                peps_parity=None):
    """
    Perform a right boundary contraction of the norm fermionic tensor network, 
    contracting column 'y' into the boundary mps, inserting Hamiltonian terms
    into the boundary contraction as necessary. Note that we try to reuse intermediates
    here so that boundary environments are only contracted when necessary, i.e. 
    when the same operator has not already been contracted into a boundary 
    environment

    Args:
        Hi: Single OPTERM
            This is one of the Hamiltonian terms in the lattice, for which
            we will be contracting the boundary environment
        y: int
            The column to be contracted into the boundary mps
        norm: FermionicTN
            The norm tensor network being contracted
        benvs: dict
            The boundary environments passed around as a dictionary.
            This dictionary contains all of the environment tensor networks
            and is the object that is being updated when boundary contractions
            are performed

    Kwargs:
        chi: int
            The boundary bond dimension to be used
        tmpdir: str
            The location of the temporary directory where temporary
            files should be stored. The tensor networks in the benvs 
            are written to this directory if tmpdir != None

    Returns:
        benvs: dict
            The boundary environment dictionary, now with a key added that is:
            ('right', y, Hi.optags) which contains the fermionic tensor network with 
            the column y contracted into the boundary environment and operators
            applied to sites in column y. Note that only the 'optags' corresponding 
            to contracted operators in Hi are included. 
    """

    # Read norm from disc
    norm = load_ftn_from_disc(norm)

    # Tag for boundary environment
    last_column = norm.col_tag(norm.Ly-1)

    # Check if the operator needs to be included
    if any(_y > y  for _x, _y in Hi.sites):

        # Get the previous environment key
        # CASE 1: All terms are completed
        if all(_y > y+1 for _x, _y in Hi.sites):
            prev_key = [Hi.optags[site] for site in Hi.sites]
            prev_key.sort()
            prev_key = ('right', y+1, tuple(prev_key))

        # CASE 2: Term started, incomplete
        elif any(_y > y+1 for _x, _y in Hi.sites) and \
             any(_y <= y+1 for _x, _y in Hi.sites):
            prev_key = [Hi.optags[site] for site in Hi.sites if site[1] > y+1]
            prev_key.sort()
            prev_key = ('right', y+1, tuple(prev_key))

        # CASE 3: Unstarted term, started here
        elif all(_y <= y+1 for _x, _y in Hi.sites):
            prev_key = ('right', y+1, 'norm')

        # Get the key for the resulting boundary env
        next_key = [Hi.optags[site] for site in Hi.sites if site[1] > y]
        next_key.sort()
        next_key = ('right', y, tuple(next_key))

        # Only do this contraction if we haven't done it before
        if not next_key in benvs:

            # Get the previous tn
            if y == norm.Ly-2:
                # Special case where no bounds have been contracted yet
                prev_tn = norm.copy()
            else:
                prev_tn = load_ftn_from_disc(benvs[prev_key]['tn']).copy()

            # Get the order of operators that have been applied
            applied_order = benvs['op_order'][prev_key] if prev_key in benvs['op_order'] else []

            # Add in operators
            for xi in range(norm.Lx):
                yi = y+1
                if (xi, yi) in Hi.sites:
                    prev_tn, applied_order = insert_op_into_tn(xi, yi, prev_tn, 
                                                               Hi.get_op((xi, yi)).copy(),
                                                               applied_order,
                                                               Hi.sites,
                                                               peps_parity)

            # Do the boundary contraction
            if y < norm.Ly-2:
                prev_tn.contract_boundary_from_right_(xrange = (0, norm.Lx-1),
                                                      yrange = (y+1, y+2),
                                                      max_bond = chi,
                                                      layer_tags = ('BRA', 'OPS', 'KET'))

            # Save the result
            prev_tn = write_ftn_to_disc(prev_tn, tmpdir)

            # Return result
            return prev_tn, next_key, last_column, applied_order

def update_ham_left_boundary_environments(y, H, norm, benvs, **kwargs):
    """
    Perform a parallelized loop over all hamiltonian terms, doing a boundary contraction
    from the left for each term (only if necessary), and updating the benvs dict to have 
    boundary environments with column y contracted into the boundary mps

    Args:
        y: int
            The column to be contracted into the boundary mps
        H: List of OPTERMs
            This is a list of the Hamiltonian terms in the lattice, for which
            we will be contracting the boundary environment
        norm: FermionicTN
            The norm tensor network being contracted
        benvs: dict
            The boundary environments passed around as a dictionary.
            This dictionary contains all of the environment tensor networks
            and is the object that is being updated when boundary contractions
            are performed

    Kwargs: (passed directly to update_ham_term_left_boundary_environments)
        chi: int
            The boundary bond dimension to be used
        tmpdir: str
            The location of the temporary directory where temporary
            files should be stored. The tensor networks in the benvs 
            are written to this directory if tmpdir != None

    Returns:
        benvs: dict
            The boundary environment dictionary, now with new keys added that are:
            ('left', y, Hi.optags) which contain the fermionic tensor networks with 
            the column y contracted into the boundary environment and operators
            applied to sites in column y. Note that only the 'optags' corresponding 
            to contracted operators in Hi are included. 
    """
    # Do boundary update term-by-term (parallelized)
    function = update_ham_term_left_boundary_environments
    iterate_over = H
    args = [y,
            write_ftn_to_disc(norm, kwargs['tmpdir']),
            benvs]
    results = parallelized_looped_function(function, iterate_over,
                                           args, kwargs)

    # Put stuff back into the resulting benvs
    for i in range(len(results)):
        if results[i] is not None:
            prev_tn = results[i][0]
            next_key = results[i][1]
            boundary_tag = results[i][2]
            applied_order = results[i][3]
            benvs[next_key] = {'tn': prev_tn,
                               'boundary_tag': boundary_tag}
            benvs['op_order'][next_key] = applied_order

    # Return results
    return benvs

def update_ham_right_boundary_environments(y, H, norm, benvs, **kwargs):
    """
    Perform a parallelized loop over all hamiltonian terms, doing a boundary contraction
    from the right for each term (only if necessary), and updating the benvs dict to have 
    boundary environments with column y contracted into the boundary mps

    Args:
        y: int
            The column to be contracted into the boundary mps
        H: List of OPTERMs
            This is a list of the Hamiltonian terms in the lattice, for which
            we will be contracting the boundary environment
        norm: FermionicTN
            The norm tensor network being contracted
        benvs: dict
            The boundary environments passed around as a dictionary.
            This dictionary contains all of the environment tensor networks
            and is the object that is being updated when boundary contractions
            are performed

    Kwargs: (passed directly to update_ham_term_right_boundary_environments)
        chi: int
            The boundary bond dimension to be used
        tmpdir: str
            The location of the temporary directory where temporary
            files should be stored. The tensor networks in the benvs 
            are written to this directory if tmpdir != None

    Returns:
        benvs: dict
            The boundary environment dictionary, now with new keys added that are:
            ('right', y, Hi.optags) which contain the fermionic tensor networks with 
            the column y contracted into the boundary environment and operators
            applied to sites in column y. Note that only the 'optags' corresponding 
            to contracted operators in Hi are included. 
    """
    # Do boundary update term-by-term (parallelized)
    function = update_ham_term_right_boundary_environments
    iterate_over = H
    args = [y, 
            write_ftn_to_disc(norm, kwargs['tmpdir']),
            benvs]
    results = parallelized_looped_function(function, iterate_over,
                                           args, kwargs)

    # Put stuff back into the resulting benvs
    for i in range(len(results)):
        if results[i] is not None:
            prev_tn = results[i][0]
            next_key = results[i][1]
            boundary_tag = results[i][2]
            applied_order = results[i][3]
            benvs[next_key] = {'tn': prev_tn,
                               'boundary_tag': boundary_tag}
            benvs['op_order'][next_key] = applied_order

    # Return results
    return benvs

def update_left_boundary_environments(y, H, norm, benvs, 
                                      chi=-1,
                                      peps_parity=None,
                                      tmpdir=None):
    """
    Simple wrapper to update the left boundary environments for the 
    hamiltonian terms and the norm

    Args:
        y: int
            The column to be contracted into the boundary mps
        H: List of OPTERMs
            This is a list of the Hamiltonian terms in the lattice, for which
            we will be contracting the boundary environment
        norm: FermionicTN
            The norm tensor network being contracted
        benvs: dict
            The boundary environments passed around as a dictionary.
            This dictionary contains all of the environment tensor networks
            and is the object that is being updated when boundary contractions
            are performed

    Kwargs:
        chi: int
            The boundary bond dimension to be used
        tmpdir: str
            The location of the temporary directory where temporary
            files should be stored. The tensor networks in the benvs 
            are written to this directory if tmpdir != None
        peps_parity: int
            The total parity of the supplied peps

    Returns:
        benvs: dict
            The boundary environment dictionary, now with new keys added that are:
            ('left', y, Hi.optags) which contain the fermionic tensor networks with 
            the column y contracted into the boundary environment and operators
            applied to sites in column y. Note that only the 'optags' corresponding 
            to contracted operators in Hi are included. 
    """
    print(f'\t\tUpdating left boundary column {y}')
    benvs = update_ham_left_boundary_environments(y, H, norm, benvs, 
                                                  chi=chi,
                                                  tmpdir=tmpdir,
                                                  peps_parity=peps_parity)
    benvs = update_norm_left_boundary_environments(y, norm, benvs, chi=chi, tmpdir=tmpdir)
    return benvs

def update_right_boundary_environments(y, H, norm, benvs, 
                                       chi=-1, 
                                       peps_parity=None,
                                       tmpdir=None):
    """
    Simple wrapper to update the right boundary environments for the 
    hamiltonian terms and the norm

    Args:
        y: int
            The column to be contracted into the boundary mps
        H: List of OPTERMs
            This is a list of the Hamiltonian terms in the lattice, for which
            we will be contracting the boundary environment
        norm: FermionicTN
            The norm tensor network being contracted
        benvs: dict
            The boundary environments passed around as a dictionary.
            This dictionary contains all of the environment tensor networks
            and is the object that is being updated when boundary contractions
            are performed

    Kwargs:
        chi: int
            The boundary bond dimension to be used
        tmpdir: str
            The location of the temporary directory where temporary
            files should be stored. The tensor networks in the benvs 
            are written to this directory if tmpdir != None
        peps_parity: int
            The total parity of the supplied peps

    Returns:
        benvs: dict
            The boundary environment dictionary, now with new keys added that are:
            ('right', y, Hi.optags) which contain the fermionic tensor networks with 
            the column y contracted into the boundary environment and operators
            applied to sites in column y. Note that only the 'optags' corresponding 
            to contracted operators in Hi are included. 
    """
    print(f'\t\tUpdating right boundary column {y}')
    benvs = update_ham_right_boundary_environments(y, H, norm, benvs,
                                                   chi=chi,
                                                   tmpdir=tmpdir,
                                                   peps_parity=peps_parity)
    benvs = update_norm_right_boundary_environments(y, norm, benvs, chi=chi, tmpdir=tmpdir)
    return benvs

def move_benvs_right(y, H, norm, benvs, chi=-1, replace_curr_col=True, peps_parity=None, tmpdir=None):
    # First, replace updated sites in all boundary envs
    """
    This function moves the boundary environments one column to the right.
    This involves:
        1 - Replacing the column 'y' with any updated tensors in the 'norm' 
            object, if 'replace_curr_col == True'
        2 - Doing a boundary contraction for the Hamiltonian and norm from 
            the left side
        3 - Deleting any boundary environments that have been used

    Args:
        y: int
            The column to be contracted into the boundary mps
        H: List of OPTERMs
            This is a list of the Hamiltonian terms in the lattice, for which
            we will be contracting the boundary environment
        norm: FermionicTN
            The norm tensor network being contracted
        benvs: dict
            The boundary environments passed around as a dictionary.
            This dictionary contains all of the environment tensor networks
            and is the object that is being updated when boundary contractions
            are performed

    Kwargs:
        chi: int
            The boundary bond dimension to be used
        replace_curr_col: bool
            If True, then the tensors in column 'y' of the 'norm' supplied
            will be put into all of the 'benvs' environments 
        peps_parity: int
            The total parity of the supplied peps
        tmpdir: str
            The location of the temporary directory where temporary
            files should be stored. The tensor networks in the benvs 
            are written to this directory if tmpdir != None

    Returns:
        benvs: dict
            The boundary environment dictionary updated to include
            left norm and hamiltonian boundaries for column 'y+1'
    """
    if replace_curr_col:
        for key in benvs:
            if key[0] == 'left' and key[1] > 0:
                for x in range(norm.Lx):
                    new_tn = transfer_site_ten(x, y, norm,
                                               load_ftn_from_disc(benvs[key]['tn']))
                    benvs[key]['tn'] = write_ftn_to_disc(new_tn, tmpdir)

    # Now do the environment update
    benvs = update_left_boundary_environments(y+1, H, 
                                              norm, benvs, 
                                              chi=chi, 
                                              tmpdir=tmpdir,
                                              peps_parity=peps_parity)

    # Delete environments that have already been used
    rm_keys = [key for key in benvs if key[0] == 'right' and key[1] == y]
    for key in rm_keys:
        removed_env = benvs.pop(key)
        delete_ftn_from_disc(removed_env['tn'])

    # Return result
    return benvs

def move_benvs_left(y, H, norm, benvs, chi=-1, replace_curr_col=True, peps_parity=None, tmpdir=None):
    """
    This function moves the boundary environments one column to the left.
    This involves:
        1 - Replacing the column 'y' with any updated tensors in the 'norm' 
            object, if 'replace_curr_col == True'
        2 - Doing a boundary contraction for the Hamiltonian and norm from 
            the right side
        3 - Deleting any boundary environments that have been used

    Args:
        y: int
            The column to be contracted into the boundary mps
        H: List of OPTERMs
            This is a list of the Hamiltonian terms in the lattice, for which
            we will be contracting the boundary environment
        norm: FermionicTN
            The norm tensor network being contracted
        benvs: dict
            The boundary environments passed around as a dictionary.
            This dictionary contains all of the environment tensor networks
            and is the object that is being updated when boundary contractions
            are performed

    Kwargs:
        chi: int
            The boundary bond dimension to be used
        replace_curr_col: bool
            If True, then the tensors in column 'y' of the 'norm' supplied
            will be put into all of the 'benvs' environments 
        peps_parity: int
            The total parity of the supplied peps
        tmpdir: str
            The location of the temporary directory where temporary
            files should be stored. The tensor networks in the benvs 
            are written to this directory if tmpdir != None

    Returns:
        benvs: dict
            The boundary environment dictionary updated to include
            right norm and hamiltonian boundaries for column 'y+1'
    """
    # First, replace updated sites in all boundary envs
    if replace_curr_col:
        for key in benvs:
            if key[0] == 'right' and key[1] < norm.Ly-1:
                for x in range(norm.Lx):
                    new_tn = transfer_site_ten(x, y, norm,
                                               load_ftn_from_disc(benvs[key]['tn']))
                    benvs[key]['tn'] = write_ftn_to_disc(new_tn, tmpdir)

    # Now do the environment update
    benvs = update_right_boundary_environments(y-1, H, 
                                               norm, benvs,
                                               chi=chi,
                                               tmpdir=tmpdir,
                                               peps_parity=peps_parity)

    # Delete environments that have already been used
    rm_keys = [key for key in benvs if key[0] == 'left' and key[1] == y]
    for key in rm_keys:
        removed_env = benvs.pop(key)
        delete_ftn_from_disc(removed_env['tn'])

    # Return result
    return benvs

def initialize_boundary_environments(y, H, norm, chi=-1, peps_parity=None, tmpdir=None):
    """
    Initialize the left and right boundary environments around column 'y'

    Args:
        y: int
            The column around which the boundary environments are contracted
        H: List of OPTERMS
            The Hamiltonian
        norm: FermionicTN
            The bra and ket peps combined into a norm

    Kwargs:
        chi: int
            The boundary bond dimension used in boundary contractions
        peps_parity: int
            THe parity of the PEPS
        tmpdir: str
            The temporary directory where intermediate files are stored

    returns:
        norm: FermionTN
            The norm (now with sites 'reordered' for boundary contraction)
        benvs: dictionary
            A dictionary containing all of the boundary environments 
            contracted around this site
    """
    # Put the norm in column order for boundary contraction
    norm = norm.reorder("col", layer_tags=('KET', 'BRA'))

    # Initialize empty dictionary to hold environments
    benvs = dict()

    # Another dictionary tracks the order of gate application in each boundary
    benvs['op_order'] = dict()

    # Initialize the left boundaries
    for col in range(y+1):
        benvs = update_left_boundary_environments(col, H, norm, benvs, 
                                                  chi=chi,
                                                  peps_parity=peps_parity,
                                                  tmpdir=tmpdir)

    # Initialize the right boundaries
    for col in range(norm.Ly-1, y-1, -1):
        benvs = update_right_boundary_environments(col, H, norm, benvs, 
                                                   chi=chi,
                                                   peps_parity=peps_parity,
                                                   tmpdir=tmpdir)

    # Return the result
    return norm, benvs

def get_left_env(x, y, H, benvs):
    """
    Get the left environment for a single OPTERM

    Args:
        x: int
            x site
        y: int
            y site
        H: OPTERM
            A single term in the hamiltonian
        benvs: dict
            The dictionary of aboundary environments

    Returns:
        left_tn: FermionicTN
            the desired left boundary environment
        left_tag: str
            The tag indicating the left boundary 
        left_key: tuple
            The key corresponding to this boundary
            environment for reference in the benvs
            dictionary
    """
    # Get the left environment key
    # CASE 1: All terms are completed
    if all(_y < y for _x, _y in H.sites):
        left_key = [H.optags[site] for site in H.sites]
        left_key.sort()
        left_key = ('left', y, tuple(left_key))

    # CASE 2: Term started, incomplete
    elif any(_y < y for _x, _y in H.sites) and \
         any(_y >= y for _x, _y in H.sites):
        #left_key = [H.optags[site] for site in H.sites if site[1] < y-1]
        left_key = [H.optags[site] for site in H.sites if site[1] < y]
        left_key.sort()
        left_key = ('left', y, tuple(left_key))

    # CASE 3: Unstarted term, started here
    elif all(_y >= y for _x, _y in H.sites):
        left_key = ('left', y, 'norm')

    # Get the left env tn
    left_tn = load_ftn_from_disc(benvs[left_key]['tn'])
    left_tag = benvs[left_key]['boundary_tag']

    return left_tn, left_tag, left_key

def get_right_env(x, y, H, benvs):
    """
    Get the right environment for a single OPTERM

    Args:
        x: int
            x site
        y: int
            y site
        H: OPTERM
            A single term in the hamiltonian
        benvs: dict
            The dictionary of aboundary environments

    Returns:
        right_tn: FermionicTN
            the desired right boundary environment
        right_tag: str
            The tag indicating the right boundary 
        right_key: tuple
            The key corresponding to this boundary
            environment for reference in the benvs
            dictionary
    """
    # Get the right environment key
    # CASE 1: All terms are completed
    if all(_y > y for _x, _y in H.sites):
        right_key = [H.optags[site] for site in H.sites]
        right_key.sort()
        right_key = ('right', y, tuple(right_key))

    # CASE 2: Term started, incomplete
    elif any(_y > y for _x, _y in H.sites) and \
         any(_y <= y for _x, _y in H.sites):
        right_key = [H.optags[site] for site in H.sites if site[1] > y]
        right_key.sort()
        right_key = ('right', y, tuple(right_key))

    # CASE 3: Unstarted term, started here
    elif all(_y <= y for _x, _y in H.sites):
        right_key = ('right', y, 'norm')

    # Get the right env tn
    right_tn = load_ftn_from_disc(benvs[right_key]['tn'])
    right_tag = benvs[right_key]['boundary_tag']

    return right_tn, right_tag, right_key

def get_column_and_env_tn(x, y, norm, H, benvs):
    """
    Returns a tensor network with the left and right environments
    around column 'y' and the tensors for column y from 'norm' and
    any operator terms in H

    Args:
        x: int
            the x position in the lattice
        y: int
            the y position in the lattice
        norm: FermionicTN
            The bra and ket peps combined into a TN
        H: OPTERM
            A single OPTERM
        benvs: dict
            The boundary environments

    Returns:
        tn: FermionicTN
            A TN with the left and right environments combined with 
            the current column and all operators in H
        applied_order: list of tuples
            A list of the order in which operators have been applied
    """

    # Get the boundary tns
    left_tn, left_tag, left_key = get_left_env(x, y, H, benvs)
    right_tn, right_tag, right_key = get_right_env(x, y, H, benvs)

    # Create the tn
    left_env = left_tn.select(left_tag).copy()
    right_env = right_tn.select(right_tag).copy()
    central_col = norm.select(norm.col_tag(y)).copy()
    tn = FermionTensorNetwork((left_env,
                               central_col,
                               right_env),
                              check_collisions=False).view_as_(FermionTensorNetwork2D, like=norm)

    # Get the order of operators that have been applied
    left_env_order = benvs['op_order'][left_key] if left_key in benvs['op_order'] else []
    right_env_order = benvs['op_order'][right_key] if right_key in benvs['op_order'] else []
    applied_order = left_env_order + right_env_order

    # Check if we need a flip
    if (len(left_env_order)  < len(H.sites) and 
        len(right_env_order) < len(H.sites) and 
        len(right_env_order)+len(left_env_order) == len(H.sites) and 
        unpack_swap_phase(applied_order, H.sites) == 1):
        tn['I0,0','KET'].data._global_flip()

    # Do a reorder
    tn = tn.reorder("row", layer_tags=('KET', 'BRA'))

    # Return resulting tn
    return tn, applied_order

def update_norm_top_environments(x, y, norm, benvs, top_envs, tmpdir=None):
    """
    Update all the top norm environment in a column

    Args:
        x: int
            The x position in the lattice
        y int
            The y position in the lattice
        norm: FermionicTN
            The norm containing the bra nad ket peps
        benvs: dict
            The boundary environments
        bottom_envs: dict
            The current set of abottom environments

    Kwargs:
        tmpdir: str
            A temporary directory where intermediate files are stored

    Returns:
        top_envs: dict
            The updated set of top environments
    """
    # Tag for boundary environment
    last_row = norm.row_tag(norm.Lx-1)

    # Use empty tensor network for boundary of top-most row
    if x == norm.Lx-1:
        tn = FermionTensorNetwork([])
        top_envs[x, 'norm'] = {'tn': write_ftn_to_disc(tn, tmpdir),
                               'boundary_tag': None}

    # First row - create a tn of the column and surrounding boundaries
    elif x == norm.Lx-2:
        # Construct the column tn
        left_tn = load_ftn_from_disc(benvs['left', y, 'norm']['tn'])
        right_tn = load_ftn_from_disc(benvs['right', y, 'norm']['tn'])
        left_tag = benvs['left', y, 'norm']['boundary_tag']
        right_tag = benvs['right', y, 'norm']['boundary_tag']
        left_env = left_tn.select(left_tag).copy()
        right_env = right_tn.select(right_tag).copy()
        central_col = norm.select(norm.col_tag(y)).copy()
        tn = FermionTensorNetwork((left_env, 
                                   central_col,
                                   right_env),
                                  check_collisions=False).view_as_(FermionTensorNetwork2D, like=norm)

        # Do a reorder
        tn = tn.reorder("row", layer_tags=('KET', 'BRA'))

        # Save the result
        top_envs[x, 'norm'] = {'tn': write_ftn_to_disc(tn, tmpdir),
                               'boundary_tag': last_row}

    # Remainder of rows via boundary contraction
    else:
        tn = load_ftn_from_disc(top_envs[x+1, 'norm']['tn']).copy()
        tn.contract_boundary_from_top_(xrange = (x+1, x+2),
                                          yrange = (max(y-1, 0), min(y+1, norm.Ly-1)),
                                          max_bond = -1)
        top_envs[x, 'norm'] = {'tn': write_ftn_to_disc(tn, tmpdir),
                               'boundary_tag': last_row}

    # return result
    return top_envs

def update_norm_bottom_environments(x, y, norm, benvs, bottom_envs, tmpdir=None):
    """
    Update all the bottom norm environment in a column

    Args:
        x: int
            The x position in the lattice
        y int
            The y position in the lattice
        norm: FermionicTN
            The norm containing the bra nad ket peps
        benvs: dict
            The boundary environments
        bottom_envs: dict
            The current set of abottom environments

    Kwargs:
        tmpdir: str
            A temporary directory where intermediate files are stored

    Returns:
        bottom_envs: dict
            The updated set of bottom environments
    """
    # Tag for boundary environment
    first_row = norm.row_tag(0)

    # Use empty tensor network for boundary of bottom-most row
    if x == 0:
        tn = FermionTensorNetwork([])
        bottom_envs[x, 'norm'] = {'tn': write_ftn_to_disc(tn, tmpdir),
                                  'boundary_tag': None}

    # First row is unique
    elif x == 1:
        # Construct the column tn
        left_tn = load_ftn_from_disc(benvs['left', y, 'norm']['tn'])
        right_tn = load_ftn_from_disc(benvs['right', y, 'norm']['tn'])
        left_tag = benvs['left', y, 'norm']['boundary_tag']
        right_tag = benvs['right', y, 'norm']['boundary_tag']
        left_env = left_tn.select(left_tag).copy()
        right_env = right_tn.select(right_tag).copy()
        central_col = norm.select(norm.col_tag(y)).copy()
        tn = FermionTensorNetwork((left_env, 
                                   central_col,
                                   right_env),
                                  check_collisions=False).view_as_(FermionTensorNetwork2D, like=norm)

        # Do a reorder
        tn = tn.reorder("row", layer_tags=('KET', 'BRA'))

        # Save the result
        bottom_envs[x, 'norm'] = {'tn': write_ftn_to_disc(tn, tmpdir),
                                  'boundary_tag': first_row}

    # Remainder of rows via boundary contraction
    else:
        tn = load_ftn_from_disc(bottom_envs[x-1, 'norm']['tn']).copy()
        tn.contract_boundary_from_bottom_(xrange = (x-2, x-1),
                                          yrange = (max(y-1, 0), min(y+1, norm.Ly-1)),
                                          max_bond = -1)
        bottom_envs[x, 'norm'] = {'tn': write_ftn_to_disc(tn, tmpdir),
                                  'boundary_tag': first_row}

    # Return result
    return bottom_envs

def update_ham_term_top_environments(Hi, x, y, norm, benvs, top_envs,
                                     peps_parity=None,
                                     tmpdir=None):
    """
    Update the top hamiltonian environment for a single term in a column

    Args:
        H: OPTERM
            The Hamiltonian term
        x: int
            The x position in the lattice
        y int
            The y position in the lattice
        norm: FermionicTN
            The norm containing the bra nad ket peps
        benvs: dict
            The boundary environments
        top_envs: dict
            The current set of top environments

    Kwargs:
        peps_parity: int
            The parity of the supplied peps
        tmpdir: str
            The location to a temporary directory where intermediates can be stored

    Returns:
        top_envs: dict
            The updated set of top environments
    """
    # Get the norm tn
    norm = load_ftn_from_disc(norm)

    # Tag for boundary environment
    last_row = norm.row_tag(norm.Lx-1)

    # Get the key for the top env
    next_key = [Hi.optags[site] for site in Hi.sites]
    next_key.sort()
    prev_key = (x+1, tuple(next_key))
    next_key = (x, tuple(next_key))

    # Use empty tensor network for boundary of top-most row
    if x == norm.Lx-1:
        tn = FermionTensorNetwork([])
        tn = write_ftn_to_disc(tn, tmpdir)
        return tn, next_key, None, None

    # First row - create a tn of the column and surrounding boundaries
    elif x == norm.Lx-2:

        # Get the tn with peps column and left and right environments
        tn, applied_order = get_column_and_env_tn(x, y, norm, Hi, benvs)

    # Remainder of rows, just use previous environments
    else:
        # Get the previous environment
        tn = load_ftn_from_disc(top_envs[prev_key]['tn']).copy()
        applied_order = top_envs['op_order'][prev_key]

    # Add operators
    if x < norm.Lx-1:

        xi, yi = x+1, y
        if (xi, yi) in Hi.sites:
            tn, applied_order = insert_op_into_tn(xi, yi, tn, 
                                                  Hi.get_op((xi, yi)).copy(),
                                                  applied_order,
                                                  Hi.sites,
                                                  peps_parity)

        # Now do boundary contraction (if needed)
        if x < norm.Lx-2: 

            # Do the boundary contraction
            tn.contract_boundary_from_top_(xrange = (x+1, x+2),
                                           yrange = (max(y-1, 0), min(y+1, norm.Ly-1)),
                                           max_bond=-1)

        # Save the resulting tn
        tn = write_ftn_to_disc(tn, tmpdir)

        # Return the results
        return tn, next_key, last_row, applied_order

def update_ham_term_bottom_environments(Hi, x, y, norm, benvs, bottom_envs,
                                        peps_parity=None,
                                        tmpdir=None):
    """
    Update the bottom hamiltonian environment for a single term in a column

    Args:
        H: OPTERM
            The Hamiltonian term
        x: int
            The x position in the lattice
        y int
            The y position in the lattice
        norm: FermionicTN
            The norm containing the bra nad ket peps
        benvs: dict
            The boundary environments
        bottom_envs: dict
            The current set of abottom environments

    Kwargs:
        peps_parity: int
            The parity of the supplied peps
        tmpdir: str
            The location to a temporary directory where intermediates can be stored

    Returns:
        bottom_envs: dict
            The updated set of bottom environments
    """
    # Get the norm tn
    norm = load_ftn_from_disc(norm)

    # Tag for boundary environment
    first_row = norm.row_tag(0)

    # Get the key for the bottom env
    next_key = [Hi.optags[site] for site in Hi.sites]
    next_key.sort()
    prev_key = (x-1, tuple(next_key))
    next_key = (x, tuple(next_key))

    # Use empty tensor network for boundary of bottom-most row
    if x == 0:
        tn = FermionTensorNetwork([])
        tn = write_ftn_to_disc(tn, tmpdir)
        return tn, next_key, None, None

    # First row - create a tn of the column and surrounding boundaries
    elif x == 1:

        # Get the tn with peps column and left and right boundaries
        tn, applied_order = get_column_and_env_tn(x, y, norm, Hi, benvs)

    # Remainder of rows, just use previous environments
    else:
        # Get the previous environment
        tn = load_ftn_from_disc(bottom_envs[prev_key]['tn']).copy()
        applied_order = bottom_envs['op_order'][prev_key]

    # Add operators
    if x > 0:

        xi, yi = x-1, y
        if (xi, yi) in Hi.sites:
            tn, applied_order = insert_op_into_tn(xi, yi, tn, 
                                                  Hi.get_op((xi, yi)).copy(),
                                                  applied_order,
                                                  Hi.sites,
                                                  peps_parity)

        # Now do boundary contraction (if needed)
        if x > 1:
            tn.contract_boundary_from_bottom_(xrange = (x-2, x-1),
                                              yrange = (max(y-1, 0), min(y+1, norm.Ly-1)),
                                              max_bond = -1)

        # Save the resulting tn
        tn = write_ftn_to_disc(tn, tmpdir)

        # Return the result
        return tn, next_key, first_row, applied_order

def update_ham_top_environments(x, y, H, norm, benvs, top_envs, **kwargs):
    """
    Update all the top hamiltonian environments in a column

    Args:
        x: int
            The x position in the lattice
        y int
            The y position in the lattice
        H: List of OPTERMS
            The Hamiltonian
        norm: FermionicTN
            The norm containing the bra nad ket peps
        benvs: dict
            The boundary environments
        top_envs: dict
            The current set of top environments

    Returns:
        top_envs: dict
            The updated set of top environments
    """
    # Do the boundary update term-by-term (parallelized)
    function = update_ham_term_top_environments
    iterate_over = H
    args = [x, y, 
            write_ftn_to_disc(norm, kwargs['tmpdir']),
            benvs,
            top_envs]
    results = parallelized_looped_function(function, iterate_over,
                                           args, kwargs)

    # Put stuff back into the top_envs object
    for i in range(len(results)):
        if results[i] is not None:
            tn = results[i][0]
            next_key = results[i][1]
            boundary_tag = results[i][2]
            applied_order = results[i][3]
            top_envs[next_key] = {'tn': tn,
                                  'boundary_tag': boundary_tag}
            if applied_order is not None:
                top_envs['op_order'][next_key] = applied_order

    return top_envs

def update_ham_bottom_environments(x, y, H, norm, benvs, bottom_envs, **kwargs):
    """
    Update all the bottom hamiltonian environments in a column

    Args:
        x: int
            The x position in the lattice
        y int
            The y position in the lattice
        H: List of OPTERMS
            The Hamiltonian
        norm: FermionicTN
            The norm containing the bra nad ket peps
        benvs: dict
            The boundary environments
        bottom_envs: dict
            The current set of abottom environments

    Returns:
        bottom_envs: dict
            The updated set of bottom environments
    """
    # Do the boundary update term-by-term (parallelized)
    function = update_ham_term_bottom_environments
    iterate_over = H
    args = [x, y, 
            write_ftn_to_disc(norm, kwargs['tmpdir']),
            benvs,
            bottom_envs]
    results = parallelized_looped_function(function, iterate_over,
                                           args, kwargs)

    # Put stuff back into the bottom_envs object
    for i in range(len(results)):
        if results[i] is not None:
            tn = results[i][0]
            next_key = results[i][1]
            boundary_tag = results[i][2]
            applied_order = results[i][3]
            bottom_envs[next_key] = {'tn': tn,
                                     'boundary_tag': boundary_tag}
            if applied_order is not None:
                bottom_envs['op_order'][next_key] = applied_order

    return bottom_envs

def update_top_environments(x, y, H, norm, benvs, top_envs, peps_parity=None, tmpdir=None):
    """
    Update all the top norm and hamiltonian environments in a column

    Args:
        x: int
            The x position in the lattice
        y int
            The y position in the lattice
        H: List of OPTERMS
            The Hamiltonian
        norm: FermionicTN
            The norm containing the bra nad ket peps
        benvs: dict
            The boundary environments
        top_envs: dict
            The current set of abottom environments

    Kwargs:
        peps_parity: int
            The parity of the supplied peps
        tmpdir: str
            A temporary directory where intermediate files can be stored

    Returns:
        top_envs: dict
            The updated set of top environments
    """
    print(f'\t\t\tUpdating top boundary row {x}')
    top_envs = update_ham_top_environments(x, y, H, norm, 
                                           benvs, top_envs,
                                           peps_parity=peps_parity,
                                           tmpdir=tmpdir)
    top_envs = update_norm_top_environments(x, y, norm, 
                                            benvs, top_envs, 
                                            tmpdir=tmpdir)
    return top_envs

def update_bottom_environments(x, y, H, norm, benvs, bottom_envs, peps_parity=None, tmpdir=None):
    """
    Update all the bottom norm and hamiltonian environments in a column

    Args:
        x: int
            The x position in the lattice
        y int
            The y position in the lattice
        H: List of OPTERMS
            The Hamiltonian
        norm: FermionicTN
            The norm containing the bra nad ket peps
        benvs: dict
            The boundary environments
        bottom_envs: dict
            The current set of abottom environments

    Kwargs:
        peps_parity: int
            The parity of the supplied peps
        tmpdir: str
            A temporary directory where intermediate files can be stored

    Returns:
        bottom_envs: dict
            The updated set of bottom environments
    """
    print(f'\t\t\tUpdating bottom boundary row {x}')
    bottom_envs = update_ham_bottom_environments(x, y, H, norm,
                                                 benvs, bottom_envs,
                                                 peps_parity=peps_parity,
                                                 tmpdir=tmpdir)
    bottom_envs = update_norm_bottom_environments(x, y, norm, 
                                                  benvs, bottom_envs, 
                                                  tmpdir=tmpdir)
    return bottom_envs

def initialize_top_environments(x, y, H, norm, benvs, peps_parity=None, tmpdir=None):
    """
    Initialize all the top environments in a column

    Args:
        x: int
            The x position in the lattice
        y int
            The y position in the lattice
        H: List of OPTERMS
            The Hamiltonian
        norm: FermionicTN
            The norm containing the bra nad ket peps
        benvs: dict
            The boundary environments

    Kwargs:
        peps_parity: int
            The parity of the supplied peps
        tmpdir: str
            A temporary directory where intermediate files can be stored

    Returns:
        top_envs: dict
            A set of top environments
    """
    # Hold all top envs as a dict
    top_envs = dict()

    # Another dictionary tracks the order of gate application in each boundary
    top_envs['op_order'] = dict()

    # Initialize the top boundaries
    for row in range(norm.Lx-1, x-1, -1):
        top_envs = update_top_environments(row, y, H, norm, benvs, top_envs, peps_parity=peps_parity, tmpdir=tmpdir)

    # Return the result
    return top_envs

def initialize_bottom_environments(x, y, H, norm, benvs, peps_parity=None, tmpdir=None):
    """
    Initialize all the bottom environments in a column

    Args:
        x: int
            The x position in the lattice
        y int
            The y position in the lattice
        H: List of OPTERMS
            The Hamiltonian
        norm: FermionicTN
            The norm containing the bra nad ket peps
        benvs: dict
            The boundary environments

    Kwargs:
        peps_parity: int
            The parity of the supplied peps
        tmpdir: str
            A temporary directory where intermediate files can be stored

    Returns:
        bottom_envs: dict
            A set of bottom environments
    """
    # Hold all bottom envs as a dict
    bottom_envs = dict()

    # Another dictionary tracks the order of gate application in each boundary
    bottom_envs['op_order'] = dict()

    # Initialize the bottom boundaries
    for row in range(x+1):
        bottom_envs = update_bottom_environments(row, y, H, norm, benvs, bottom_envs, peps_parity=peps_parity, tmpdir=tmpdir)

    # Return the result
    return bottom_envs

def move_top_environments_up(x, y, H, norm, benvs, top_envs, peps_parity=None, replace_site=True):
    """
    Update the top environments of a column, moving upwards in the column, 
    only requires deleting unecessary environments
    """
    # Requires deleting environments that were already used
    rm_keys = [key for key in top_envs if key[0] == x]
    for key in rm_keys:
        removed_env = top_envs.pop(key)
        delete_ftn_from_disc(removed_env['tn'])
    return top_envs

def match_phase(ref_ten, target_ten, virtual=False):
    """
    Return a version of target_ten whose phase matches ref_ten
    """
    # Get the local inds that need flipping
    ref_local_inds = ref_ten.phase['local_inds'] if 'local_inds' in ref_ten.phase else []
    target_local_inds = target_ten.phase['local_inds'] if 'local_inds' in target_ten.phase else []
    local_flip_inds = [ind for ind in ref_local_inds if ind not in target_local_inds] + \
                      [ind for ind in target_local_inds if ind not in ref_local_inds]

    # See if needs a global flip
    ref_global_flip = ref_ten.phase['global_flip'] if 'global_flip' in ref_ten.phase else False
    target_global_flip = target_ten.phase['global_flip'] if 'global_flip' in target_ten.phase else False
    global_flip = (ref_global_flip != target_global_flip)

    # Now do the flipping
    target_ten = target_ten.flip(global_flip=global_flip,
                                 local_inds=local_flip_inds,
                                 inplace=virtual)

    # Return result
    return target_ten

def transfer_site_ten(x, y, ref_tn, target_tn, virtual=False):
    """
    Replace the bra and ket sites at (x, y) in the target_tn
    with the tensors from the ref_tn
    """
    # Copy supplied tensor networks
    if not virtual:
        ref_tn = ref_tn.copy()
        target_tn = target_tn.copy()

    # Get information for the reference bra/ket tensors
    ref_bra_site = ref_tn[(f'I{x},{y}','BRA')]
    ref_ket_site = ref_tn[(f'I{x},{y}','KET')]

    # Get information for the target bra/ket tensors
    target_bra_site = target_tn[(f'I{x},{y}','BRA')]
    target_ket_site = target_tn[(f'I{x},{y}','KET')]

    # Do a phase flip on the target site
    ref_bra_site = match_phase(target_bra_site, ref_bra_site)
    ref_ket_site = match_phase(target_ket_site, ref_ket_site)

    # Check if transpose has happened
    if ref_bra_site.inds != target_bra_site.inds:
        raise ValueError('Reference and target bra inds do not match')
    if ref_ket_site.inds != target_ket_site.inds:
        raise ValueError('Reference and target ket inds do not match')

    # Now replace the data
    target_bra_site.modify(data=ref_bra_site.data)
    target_ket_site.modify(data=ref_ket_site.data)

    return target_tn

def move_bottom_environments_up(x, y, H, norm, benvs, bottom_envs, replace_site=True, peps_parity=None, tmpdir=None):
    """
    Update the bottom environments of a column, moving upwards in the column, 
    and delet any no longer necessary environments

    Args:
        x: int
            The x position in the lattice
        y int
            The y position in the lattice
        H: List of OPTERMS
            The Hamiltonian
        norm: FermionicTN
            The norm containing the bra nad ket peps
        benvs: dict
            The boundary environments
        bottom_envs: dict
            The current set of bottom environments

    Kwargs:
        replace_site: bool
            If True, then the tensors from the norm are put into the
            bottom environment tensors before the environment is updated
        peps_parity: int
            The parity of the supplied peps
        tmpdir: str
            A temporary directory where intermediate files can be stored

    Returns:
        bottom_envs: dict
            An updated set of bottom environments
    """
    # First, replace updated site in all envs
    if replace_site:
        for key in bottom_envs:
            if type(key[0]) == int and key[0] > 0:
                new_tn = transfer_site_ten(x, y, norm,
                                           load_ftn_from_disc(bottom_envs[key]['tn']))
                bottom_envs[key]['tn'] = write_ftn_to_disc(new_tn, tmpdir)

    # Do the environment update
    bottom_envs = update_bottom_environments(x+1, y, H, norm, benvs, bottom_envs, peps_parity=peps_parity, tmpdir=tmpdir)

    # Delete used bottom environments
    rm_keys = [key for key in bottom_envs if key[0] == x]
    for key in rm_keys:
        removed_env = bottom_envs.pop(key)
        delete_ftn_from_disc(removed_env['tn'])
    
    # Return results
    return bottom_envs

def linearized_site(site_ten, constructor=None):
    """
    Convert a supplied site tensor to a vector using 
    the supplied constructor
    """
    return constructor.tensor_to_vector(site_ten.data)

def get_phys_bond_identity(symmetry):
    """
    Return an identity that acts on the physical sites 
    """
    if symmetry == Z2:
        return eye({Z2(0):2, Z2(1):2}, flat=True)
    elif symmetry == U1:
        return eye({U1(0):1, U1(1):2, U1(2):1}, flat=True)
    else:
        raise ValueError('Symmetry type not supported for physical bond identity')

def fermion_tensors_agree(ref, res, print_details=False):
    """
    Check to see if two fermion tensors are similar

    Args:
        ref: FermionTensor
            The reference
        res: FermionTensor
            The resulting tensor that is compared to the reference

    Kwargs:
        print_details: bool
            If true, then this will print details

    returns:
        agree: bool
            True if the tensors agree, false if they do not
    """
    ref = ref.data.to_sparse()
    res = res.data.to_sparse()

    ref_q_labels = [block.q_labels for block in ref.blocks]
    res_q_labels = [block.q_labels for block in res.blocks]

    q_labels = ref_q_labels if len(ref_q_labels) > len(res_q_labels) else res_q_labels

    total_diff = 0.
    for q_label in q_labels:
        if (q_label in ref_q_labels) and (q_label in res_q_labels):
            ref_block_ind = ref_q_labels.index(q_label)
            res_block_ind = res_q_labels.index(q_label)
            if ref.blocks[ref_block_ind].ravel().shape == res.blocks[res_block_ind].ravel().shape:
                total_diff += sum(abs(ref.blocks[ref_block_ind].ravel()-res.blocks[res_block_ind].ravel()))
                if print_details:
                    print('ref',ref.blocks[ref_block_ind].ravel())
                    print('res',res.blocks[res_block_ind].ravel())
            else:
                if print_details:
                    print('block shapes disagreed:')
                    print('ref',ref.blocks[ref_block_ind].ravel())
                    print('res',res.blocks[ref_block_ind].ravel())
                return False
        else:
            if q_label in ref_q_labels:
                ref_block_ind = ref_q_labels.index(q_label)
                total_diff += sum(abs(ref.blocks[ref_block_ind].ravel()))
            else:
                res_block_ind = res_q_labels.index(q_label)
                total_diff += sum(abs(res.blocks[res_block_ind].ravel()))
    if abs(total_diff) < 1e-12:
        return True
    else:
        return False

def local_norm(x, y, H, norm, benvs, 
               top_envs, bottom_envs, 
               dense=True, linop=True, 
               constructor=None, peps_parity=None, 
               symmetry=None):
    """
    Construct a local version of the norm

    Args:
        x: int
            x position of the tensor
        y: int
            y position of the tensor
        H: list of OPTERMS
            The Hamiltonian
        norm: FermionicTN
            The bra and ket combined into a fermionic tn
        benvs: dict
            The dictioanry containing the left and right 
            boundary environments
        top_envs: dict
            The dictionary containing the top
            boundary environments
        bottom_envs: dict
            THe dictionary containing the bottom
            boundary environments

    Kwargs:
        dense: bool
            Whether to return a dense tensor or a uncontracte tn
        linop: bool
            Whether to return a linear operator or dense tensor
        constructor:
            A constructor supplied to make the Hamiltonian
            if None, then this will generate a constructor
        peps_parity: int
            The parity of the supplied peps
        symmetry:
            The symmetry of the supplied peps

    Returns:
        ket_ten:
            The original ket_ten, returned for reference 
        ket_vec: 
            A vectorized fermionic ket vector to be used in the optimization
        Nloc: FTNLinearOperator
            The Local version of the norm as a linear operator
    """
    # Get the bottom boundary
    bottom_env = load_ftn_from_disc(bottom_envs[x, 'norm']['tn'])
    bottom_tag = bottom_envs[x, 'norm']['boundary_tag']
    bottom_tn = bottom_env.select(bottom_tag).copy()

    # Get the top boundary
    top_env = load_ftn_from_disc(top_envs[x, 'norm']['tn'])
    top_tag = top_envs[x, 'norm']['boundary_tag']
    top_tn = top_env.select(top_tag).copy()

    # Get the optimization row
    if x > 0:
        central_row = bottom_env.select(bottom_env.row_tag(x)).copy()
    else:
        central_row = top_env.select(top_env.row_tag(x)).copy()

    # Combine to create the local norm
    N = bottom_tn & central_row & top_tn

    # Get the info for bra & ket sites
    bra_ten = N[(f'I{x},{y}','BRA')]
    ket_ten = N[(f'I{x},{y}','KET')]
    bra_tid = bra_ten.get_fermion_info()[0]
    ket_tid = ket_ten.get_fermion_info()[0]

    # Reorder
    N.fermion_space._reorder_from_dict({bra_tid: len(N.tensors)-1,
                                        ket_tid: 0})
    N._refactor_phase_from_tids((bra_tid, ket_tid))

    # Reindex for contraction with identity
    bra_ten.reindex_({f'k{x},{y}': f'b{x},{y}'})

    # Pop out tensor
    N._pop_tensor(bra_tid)
    N._pop_tensor(ket_tid)

    # Copy so that we get new fermion space
    N = N.copy()

    # Combine with Identity
    identity_mat = get_phys_bond_identity(symmetry)
    I = FermionTensor(identity_mat.copy(),
                      inds = [f'b{x},{y}', f'k{x},{y}'],
                      tags = [f'I{x},{y}', f'COL{y}', f'ROW{x}', 'OPS'])
    N |= I

    # Contract 
    if dense:
        N = N ^ all

    # Convert to a linear operator
    if linop:
        N = FTNLinearOperator([N,], 
                              bra_ten.inds[::-1], 
                              ket_ten.inds, 
                              ket_ten.net_symmetry,
                              constructor=constructor)
        ket_vec = linearized_site(ket_ten, N.constructor)

        return ket_ten, ket_vec, N
    else:
        return N

def local_ham_term(Hi, x, y, norm, benvs, 
                   top_envs, bottom_envs,
                   dense=True, tmpdir=None,
                   peps_parity=None,
                   symmetry=None):
    """
    Construct a local version of the Hamiltonian for a single Hamiltonian term

    Args:
        x: int
            x position of the tensor
        y: int
            y position of the tensor
        Hi:  OPTERMS
            The Hamiltonian term
        norm: FermionicTN
            The bra and ket combined into a fermionic tn
        benvs: dict
            The dictioanry containing the left and right 
            boundary environments
        top_envs: dict
            The dictionary containing the top
            boundary environments
        bottom_envs: dict
            THe dictionary containing the bottom
            boundary environments

    Kwargs:
        dense: bool
            Whether to return a dense tensor or a uncontracted tensor network
        tmpdir: str
            The temporary directory used to store intermediate files
        peps_parity: int
            THe parity of the supplied peps
        symmetry: 
            The type of symmetry used in the PEPS

    Returns:
        bra_ten:
            The original bar_ten, returned for reference 
        ket_ten:
            The original ket_ten, returned for reference 
        Hloc: FermionicTN
            The Local version of the Hamiltonian term
    """
    # Get the norm tn
    norm = load_ftn_from_disc(norm)
    
    # Get the environment keys
    key = [Hi.optags[site] for site in Hi.sites]
    key.sort()
    key = (x, tuple(key))

    if x > 0:
        # Get the bottom half of the local tn
        bottom_env = load_ftn_from_disc(bottom_envs[key]['tn'])
        bottom_tag = bottom_envs[key]['boundary_tag']
        Hi_tn = bottom_env.select((bottom_tag, bottom_env.row_tag(x)), which='any').copy()
        bottom_order = bottom_envs['op_order'][key] if key in bottom_envs['op_order'] else []
        bottom_order = [site for site in bottom_order if site in get_operator_sites(Hi_tn)]

        # Add the top half of the local tn
        top_env = load_ftn_from_disc(top_envs[key]['tn'])
        top_tag = top_envs[key]['boundary_tag']
        top_tn = top_env.select(top_tag).copy()
        top_order = top_envs['op_order'][key] if key in top_envs['op_order'] else []
        top_order = [site for site in top_order if site in get_operator_sites(top_tn)]
        Hi_tn = Hi_tn & top_tn

    else:
        # Get the bottom half of the local tn
        bottom_env = load_ftn_from_disc(bottom_envs[key]['tn'])
        bottom_tag = bottom_envs[key]['boundary_tag']
        Hi_tn = bottom_env.select(bottom_tag).copy()
        bottom_order = bottom_envs['op_order'][key] if key in bottom_envs['op_order'] else []
        bottom_order = [site for site in bottom_order if site in get_operator_sites(Hi_tn)]

        # Add the top half of the local tn
        top_env = load_ftn_from_disc(top_envs[key]['tn'])
        top_tag = top_envs[key]['boundary_tag']
        top_tn = top_env.select((top_tag, top_env.row_tag(x)), which='any').copy()
        top_order = top_envs['op_order'][key] if key in top_envs['op_order'] else []
        top_order = [site for site in top_order if site in get_operator_sites(top_tn)]
        Hi_tn = Hi_tn & top_tn

    # Check if we need a flip
    applied_order = bottom_order + top_order
    if (len(bottom_order) < len(Hi.sites) and # Incomplete bottom term
        len(top_order) < len(Hi.sites) and # Incomplete top term
        len(bottom_order) + len(top_order) == len(Hi.sites) and # Complete full term
        all([(site[1] == y) for site in Hi.sites]) and # Must be an op in this column
        ((x, y) not in Hi.sites) and 
        unpack_swap_phase(applied_order, Hi.sites) == 1):
        Hi_tn['I0,0','KET'].data._global_flip()

    # Get the info for the bra & ket sites
    bra_ten = Hi_tn[(f'I{x},{y}','BRA')]
    ket_ten = Hi_tn[(f'I{x},{y}','KET')]
    bra_tid = bra_ten.get_fermion_info()[0]
    ket_tid = ket_ten.get_fermion_info()[0]

    # Reindex for contraction with operator
    bra_ten.reindex_({f'k{x},{y}': f'b{x},{y}'})

    # Add operator
    if (x, y) in Hi.sites:

        # Put the operator into the tn
        Hi_tn,_ = insert_op_into_tn(x, y, Hi_tn, 
                                    Hi.get_op((x, y)).copy(),
                                    applied_order,
                                    Hi.sites,
                                    peps_parity,
                                    contract=False)

    else:

        # Put an identity operator in
        identity_mat = get_phys_bond_identity(symmetry)
        I = FermionTensor(identity_mat.copy(),
                          inds = [f'b{x},{y}', f'k{x},{y}'],
                          tags = [f'I{x},{y}', f'COL{y}', f'ROW{x}', 'OPS'])

        Hi_tn,_ = insert_op_into_tn(x, y, Hi_tn, 
                                    I,
                                    Hi.sites,
                                    Hi.sites,
                                    peps_parity,
                                    contract=False)

    # Reorder
    Hi_tn.fermion_space._reorder_from_dict({bra_tid: len(Hi_tn.tensors)-1,
                                            ket_tid: 0})
    Hi_tn._refactor_phase_from_tids((bra_tid, ket_tid))

    # Pop out tensors
    bra_ten = Hi_tn._pop_tensor(bra_tid)
    ket_ten = Hi_tn._pop_tensor(ket_tid)

    # Copy so that we get new fermion space
    Hi_tn = Hi_tn.copy()

    # Contract
    if dense:
        Hi_tn = Hi_tn ^ all

    # Multiply by prefactor
    Hi_tn *= Hi.prefactor

    # Write the resulting tensor to disc
    Hi_tn = write_ftn_to_disc(FermionTensorNetwork([Hi_tn]), tmpdir)
    bra_ten = write_ftn_to_disc(FermionTensorNetwork([bra_ten]), tmpdir)
    ket_ten = write_ftn_to_disc(FermionTensorNetwork([ket_ten]), tmpdir)

    return bra_ten, ket_ten, Hi_tn

def local_ham(x, y, H, norm, benvs, 
              top_envs, bottom_envs,
              linop=True, constructor=None,
              **kwargs):
    """
    Construct a local version of the Hamiltonian

    Args:
        x: int
            x position of the tensor
        y: int
            y position of the tensor
        H: list of OPTERMS
            The Hamiltonian
        norm: FermionicTN
            The bra and ket combined into a fermionic tn
        benvs: dict
            The dictioanry containing the left and right 
            boundary environments
        top_envs: dict
            The dictionary containing the top
            boundary environments
        bottom_envs: dict
            THe dictionary containing the bottom
            boundary environments

    Kwargs:
        linop: bool
            Whether to return a linear operator or dense tensor
        constructor:
            A constructor supplied to make the Hamiltonian
            if None, then this will generate a constructor

    Returns:
        ket_ten:
            The original ket_ten, returned for reference 
        ket_vec: 
            A vectorized fermionic ket vector to be used in the optimization
        Hloc: FTNLinearOperator
            The Local version of the Hamiltonian as a linear operator
    """
    # Construct the local hamiltonian term-by-term (parallelized)
    function = local_ham_term
    iterate_over = H
    args = [x, y,
            write_ftn_to_disc(norm, kwargs['tmpdir']),
            benvs,
            top_envs,
            bottom_envs]
    results = parallelized_looped_function(function, iterate_over,
                                           args, kwargs)

    # Put terms into a single hamiltonian
    Hloc = None
    for i in range(len(results)):
        if results[i] is not None:
            bra_ten = results[i][0]
            ket_ten = results[i][1]
            Hi_tn = results[i][2]

            Hi_tn = load_ftn_from_disc(Hi_tn).tensors[0]
            bra_ten = load_ftn_from_disc(bra_ten).tensors[0]
            ket_ten = load_ftn_from_disc(ket_ten).tensors[0]

            if Hloc is None:
                Hloc = Hi_tn
            else:
                Hloc += Hi_tn

    # Convert to a linear operator (if needed)
    if linop:
        Hloc = FTNLinearOperator([Hloc,], 
                                 bra_ten.inds[::-1],
                                 ket_ten.inds, 
                                 ket_ten.net_symmetry,
                                 constructor=constructor)
        ket_vec = linearized_site(ket_ten, Hloc.constructor)

        # Return result
        return ket_ten, ket_vec, Hloc
    else:
        return Hloc

def replace_norm_site(x, y, norm, ket_vec, ket_ten,
                      constructor=None, linearized=True,
                      virtual=False, linear_combination_step_size=1.):
    """
    Replace the bra and ket tensors in norm with the ket tensor supplied
    in ket_vec. 

    Args:
        x: int
            x position of the tensor
        y: int
            y position of the tensor
        norm: FermionicTN
            The bra and ket combined into a fermionic tn
        ket_vec: 
            A vectorized fermionic ket vector to be placed into
            the norm
        ket_ten:
            The original ket_ten, used to do the optimization,

    Kwargs:
        Constructor:
            The constructor used to make the ket_vec 
        virtual: bool  
            Whether the norm should be copied (virtual == False)
            or replaced in place (virtual == True)
        linear_combination_step_size: float
            Whether to take a linear combination of the previous
            ket tensor and the optimized ket tensor

    Returns:
        Norm: FermionTN
            The norm tensor network with the tensors at site (x, y)
            replaced
    """
    # Copy the norm
    if not virtual:
        norm = norm.copy()

    # Get information for the bra/ket tensors
    replace_ket_ten = norm[(f'I{x},{y}','KET')]
    replace_bra_ten = norm[(f'I{x},{y}','BRA')]

    # Convert the provided vector into a tensor
    ket_vec = constructor.vector_to_tensor(ket_vec, ket_ten.data.dq)

    # Do a small step
    linear_combination_theta = np.pi/2.*linear_combination_step_size
    ket_vec = np.sin(linear_combination_theta)*ket_vec + \
              np.cos(linear_combination_theta)*ket_ten.data

    # Put into the fermion tensor
    ket_ten.modify(data=ket_vec)

    # Get a corresponding bra tensor
    bra_ten = ket_ten.H
    bra_ten.reindex_(dict(zip([ind for ind in bra_ten.inds if ind[0] == '_'],
                              [ind+'*' for ind in bra_ten.inds if ind[0] == '_'])))

    # Update the phase on the input tensors
    ket_ten = match_phase(replace_ket_ten, ket_ten)
    bra_ten = match_phase(replace_bra_ten, bra_ten)

    # Put the data into the tensors
    replace_ket_ten.modify(data=ket_ten.data)
    replace_bra_ten.modify(data=bra_ten.data)

    # return result
    return norm

def evaluate_linop_expectation(ket, linop):
    """
    Provided a linear operator 'linop' and a linearized ket 'ket', this
    returns the expectation value computed with the linop
    """
    # Convert the linearized ket to a FermionTensor
    ket_data = linop.vector_to_tensor(ket)
    ket_ten = FermionTensor(ket_data, inds=linop.right_inds)

    # Put ket into linop tn
    fs, tensors = linop.get_contraction_kits()
    tensors.append(ket_ten)
    fs.insert_tensor(0, ket_ten, virtual=True)

    # Get the bra tensor
    bra_ten = ket_ten.H
    bra_ten.reindex_(dict(zip([ind for ind in bra_ten.inds if ind[0] == '_'],
                              [ind+'*' for ind in bra_ten.inds if ind[0] == '_'])))
    bra_ten.reindex_(dict(zip([ind for ind in bra_ten.inds if ind[0] == 'k'],
                              ['b'+ind[1:] for ind in bra_ten.inds if ind[0] == 'k'])))

    # Combine into a tn
    tensors.append(bra_ten)
    fs.insert_tensor(len(tensors)-1, bra_ten, virtual=True)

    # Set up the contraction
    expr = tensor_contract(*tensors, optimize=linop.optimize, **linop._kws)
    E = _launch_fermion_expression(expr, tensors, backend=linop.backend, inplace=True)

    return E

def local_energy(ket, Hi, Ni, normalize=True):
    """
    Calculate the energy of the localized site 'ket' 
    Hamiltonian 'Hi', and norm 'Ni'
    """
    # Evaluate energy
    E = evaluate_linop_expectation(ket, Hi)
    # Evaluate N
    if normalize:
        N = evaluate_linop_expectation(ket, Ni)
        return E/N
    return E

def calc_energy(peps, H, chi, write_to_disc=False, around=(0,0), symmetry=None):
    """
    Do a boundary contraction to get the full system energy

    Args:
        peps: FermionicPEPS
            The state for which we are calulating the energy
        H: List of OPTERMS
            The Hamiltonian
        chi: int
            The maximum bond dimension of the boundary mps

    Kwargs:
        write_to_disc: bool 
            Whether or not to write intermediate variables to disc
        around: tuple of two ints
            The site around which to do the boundary contraction, (x, y)
        symmetry: 
            The symmetry of the supplied peps

    Return:
        E0: the resulting normalized energy
    """
    # Set up temporary directory to write intermediates to
    if write_to_disc:
        tmpdir = create_rand_tmpdir()
    else:
        tmpdir = None

    # Figure out parity of initial peps
    peps_parity = peps_total_parity(peps, supplied_norm=False)

    # Combine the bra with a ket
    norm = peps.make_norm()
    norm = norm.reorder('col', layer_tags=('KET', 'BRA'))

    # Do a boundary contraction around the given column
    norm, benvs = initialize_boundary_environments(around[1], H, norm, 
                                                   chi=chi,
                                                   peps_parity=peps_parity, 
                                                   tmpdir=tmpdir)

    # Contract around this row
    top_envs = initialize_top_environments(around[0], around[1], H, norm, benvs,
                                           peps_parity=peps_parity,
                                           tmpdir=tmpdir)
    bottom_envs = initialize_bottom_environments(around[0], around[1], H, norm, benvs,
                                                 peps_parity=peps_parity,
                                                 tmpdir=tmpdir)

    # Get into a local Norm/Hamiltonian
    ket0_ten, ket0_vec, Ni = local_norm(around[0], around[1], H, 
                                        norm, benvs,
                                        top_envs, bottom_envs,
                                        peps_parity=peps_parity, 
                                        symmetry=symmetry)
    _, _, Hi = local_ham(around[0], around[1], H, 
                         norm, benvs, 
                         top_envs, bottom_envs,
                         constructor=Ni.constructor, 
                         peps_parity=peps_parity, 
                         symmetry=symmetry)

    # Compute energy
    E0 = local_energy(ket0_vec, Hi, Ni)

    # Return the result
    return E0

#####################################################################################
# BASIC DMRG FUNCTIONS
#####################################################################################

def linop_to_mat(linop):
    '''
    Hacky way to convert a FTNLinearOperator to a 
    dense square numpy matrix
    '''
    matdim = linop.shape[0]
    mat = np.zeros([matdim, matdim])
    for i in range(matdim):
        vec = np.zeros([matdim])
        vec[i] = 1.
        matvec = linop._matvec(vec)
        mat[:,i] = matvec
    return mat

def get_param_file():
    """
    Returns a file 'tmpdir/running_parameters'
    where the dynamically updatable optimization
    parameters are stored
    """
    param_file = create_rand_tmpdir()
    if param_file[-1] != '/':
        param_file = param_file + '/'
    param_file = param_file + 'running_parameters'
    return param_file

def load_dynamical_parameters():
    """
    This loads parameters from the file created by 
    'save_dynamical_parameters', meaning if you alter
    the information in that file, then you will 
    be able to update optimization parameters while 
    the calculation is running
    """
    # Get the file name with these parameters
    param_file = get_param_file()
    
    # Load parameters if file is there
    if os.path.isfile(param_file):
        with open(param_file, 'rb') as f:
            running_parameters = pickle.load(f)
        return running_parameters
    else:
        return None

def save_dynamical_parameters(eig_step_size=0.1, eig_maxiter=1000,
                              eig_tol=1e-8, eig_k=1, 
                              eig_which='SA', eig_backend='lobpcg', 
                              eig_fallback_to_scipy=True,
                              eig_ncv=10, add_noise=0.,
                              allow_energy_increase=False,
                              chi=-1, **other_stuff):
    """
    This writes all of the optimization parameters
    to a temporary file in 'tmpdir' so that they 
    can be updated
    """
    # Get the file name with these parameters
    param_file = get_param_file()

    # Put all parameters into a dictionary
    running_parameters = {'eig_step_size': eig_step_size,
                          'eig_maxiter': eig_maxiter,
                          'eig_tol': eig_tol,
                          'eig_k': eig_k,
                          'eig_which': eig_which,
                          'eig_backend': eig_backend,
                          'eig_fallback_to_scipy': eig_fallback_to_scipy,
                          'allow_energy_increase': allow_energy_increase,
                          'eig_ncv': eig_ncv,
                          'chi': chi,
                          'add_noise': add_noise}
    print('saving parameters')
    print(running_parameters)
    with open(param_file, 'wb') as f:
        pickle.dump(running_parameters, f)

def dmrg_site_update(x, y, H, norm, benvs, top_envs, bottom_envs, 
                     eig_step_size=0.1, tmpdir=None, start_time=None,
                     eig_maxiter=1000, eig_tol=1e-8, eig_k=1,
                     eig_which='SA', eig_backend='lobpcg',
                     eig_fallback_to_scipy=True,
                     eig_ncv=10, add_noise=0.,
                     symmetry=None, peps_parity=None, 
                     check_hermiticity=False,
                     check_site_replacement=True,
                     allow_energy_increase=False,
                     check_dense_eigenproblem=False):
    """
    Do a local update of the peps tensors at sites x and y

    Args:
        x: int
            The x location to perform the site update
        y: int
            The y location to perform the site update
        H: list of OPTERMS
            The Hamiltonian
        norm: FermionicTN
            The norm TN containing the PEPS bra and ket
        benvs: dictionary
            The dictionary containing all of the 
            left and right boundary environmnets
        top_envs: dictionary
            The dictionary containing all of the top
            boundary environments for this column
        bottom_envs: dictionary
            The dictionary containing all of the bottom
            boundary environments for this column

    Kwargs:
        eig_step_size: float
            How large of a step to take in the direction of the solution
            of the local eigenproblem
        tmpdir: str
            The temporary directory where intermediate files
            can be stored
        start_time: float
            The starting time of this calcualtion (used to print
            out computational times)
        eig_maxiter: int
            The number of iterations that the eigensolver
            can take
        eig_tol: float
            The tolerance for the solution of the local eigenproblem
        eig_k: int
            the k parameter supplied to the local iterative eigensolver
        eig_which: str
            Supplied as the which parameter to the local eigensolver,
            indicating which eigenvalue to target
        eig_backend: str
            Which eigensolver backend to use, see those available
            in quimb
        eig_fallback_to_scipy: bool
            Whether to use scipy if the intended eigensolver fails
        eig_ncv: int
            The ncv parameter supplied to the local eigenproblem solver
        add_noise: float
            Random noise is added with maximum magnitude 'add_noise'
            to the solved eigensolution
        symmetry: str
            The symmetry used for the peps (i.e. U1, Z2)
        peps_parity: int
            The parity of the supplied peps
        check_hermiticity: bool 
            Check whether the local Hamiltonian and norm are hermitian
        check_site_replacement: bool    
            A debugging check
        allow_energy_increase: bool
            If True, then the energy can increase, otherwise, any time
            the eigensolver results in a higher energy, it will be
            discarded and the previous local state will be kept
        check_dense_eigenproblem: bool
            A debugging tool to run the dense eigensolver instead
            of the iterative version

    Returns:
        Ei: float
            The resulting energy
        norm: FermionicTN
            The resulting norm TN with the updated PEPS bra and ket included
    """

    # Get updated parameters
    params = load_dynamical_parameters()
    eig_step_size         = params['eig_step_size']
    eig_maxiter           = params['eig_maxiter']
    eig_tol               = params['eig_tol']
    eig_k                 = params['eig_k']
    eig_which             = params['eig_which']
    eig_backend           = params['eig_backend']
    eig_fallback_to_scipy = params['eig_fallback_to_scipy']
    eig_ncv               = params['eig_ncv']
    add_noise             = params['add_noise']
    allow_energy_increase = params['allow_energy_increase']

    # Get local versions of the norm and hamiltonian
    print(f'\t\t\tGetting local norm at ({x}, {y}) ({time.time()-start_time} s)')
    ket0_ten, ket0_vec, Ni = local_norm(x, y, H, 
                                        norm, benvs, 
                                        top_envs, bottom_envs,
                                        peps_parity=peps_parity,
                                        symmetry=symmetry)

    print(f'\t\t\tGetting local ham at ({x}, {y}) ({time.time()-start_time} s)')
    _, _, Hi = local_ham(x, y, H,
                         norm, benvs,
                         top_envs, bottom_envs,
                         constructor=Ni.constructor,
                         peps_parity=peps_parity,
                         tmpdir=tmpdir,
                         symmetry=symmetry)

    # Check local hamiltonia/norm
    if check_hermiticity:
        print('Local Ham Hermitian?',np.all(np.isclose(linop_to_mat(Hi), 
                                             linop_to_mat(Hi).conj().transpose())))
        print('Local Norm Hermitian?',np.all(np.isclose(linop_to_mat(Ni), 
                                             linop_to_mat(Ni).conj().transpose())))

    # Compute the local energy (for reference)
    print(f'\t\t\tEvaluating Initial Energy ({time.time()-start_time} s)')
    E0 = local_energy(ket0_vec, Hi, Ni)
    print(f'\t\t\tInitial Energy = {E0} ({time.time()-start_time} s)')

    # Compute the local energy from dense matrices (for reference)
    if check_dense_eigenproblem:
        import scipy.linalg as sla
        Ei_mat, keti_mat = sla.eigh(linop_to_mat(Hi), b=linop_to_mat(Ni))
        inds = np.argsort(Ei_mat)
        Ei_mat = Ei_mat[inds[0]]
        keti_mat = keti_mat[:, inds[0]]
        print(f'\t\t\tDense energy at {(x, y)} = {Ei_mat}')

    # Solve the local eigenproblem
    print(f'\t\t\tSolving Eigenproblem ({time.time()-start_time} s)')
    Ei_eig, keti_vec = eigh(Hi, B=Ni, v0=ket0_vec,
                            tol=eig_tol, maxiter=eig_maxiter,
                            ncv=eig_ncv, k=eig_k, which=eig_which,
                            backend=eig_backend, 
                            fallback_to_scipy=eig_fallback_to_scipy)
    Ei_eig = Ei_eig[0]
    keti_vec = keti_vec[:, 0]
    print(f'\t\t\tEigenproblem energy: {Ei_eig}')

    # Normalize
    print(f'\t\t\tNormalizing Result ({time.time()-start_time} s)')
    normval = evaluate_linop_expectation(keti_vec, Ni)
    keti_vec /= normval ** (1./2.)

    # Take a small step in that direction
    print(f'\t\t\tTaking Small Step ({time.time()-start_time} s)')
    theta = np.pi/2.*eig_step_size
    keti_vec = np.sin(theta)*keti_vec + np.cos(theta)*ket0_vec

    # Normalize
    print(f'\t\t\tNormalizing after step ({time.time()-start_time} s)')
    normval = evaluate_linop_expectation(keti_vec, Ni)
    keti_vec /= normval ** (1./2.)

    # Evaluate energy
    print(f'\t\t\tEvaluating resulting energy ({time.time()-start_time} s)')
    Ei = local_energy(keti_vec, Hi, Ni)

    # If energy is not between initial energy 
    # and eigen energy, then take one more eigensolver
    # step (this seems to solve this problem with lobpcg)
    if ((Ei < Ei_eig and Ei < E0) or (Ei > Ei_eig and Ei > E0)) \
       and eig_fallback_to_scipy and not np.isclose(eig_step_size, 1.):
        print(f'\t\t\t! Linear Combination failed, taking additional eigsolver step')
        Ei_eig, keti_vec = eigh(Hi, B=Ni, v0=keti_vec,
                                tol=eig_tol, maxiter=1,
                                ncv=eig_ncv, k=eig_k, which=eig_which,
                                backend=eig_backend, 
                                fallback_to_scipy=eig_fallback_to_scipy)
        Ei_eig = Ei_eig[0]
        keti_vec = keti_vec[:, 0]
        
        # Normalize
        normval = evaluate_linop_expectation(keti_vec, Ni)
        keti_vec /= normval ** (1./2.)

        # Take a small step in that direction
        theta = np.pi/2.*eig_step_size
        keti_vec = np.sin(theta)*keti_vec + np.cos(theta)*ket0_vec

        # Normalize
        normval = evaluate_linop_expectation(keti_vec, Ni)
        keti_vec /= normval ** (1./2.)

        # Evaluate energy
        Ei = local_energy(keti_vec, Hi, Ni)

        if ((Ei < Ei_eig and Ei < E0) or (Ei > Ei_eig and Ei > E0)):
            print(f'\t\t\t!! 2nd Linear Combination Failed too!')

    # Check whether or not to keep state 
    if (not allow_energy_increase) and (Ei > E0):
        Ei = E0
        keti_vec = ket0_vec
        print(f'\t\t\tEnergy Increase Not Allowed! Keeping Previous State')

    # Add some noise
    print(f'\t\t\tEnergy After Step = {Ei} ({time.time()-start_time} s)')
    keti_vec += add_noise * np.max(np.abs(keti_vec)) * np.random.random(keti_vec.shape)
    Ei = local_energy(keti_vec, Hi, Ni)
    print(f'\t\t\tEnergy After Noise = {Ei} ({time.time()-start_time} s)')


    # Print the energy
    if start_time is None:
        print(f'\t\t\tLocal E({x},{y}) = {Ei}')
    else:
        print(f'\t\t\tLocal E({x},{y}) = {Ei} ({time.time()-start_time} s)')


    # Put the resulting peps tensor back into the norm
    print(f'\t\t\tReplacing Norm Site ({time.time()-start_time} s)')
    norm = replace_norm_site(x, y, norm, keti_vec, ket0_ten, 
                             constructor=Ni.constructor)

    # Check to ensure replaced site energy is correct
    if check_site_replacement:
        ket0_ten, ket0_vec, Ni = local_norm(x, y, H, 
                                            norm, benvs, 
                                            top_envs, bottom_envs,
                                            peps_parity=peps_parity,
                                            symmetry=symmetry)
        _, _, Hi = local_ham(x, y, H,
                             norm, benvs,
                             top_envs, bottom_envs,
                             constructor=Ni.constructor,
                             peps_parity=peps_parity,
                             tmpdir=tmpdir,
                             symmetry=symmetry)
        _Ei = local_energy(keti_vec, Hi, Ni)
        print(f'\t\t\tReplaced Norm Site Energy = {_Ei} ({time.time()-start_time} s)')

    # Return the resulting energy & state
    return Ei, norm

def dmrg_column_sweep(y, H, norm, benvs, peps, peps_parity=None, tmpdir=None, start_time=None, save_peps_loc=None, **site_update_args):
    """
    Run a sweep through all of the tensors in a single column 
    of the supplied peps, doing a DMRG update for each one iteratively. 

    Args:
        y: int
            The column to be optimized
        H: List of OPTERMS
            The Hamiltonian expressed as a list of OPTERMS
        norm: FermionicTN
            The norm of the supplied PEPS (includes bra and ket)
        benvs: dictionary
            The dictionary containing all of the boundary environments
            around this column
        peps: FermionicPEPS
            The PEPS that is being optimized

    Kwargs:
        peps_parity: int
            The parity of the supplied peps
        tmpdir: str
            The location of a temporary directory where intermediate
            files will be stored
        start_time: float
            The start time for the calculation
        save_peps_loc: str
            A filename where the optimized peps will be written after 
            each optimization step

    Returns:
        E: float
            The energy of the optimized peps
        peps: FermionicTN
            The resulting optimized PEPS bra and ket tensors stored as a norm
    """
    # Initialize top and bottom environments
    x = 0
    print(f'\t\tInitializing top column environments ({time.time()-start_time} s)')
    top_envs = initialize_top_environments(x, y, H, norm, benvs, peps_parity=peps_parity, tmpdir=tmpdir)
    print(f'\t\tInitializing bottom column environments ({time.time()-start_time} s)')
    bottom_envs = initialize_bottom_environments(x, y, H, norm, benvs, peps_parity=peps_parity, tmpdir=tmpdir)
    
    # Loop through rows doing updates
    for x in range(norm.Lx):
        print(f'\t\tStarting site update ({x},{y}) ({time.time()-start_time} s)')

        # Do the site update
        E, norm = dmrg_site_update(x, y, H, norm, benvs, 
                                   top_envs, bottom_envs, 
                                   peps_parity=peps_parity, 
                                   tmpdir=tmpdir, 
                                   start_time=start_time,
                                   **site_update_args)

        # Update the top/bottom environments
        print(f'\t\tMoving Top Environments up ({x},{y}) ({time.time()-start_time} s)')
        top_envs = move_top_environments_up(x, y, H, norm, benvs, top_envs, peps_parity=peps_parity)
        print(f'\t\tMoving Bottom Environments down ({x},{y}) ({time.time()-start_time} s)')
        bottom_envs = move_bottom_environments_up(x, y, H, norm, benvs, bottom_envs, peps_parity=peps_parity, tmpdir=tmpdir)

        # Save the current result
        if save_peps_loc is not None:
            peps = FermionTensorNetwork(norm['KET']).view_like_(peps)
            write_ftn_to_disc(peps, save_peps_loc+str(time.time()), provided_filename=True) 

    # Clear the top/bottom environments from disc
    print(f'\t\tRemoving old top environments ({time.time()-start_time} s)')
    remove_env_from_disc(top_envs)
    print(f'\t\tRemoving old bottom environments ({time.time()-start_time} s)')
    remove_env_from_disc(bottom_envs)

    # Return the result
    return E, norm

def dmrg_sweep_right(H, peps, chi=-1, benvs=None, peps_parity=None, tmpdir=None, save_peps_loc=None, **sweep_args):
    """
    Run a right sweep of the DMRG algorithm for a fermionic PEPS with the 
    Hamiltonian supplied as a list of OPTERMs 

    Args:
        H: List of OPTERMS
            The Hamiltonian expressed as a list of OPTERMS
        peps: FermionicPEPS
            An initial guess to be used in the DMRG optimization

    Kwargs:
        chi: int
            The maximum bond dimension to be used in boundary contractions
        benvs: dict or None
            If dict, then the boundary environments contained in the 
            dictionary will be reused, otherwise new ones will be 
            generated
        peps_parity: int
            The parity of the supplied peps
        tmpdir: str
            The location of a temporary directory where intermediate
            files will be stored
        save_peps_loc: str
            A filename where the optimized peps will be written after 
            each optimization step

    Returns:
        E: float
            The energy of the optimized peps
        peps: FermionicPEPS
            The resulting optimized PEPS
        benvs: dict
            The dictionary containing all of the boundary environments
            calculated during this sweep
    """
    # Get a norm object
    norm = peps.make_norm()
    norm = norm.reorder("col", layer_tags=('KET', 'BRA'))

    # Load parameters (in case chi has been updated)
    params = load_dynamical_parameters()
    chi = params['chi']

    # Compute the initial boundary environments
    if benvs is None:
        print(f'\tInitializing Boundary Environments with chi={chi}')
        norm, benvs = initialize_boundary_environments(0, H, norm, chi=chi, peps_parity=peps_parity, tmpdir=tmpdir)

    # Do optimization in each column
    for y in range(peps.Ly):
        print('\t\tDoing right sweep over rows in column {}'.format(y))

        # Optimize each column
        E, norm = dmrg_column_sweep(y, H, norm, benvs, peps, peps_parity=peps_parity, tmpdir=tmpdir, save_peps_loc=save_peps_loc, **sweep_args)

        # Update environments
        if y != peps.Ly-1:
            benvs = move_benvs_right(y, H, norm, benvs, chi=chi, peps_parity=peps_parity, tmpdir=tmpdir)

    # Get a peps back
    peps = FermionTensorNetwork(norm['KET']).view_like_(peps)

    # Save the result
    if save_peps_loc is not None:
        write_ftn_to_disc(peps, save_peps_loc, provided_filename=True) 

    # Return result
    return E, peps, benvs

def dmrg_sweep_left(H, peps, chi=-1, benvs=None, peps_parity=None, tmpdir=None, save_peps_loc=None, **sweep_args):
    """
    Run a left sweep of the DMRG algorithm for a fermionic PEPS with the 
    Hamiltonian supplied as a list of OPTERMs 

    Args:
        H: List of OPTERMS
            The Hamiltonian expressed as a list of OPTERMS
        peps: FermionicPEPS
            An initial guess to be used in the DMRG optimization

    Kwargs:
        chi: int
            The maximum bond dimension to be used in boundary contractions
        benvs: dict or None
            If dict, then the boundary environments contained in the 
            dictionary will be reused, otherwise new ones will be 
            generated
        peps_parity: int
            The parity of the supplied peps
        tmpdir: str
            The location of a temporary directory where intermediate
            files will be stored
        save_peps_loc: str
            A filename where the optimized peps will be written after 
            each optimization step

    Returns:
        E: float
            The energy of the optimized peps
        peps: FermionicPEPS
            The resulting optimized PEPS
        benvs: dict
            The dictionary containing all of the boundary environments
            calculated during this sweep
    """
    # Get a norm object
    norm = peps.make_norm()
    norm = norm.reorder("col", layer_tags=('KET', 'BRA'))

    # Load parameters (in case chi has been updated)
    params = load_dynamical_parameters()
    chi = params['chi']

    # Compute the initial boundary environments
    if benvs is None:
        print(f'\tInitializing Boundary Environments with chi={chi}')
        norm, benvs = initialize_boundary_environments(peps.Ly-1, H, norm, chi=chi, peps_parity=peps_parity, tmpdir=tmpdir)

    # Loop through all columns
    for y in reversed(range(peps.Ly)):
        print('\t\tDoing left sweep over rows in column {}'.format(y))

        # Optimize each column
        E, norm = dmrg_column_sweep(y, H, norm, benvs, peps, peps_parity=peps_parity, tmpdir=tmpdir, save_peps_loc=save_peps_loc, **sweep_args)

        # Update environments
        if y != 0:
            benvs = move_benvs_left(y, H, norm, benvs, chi=chi, peps_parity=peps_parity, tmpdir=tmpdir)

    # Get a peps back
    peps = FermionTensorNetwork(norm['KET']).view_like_(peps)

    # Save the result
    if save_peps_loc is not None:
        write_ftn_to_disc(peps, save_peps_loc, provided_filename=True) 

    # Return result
    return E, peps, benvs

def dmrg(H, peps, 
         chi = -1,
         maxiter = 1000, 
         conv_tol = 1e-5,
         write_to_disc = False,
         balance_bonds = False,
         equalize_norms = False,
         start_time = None,
         **sweep_args):
    """
    Run the DMRG algorithm for a fermionic PEPS with the 
    Hamiltonian supplied as a list of OPTERMs 

    Args:
        H: List of OPTERMS
            The Hamiltonian expressed as a list of OPTERMS
        peps: FermionicPEPS
            An initial guess to be used in the DMRG optimization

    Kwargs:
        chi: int
            The maximum bond dimension to be used in boundary contractions
        maxiter: int
            The maximum number of sweeps through the lattice
        conv_tol: float
            The tolerance for convergence of the sweeps
        write_to_disc: bool
            If True, then intermediates will be written to disc
            otherwise, all will be stored in memory
        balance_bonds: bool
            This is a procedure that is run between each sweep
            to balance factors within the bonds of the peps and 
            is generally used to increase stability
        equalize_norms: bool
            This is a procedure that is run between each sweep
            to make the norms of tensors in the lattice equal and
            is generally used to incrase calcualtion stability
        start_time: float
            A start time for the calculation (this will be automatically
            generated if none is supplied)
        sweep_args: dict
            A dictionary of further arguments that will be passed to the sweep function

    Returns:
        E: float
            The energy of the optimized peps
        peps: FermionicPEPS
            The resulting optimized PEPS
    """

    # Set up initial run parameters
    if start_time == None:
        start_time = time.time()

    if write_to_disc:
        tmpdir = create_rand_tmpdir()
    else:
        tmpdir = None

    save_dynamical_parameters(chi=chi, **sweep_args)

    # Set initial energy
    E0 = np.inf

    # Figure out parity of initial peps
    peps_parity = peps_total_parity(peps, supplied_norm=False)

    # Do many DMRG sweeps until convergence
    benvs = None
    converged = False
    for itercnt in range(maxiter):

        # Do right sweep optimization
        E, peps, benvs = dmrg_sweep_right(H, peps, 
                                          chi=chi, 
                                          benvs=benvs, 
                                          peps_parity=peps_parity, 
                                          tmpdir=tmpdir,
                                          start_time=start_time,
                                          **sweep_args)

        # Do left sweep optimization
        E, peps, benvs = dmrg_sweep_left(H, peps, 
                                         chi=chi, 
                                         benvs=benvs, 
                                         peps_parity=peps_parity,
                                         tmpdir=tmpdir,
                                         start_time=start_time,
                                         **sweep_args)

        # Make things numerically better conditioned
        if balance_bonds:
            peps.balance_bonds_()
            benvs = None
        if equalize_norms:
            peps.equalize_norms_()
            benvs = None

        # Print Results
        print(f'Iteration {itercnt} Energy {E} ({time.time()-start_time} s)')

        # Check for convergence
        if abs((E-E0)/E0) < conv_tol:
            converged = True
            break
        else:
            E0 = E

    if converged:
        print('Calculation converged after {} iteractions, E = {}'.format(itercnt, E))
    else:
        print('Calculation not converged after {} sweeps, E = {}'.format(itercnt, E))

    # Return optimized results
    return E, peps
