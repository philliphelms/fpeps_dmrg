# fpeps_dmrg
This is an implementation of a DMRG-style optimization for a 2D fermionic PEPS.

## Getting Started
To get started, you will need to install an implementation of quimb that supports fermions, 
with a few minor modifications that make it possible to run calculations with this code.
This is available at:

https://github.com/philliphelms/quimb/tree/refactor

The quimb library has a number of dependencies that may need to be installed. 

Additionally, you will need pyblock3, which is the 
library that handles the symmetric tensor operations 
that are required for use with quimb, 
with the library available here:

https://github.com/block-hczhai/pyblock3-preview

Additionally, you will need a working build of mpi4py. 

## Running an initial calculation
The file fpeps_dmrg/run.py will run a Hubbard Model calculation. As an example, you can 
run a Hubbard Model calculation at half filling with U=8, with a PEPS bond dimension of 
D=3 
```
> Nx=4
> Ny=4
> Ne=16
> D=3
> U=8
> python run.py $Nx $Ny $Ne $D $U
```
This will start a ground state calculation and save the optimized peps at each time step in the 
directory `./fpeps_dmrg/saved_states/`
