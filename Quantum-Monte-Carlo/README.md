# QMCMC Correlations

Final project for a computational physics course on quantum mechanics. It is a
quantum markov-chain monte carlo simulation that generates sample "paths" for a
quantum particle in a simple harmonic oscillator potential well. In particular,
the Metropolis method, as described in [the original
paper](http://dx.doi.org/10.1063/1.1699114), is employed to generate the new
path from an old one. Simply, a new move is always accepted if it lowers the
total energy of the system, but, if it increases the energy, then it is
accepted with a probability depending on the change in energy.

Generating many such paths allows us to sample the state space of the system
that is near to the ground state. Hence a histogram of the sampled paths
represents the norm squared of the wavefunction, and we can estimate the energy
by taking an average of the energies of each path.

The energy levels higher than the ground state can also be estimated using the
generalized eigenvalue method from lattice field theory. Essentially this
amounts to computing a time series of correlation matricies and computing their
eigenvalues. With this method, an estimate of the ground and first two excited
state energies were estimated.

This was a pair project with Benjamin Hope. We learned the theory together but
had separate implementations. His was CPU based whereas mine used CUDA and the GPU
to generate the paths. I could not find an efficient algorithm to compute the
correlation matricies on the GPU, so those calculations do use the CPU, but are
accelerated with OpenMP. My implementation was faster overall, but the results
did not match the theory as well as Benjamin's. Refer to the presentation for
a summary of our results.

For a much more detailed and complete explanation of the theory, please refer
to the following:
* [Monte Carlo Method in Quantum Field Theory by Colin Morningstar](http://arxiv.org/pdf/hep-lat/0702020v1.pdf)
* [Paper on the Generalized Eigenvalue Method](http://arxiv.org/abs/0902.1265v2)
* [DESY Summer Program Student Project](http://www-zeuthen.desy.de/students/2012/reports/Aleksandra_Willian.pdf)

## Description of the Code

My implementation of the project is included. A CUDA enabled NVIDIA GPU is
required since most of the calculations are done on the GPU. I learned and used
[Thrust](https://code.google.com/p/thrust/), a template only library with GPU
acceleration of common STL algorithms, to perform many of the operations. This
made development and testing simpler and also makes the code much more
readable.

* MetropolisSampler – My CUDA implementation of the “markov machine” that uses
  the metropolis algorithm to generate sample distributions. The files contains
  a class definition and the necessary device functions to perform the
  computations.

* PathAnalyzer – My attempt at a class to compute the correlation matrices on
  the GPU using Thrust. After writing it, I realized that the many calls to the
  Thrust library produced an inefficient algorithm. It would be better to write
  specialized kernels that take advantage of the shared memory to speed up the
  computation. This class was not used to produce the final results.

* ThrustHistogram – A basic implementation of a histogram class using the
  Thrust library to parallelize the computations on the GPU. The data and
  computations are all done device side.

* main.cpp – The main function that runs the simulation and computes the
  correlation matrices. Produces multiple text files with the results outputted
  into the outputs folder.

* Makefile – My makefile to compile the program. There aren't any external
  dependencies other than those that are default in the CUDA SDK. Mainly the
  environment variables for paths and the compute arch and code flags may need
  to be modified to get it to compile on your system.

* makeplots.sh – Bash script to automate the generation of the plots using
  gnuplot. Genereates the wavefunction squared overlayed on the potential well
  and a gif of some sample paths that were generated during the simulation.

* my_helper.h – Little header file with some helper functions. Only the CUDA_CALL
  macro is used in this project.

* MATLAB – Contains a couple of matlab scripts used to analyze the outputs of the
  simulation and produce the graphs shown in the presentation.

## License

If you happen to find this project useful, you are welcome to use it. Simply
drop me an email to make me feel good about myself if you do.

The MIT License (MIT)

Copyright (c) 2014 Travis Johnson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
