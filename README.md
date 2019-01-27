# DFT Parallel Computation
## Project Description
Implementation of 2 - Dimensional DFT (Discrete Fourier Transform) using parallel computation methods on
- C++ 11 threads
- MPI (Message Passing Interface)
- GPU using CUDA

Run times are calculated for the different input image sizes of 128x128, 256x256, 512x512, 1024x1024 and 2048x2048

For analysis, check 'Summary Report.pdf'

## Installation Instructions

Modules to load 
- module load gcc/4.9.0
- module load cmake/3.9.1
- module load openmpi
- module load cuda

CMake used to configure project

Instructions to run
- 'forward' for DFT, 'reverse' for inverse DFT
- p31 for C++11 threads, p32 for MPI, p33 for CUDA

eg:
./p31 forward Tower256.txt Output256.txt
./p32 reverse InputFile.txt OutFile.txt
