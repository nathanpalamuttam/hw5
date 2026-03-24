#!/usr/bin/bash
#SBATCH -A m4341_g
#SBATCH -t 00:10:00
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q regular



srun -G 1 -n 1 ./build/test_spmm matrices/stomach.mtx 64
srun -G 1 -n 1 ./build/test_spmm matrices/stomach.mtx 512

srun -G 1 -n 1 ./build/test_spmm matrices/delaunay_n22.mtx 64
srun -G 1 -n 1 ./build/test_spmm matrices/delaunay_n22.mtx 512
