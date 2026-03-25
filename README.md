# Homework 5: Sparse-Dense Matrix Multiply (SpMM) with CUDA #

## Introduction ## 

Sparse-Dense Matrix Multiply (SpMM) is a sparse linear algebra primitive that multiplies a sparse matrix $A \in \mathbb{R}^{m \times n}$ with a dense matrix $X \in \mathbb{R}^{n \times k}$, producing a dense output matrix $Y \in \mathbb{R}^{m \times k}$. In other words, we compute 
$$
    Y = AX
$$
SpMM is one of the most important sparse linear algebra primitives, with applications in graph neural network training, numerical linear algebra, and clustering. 
Oftentimes, SpMM is the bottleneck in these applications. 
Therefore, access to a high performance parallel implementation of SpMM is of great importance. 

In this assignment, we will be implementing a parallel SpMM kernel for NVIDIA GPUs. 
Your kernel will be benchmarked on three large sparse matrices with different sparsity structures.
For each matrix, the attained computational throughput (GFLOPS/s) of your SpMM kernel will be measured.
The goal is to implement a kernel that achieves a high throughput on all three sparse matrices.


## Build Instructions ##
To build the driver program, run the following

`mkdir build && cd build`

`cmake ..`

`make test_spmm`

This will generate a binary called `test_spmm`, which can be run with a command like this

`./test_spmm /path/to/matrix k`

where `/path/to/matrix` is a relative path to a matrix stored in matrix market (`.mtx`) format, and `k` is the number of colums in the dense matrix.

The `test_spmm` executable does two things.
First, it will run a correctness check that compares the output of your SpMM kernel to the output of the SpMM kernel found in the [cuSPARSE library](https://docs.nvidia.com/cuda/cusparse/contents.html). 
Second, it runs a simple benchmark that compares the throughput in terms of GFLOPS/s of your SpMM kernel to the throughput of cuSPARSE's SpMM kernel.


## Running Benchmarks ## 

Your kernel will be evaluated on three large test matrices from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/).
Each test matrix has at least 10s of millions of nonzeros, and each one has a different sparsity structure. 
The three test matrices are:

- delaunay_n24: https://sparse.tamu.edu/DIMACS10/delaunay_n24  
- nlpkkt120: https://sparse.tamu.edu/Schenk/nlpkkt120
- Cube_Coup_dt0: https://sparse.tamu.edu/Janna/Cube_Coup_dt0

For each matrix, we'll run your kernel with `k=64, 256`, i.e. different numbers of columns in the dense matrix. 

To fetch and unzip the matrices from SuiteSparse, run `sh get_matrices.sh`, which will place the matrices in the `matrices` directory.
Additionally, the `matrices` directory contains a small matrix called `n16.mtx`, which can be used for debugging purposes.

If you would like to benchmark your implementation on other test matrices other than the ones obtained by `get_matrices.sh`, feel free to fetch other matrices from SuiteSparse.
To fetch a specific matrix from SuiteSparse, left-click the `Matrix Market` button next to the matrix, copy the link referenced by that button, then run `wget <link>` to fetch a tarball containing the matrix. Then, you can `tar -xzf </path/to/tarball>` to decompress the matrix.

## Grading ## 

Your grade out of 100 points will be based on the following things:

- 10/100: Checkpoint
- 70/100: Performance
- 10/100: Leaderboard Submission
- 20/100: Writeup

### Checkpoint ### 

Your code must produce correct results for all three matrices to pass the checkpoint.

### Performance

Your GFLOPS/s will be computed on all three matrices for $k=64, 256$ and averaged. That will represent your final performance metric. If you get an average of at least TODO GFLOPS/s, you'll get at least a B+ on the performance part of your grade. Beyond that you'll be graded on a curve based on performance relative to the other students in the class. 

### Leaderboard

There will be a leaderboard, it is still under construction, but it will function the same as the leaderboard in HW3 and HW4. 
You'll be ranked on the leaderboard based off of your average GFLOPS/s across all three test matrices and the two test values of `k`. 

### Writeup ### 

Writeup needs a bar plot with one collection of bars per matrix showing the GFLOPS/s of your kernel on each matrix relative to the GFLOPS/s attained by cuSPARSE SpMM. 
There should be a short caption describing the plot. 

## Implementation Guide ##

### Important Files ###
The main file that you'll need to work with is `include/spmm.cuh`. This file contains the template SpMM kernel that you'll need to implement. Within this file, there are two importation functions

1. `__global__ void SpMM()` is the actual kernel that performs the SpMM operation on the GPU.

2. `void SpMM_wrapper()` is a simple wrapper function callable from the host that will call the GPU SpMM kernel. 
You can use this function to set anything up or perform any preprocessing steps that are necessary for your implementation to function.

The driver program can be found in `tests/test_spmm.cu`, although you should not edit this file at all.


### CSR Data Structure ###
Your implementation will need to function with the sparse matrix `A` stored in Compressed Sparse Row (CSR) format.
CSR is a standard compressed storage format used to efficiently store and process large sparse matrices. 

The CSR format uses three arrays `values`, `colinds`, and `rowptrs`.
The `values` array stores the values of the nonzeros in `A`.
The `colinds` array stores the column index of each nonzero in `A`. It is the same length as the `values` array.
Finally, the `rowptrs` array stores the start index of each row of `A` in `values` and `colinds`. For example, if `rowptrs[i] = 6`, then `values[6]` contains the first nonzero element in the ith row of `A`, and `colinds[6]` contains the column index of that element.
Additionally, if `rowptrs[i+1] = 11`, then the range `values[6:11]` contains all the nonzeros in the ith row of `A`.

For more information on CSR, see [Wikipedia](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)).

In this repo, `include/CSR.hpp` defines a class that can be used to store a sparse matrix using the CSR storage format. 

### Parallelization Strategy ###
In this assignment, the dense matrices are all stored in **row-major order.** 
This is different from HW1, where they were stored in column-major order. 
Since $A$ is stored in CSR and $X$ is stored in row-major order, a good way to implement your SpMM kernel is using a row-by-row approach, where the output $Y$ is computed a row at a time. 
This requires iterating through the nonzero elements in each row of $A$, then for each one, you multiply it with a single row of $X$, accumulating the result into a partial sum of a row of $Y$.
This requires contiguous accesses to the nonzeros in each row of $A$, and contiguous accesses to each row of $X$ and $Y$.
Both requirements align well with CSR and row-major order. 

The main challenge associated with parallelizing SpMM on a GPU using this row-by-row appraoch is load-balancing among the rows of the sparse matrix $A$.
In particular, each row of $A$ will typically have a different number of nonzero elements. 
The cost of multiplying a given row of $A$ with of $X$ (which will produce a complete row of $Y$) is determined by the number of nonzeros in that row of $A$.
Therefore, the computation needed to produce each row of $Y$ is different, and you must think about how to split up the task of computing each row of the output $Y$ among warps/thread blocks effectively. 

Furthermore, another challenge concerns how to effectively merge partial results of each row of $Y$.
A naive approach would be to accumulate each row of $Y$ in global memory using atomic additions. 
This will produce correct results, but there are significantly better approaches that involve the use of shared memory. 
Furthermore, one can make use of the Block-wide and Warp-wide functions in the [CUDA Unbound (CUB)](https://docs.nvidia.com/cuda/cub/index.html) library to implement effective merging of partial results of each row. 

**We'll note that there are other ways you can parallelize a CSR + row-major SpMM kernel. The row-by-row approach is only one such strategy.**

A final, rather exotic optimization you might try is reordering the rows/columns of each matrix to try to improve data locality/load balancing. 
Generally, if you can permute the matrix so it has something that looks like a block-diagonal or banded sparsity pattern, your performance might improve. 
However, you have to account for the potential overhead associated with computing and applying the reordering, which can be significant. 
It is recommended to attempt reordering only if you have exhausted other options and are otherwise happy with your kernel's performance. 


### References ### 

These are some optional references you can look at for inspiration/more information.

* [cuSPARSE SpMM API](https://docs.nvidia.com/cuda/cusparse/#cusparsespmm)

* [Yang, Carl, Aydın Buluç, and John D. Owens. "Design principles for sparse matrix multiplication on the gpu." European Conference on Parallel Processing. Cham: Springer International Publishing, 2018.](https://arxiv.org/pdf/1803.08601)

