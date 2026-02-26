# mHC-proj

**mHC-proj** is a CUDA library that accelerates the Birkhoff projection operator in [manifold-constrained hyper-connections (mHC)](https://arxiv.org/pdf/2512.24880). The current implemention supports the expansion rate of $n=4$ in mHC, providing highly optimized CUDA kernels that output $4\times 4$ doubly stochastic matrices from unconstrained ones. Both forward and backward passes are supported.

## Method

This library uses a different algorithm from the Sinkhorn-Knopp method that is suggested by the [mHC paper](https://arxiv.org/pdf/2512.24880). It has the following highlights:

1. **Forward pass via Newton's method**: We reformulate the dual of the Birkhoff projection problem as an unconstrained convex optimization in $\mathbb{R}^{3}$, and derive closed-form expressions for the gradient and Hessian. This enables the use of Newton's method, which converges quadratically and typically requires far fewer iterations than Sinkhorn-Knopp.

2. **Backward pass via implicit differentiation**: Instead of backpropagating through the iterative solver, we derive an analytical expression for the derivative of the projection using the implicit function theorem. This allows us to compute gradients exactly and efficiently, without storing intermediate iterates.

3. **GPU-efficient implementation**: We design a warp-level CUDA kernel that processes two $4\times4$ matrices simultaneously using only register-level primitives. The implementation avoids shared memory and global memory I/O, achieving high throughput with minimal overhead.

## Installation

After installing CUDA-based PyTorch and setting up the CUDA development environment, enter the project directory and run:

```bash
pip3 install --no-build-isolation .
```

## Benchmark

We benchmark the following four open-source implementations on an Nvidia RTX 6000 Ada GPU:

1. **Vanilla**: A simple implementation of the Sinkhorn–Knopp algorithm using pure PyTorch code.

2. **Triton-Sinkhorn**: A CUDA-fused implementation of the Sinkhorn–Knopp algorithm backed by OpenAI Trition: https://github.com/LottoLottoLotto/triton-sinkhorn.

3. **mHC.cu**: A CUDA implementation of mHC: https://github.com/AndreSlavescu/mHC.cu.

4. **mHC-proj**: This library.

### Accuracy

We randomly generate $N=10000$ matrices of size $4\times 4$, forming an $N\times 4\times 4$ tensor $R$ as the input of the four implementations. Each of them outputs a tensor $T$ consisting of $N$ matrices of size $4\times 4$. If the projection is accurate, then each $T_i$ is doubly stochastic, so we measure the error as

$$
\mathrm{Err}(T_i)=\Vert T_i\mathbf{1}_4-\mathbf{1}_4 \Vert_1 + \Vert T_i^\top\mathbf{1}_4-\mathbf{1}_4 \Vert_1.
$$

The mean, median, and maximum of the $N$ error values are summarized below:

| Input Distribution | Method | Mean Error | Median Error | Max Error |
|--------------------|--------|------------|--------------|-----------|
| $N(0,1)$ | Vanilla | $8.34\times 10^{-6}$ | $7.79\times 10^{-6}$ | $1.38\times 10^{-3}$ |
| | Triton-Sinkhorn  | $9.38\times 10^{-7}$ | $3.28\times 10^{-7}$ | $1.38\times 10^{-3}$ |
| | mHC.cu           | $8.67\times 10^{-7}$ | $2.61\times 10^{-7}$ | $1.38\times 10^{-3}$ |
| | **mHC-proj**     | $6.54\times 10^{-7}$ | $5.27\times 10^{-7}$ | $2.50\times 10^{-6}$ |

| Input Distribution | Method | Mean Error | Median Error | Max Error |
|--------------------|--------|------------|--------------|-----------|
| $N(0,10^2)$ | Vanilla | $7.25\times 10^{-2}$ | $6.51\times 10^{-2}$ | $0.83$  |
| | Triton-Sinkhorn     | $7.25\times 10^{-2}$ | $6.51\times 10^{-2}$ | $0.83$  |
| | mHC.cu              | $8.79\times 10^{-2}$ | $6.55\times 10^{-2}$ | $4.00$  |
| | **mHC-proj**        | $1.59\times 10^{-3}$ | $8.98\times 10^{-7}$ | $0.091$ |


### Run time

We fix the input distribution to be $N(0,10^2)$, and measure the run time of different implementations for various batch sizes $N$. The time is normalized such that in each configuration **mHC-proj** has one unit of run time.

Forward pass only:

| Batch size | Vanilla | Triton-Sinkhorn | mHC.cu | mHC-proj |
|------------|---------|-----------------|--------|----------|
| 0.5K       | 55.71   | 10.06           | 3.158  | 1.000    |
| 2K         | 55.53   | 19.12           | 3.161  | 1.000    |
| 8K         | 55.98   | 86.68           | 4.142  | 1.000    |
| 32K        | 24.27   | 198.0           | 2.355  | 1.000    |
| 128K       | 9.823   | 271.1           | 1.338  | 1.000    |

Forward and backward passes:

| Batch size | Vanilla | Triton-Sinkhorn | mHC.cu | mHC-proj |
|------------|---------|-----------------|--------|----------|
| 0.5K       | 136.1   | 9.825           | 3.523  | 1.000    |
| 2K         | 136.4   | 12.74           | 3.543  | 1.000    |
| 8K         | 135.1   | 52.87           | 4.085  | 1.000    |
| 32K        | 95.59   | 180.5           | 6.903  | 1.000    |
| 128K       | 52.59   | 370.5           | 21.89  | 1.000    |

