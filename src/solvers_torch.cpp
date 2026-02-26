// Only compile this file when PyTorch is available
#if defined(TORCH_BUILD)

#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void birkhoff_proj_n4(
    const float* d_R, float* d_T, float tol, int N, cudaStream_t stream = cudaStreamPerThread
);

void birkhoff_proj_n4_backward(
    const float* d_G, const float* d_T, float* d_D, int N, cudaStream_t stream = cudaStreamPerThread
);

void sinkhorn_knopp_n4(
    const float* d_M, float* d_T, int N, int num_iters,
    cudaStream_t stream = cudaStreamPerThread
);

void sinkhorn_knopp_n4_backward(
    const float* d_G, const float* d_M, float* d_D, int N, int num_iters,
    cudaStream_t stream = cudaStreamPerThread
);

namespace py = pybind11;

py::dict birkhoff_proj_n4(
    torch::Tensor R,
    float tol,
    const py::kwargs& kwargs
)
{
    // Check input dimensions
    if (R.dim() != 3)
    {
        throw std::runtime_error("R must be a tensor of size B x 4 x 4");
    }

    const int batch_size = R.size(0);
    if (R.size(1) != 4 || R.size(2) != 4)
    {
        throw std::runtime_error("R must be a tensor of size B x 4 x 4");
    }

    // Ensure that inputs are on GPU with a float type
    // Zero-copy if already satisfying the conditions
    auto src_options = R.options();
    auto src_device = R.device();
    // If R is already on GPU, preserve its device ID
    // Otherwise use the default device
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    if (src_device.is_cuda())
    {
        options = options.device(torch::kCUDA, src_device.index());
    }
    R = R.to(options);

    // Set active device
    const at::cuda::CUDAGuard device_guard(options.device());

    // Ensure contiguous and get data pointers
    R = R.contiguous();
    // Pointing to device memory
    const float* d_R_ptr = R.data_ptr<float>();

    // Create output tensors that are float and on device
    torch::Tensor T = torch::empty({batch_size, 4, 4}, options);
    // Pointing to device memory
    float* d_T_ptr = T.data_ptr<float>();

    // Call CUDA function
    birkhoff_proj_n4(
        d_R_ptr, d_T_ptr, tol, batch_size, cudaStreamPerThread
    );

    // Convert T when necessary using the input tensor options
    T = T.to(src_options);

    // Create result dictionary
    py::dict result;
    result["T"] = T;

    return result;
}

py::dict birkhoff_proj_n4_backward(
    torch::Tensor G,
    torch::Tensor T,
    const py::kwargs& kwargs
)
{
    // Check input dimensions
    if (G.dim() != 3)
    {
        throw std::runtime_error("G must be a tensor of size B x 4 x 4");
    }
    if (T.dim() != 3)
    {
        throw std::runtime_error("T must be a tensor of size B x 4 x 4");
    }

    const int batch_size = G.size(0);
    if (G.size(1) != 4 || G.size(2) != 4)
    {
        throw std::runtime_error("G must be a tensor of size B x 4 x 4");
    }
    if (T.size(0) != batch_size || T.size(1) != 4 || T.size(2) != 4)
    {
        throw std::runtime_error("T must be a tensor of size B x 4 x 4");
    }

    // Ensure that inputs are on GPU with a float type
    // Zero-copy if already satisfying the conditions
    auto src_options = G.options();
    auto src_device = G.device();
    // If G is already on GPU, preserve its device ID
    // Otherwise use the default device
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    if (src_device.is_cuda())
    {
        options = options.device(torch::kCUDA, src_device.index());
    }
    G = G.to(options);
    T = T.to(options);

    // Set active device
    const at::cuda::CUDAGuard device_guard(options.device());

    // Ensure contiguous and get data pointers
    G = G.contiguous();
    T = T.contiguous();
    // Pointing to device memory
    const float* d_G_ptr = G.data_ptr<float>();
    const float* d_T_ptr = T.data_ptr<float>();

    // Create output tensors that are float and on device
    torch::Tensor D = torch::empty({batch_size, 4, 4}, options);
    // Pointing to device memory
    float* d_D_ptr = D.data_ptr<float>();

    // Call CUDA function
    birkhoff_proj_n4_backward(
        d_G_ptr, d_T_ptr, d_D_ptr, batch_size, cudaStreamPerThread
    );

    // Convert D when necessary using the input tensor options
    D = D.to(src_options);

    // Create result dictionary
    py::dict result;
    result["D"] = D;

    return result;
}

py::dict sinkhorn_knopp_n4(
    torch::Tensor M,
    int max_iter,
    const py::kwargs& kwargs
)
{
    // Check input dimensions
    if (M.dim() != 3)
    {
        throw std::runtime_error("M must be a tensor of size B x 4 x 4");
    }

    const int batch_size = M.size(0);
    if (M.size(1) != 4 || M.size(2) != 4)
    {
        throw std::runtime_error("M must be a tensor of size B x 4 x 4");
    }

    // Ensure that inputs are on GPU with a float type
    // Zero-copy if already satisfying the conditions
    auto src_options = M.options();
    auto src_device = M.device();
    // If M is already on GPU, preserve its device ID
    // Otherwise use the default device
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    if (src_device.is_cuda())
    {
        options = options.device(torch::kCUDA, src_device.index());
    }
    M = M.to(options);

    // Set active device
    const at::cuda::CUDAGuard device_guard(options.device());

    // Ensure contiguous and get data pointers
    M = M.contiguous();
    // Pointing to device memory
    const float* d_M_ptr = M.data_ptr<float>();

    // Create output tensors that are float and on device
    torch::Tensor T = torch::empty({batch_size, 4, 4}, options);
    // Pointing to device memory
    float* d_T_ptr = T.data_ptr<float>();

    // Call CUDA function
    sinkhorn_knopp_n4(
        d_M_ptr, d_T_ptr, batch_size, max_iter, cudaStreamPerThread
    );

    // Convert T when necessary using the input tensor options
    T = T.to(src_options);

    // Create result dictionary
    py::dict result;
    result["T"] = T;

    return result;
}

py::dict sinkhorn_knopp_n4_backward(
    torch::Tensor G,
    torch::Tensor M,
    int max_iter,
    const py::kwargs& kwargs
)
{
    // Check input dimensions
    if (G.dim() != 3)
    {
        throw std::runtime_error("G must be a tensor of size B x 4 x 4");
    }
    if (M.dim() != 3)
    {
        throw std::runtime_error("M must be a tensor of size B x 4 x 4");
    }

    const int batch_size = G.size(0);
    if (G.size(1) != 4 || G.size(2) != 4)
    {
        throw std::runtime_error("G must be a tensor of size B x 4 x 4");
    }
    if (M.size(0) != batch_size || M.size(1) != 4 || M.size(2) != 4)
    {
        throw std::runtime_error("M must be a tensor of size B x 4 x 4");
    }

    // Ensure that inputs are on GPU with a float type
    // Zero-copy if already satisfying the conditions
    auto src_options = G.options();
    auto src_device = G.device();
    // If G is already on GPU, preserve its device ID
    // Otherwise use the default device
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    if (src_device.is_cuda())
    {
        options = options.device(torch::kCUDA, src_device.index());
    }
    G = G.to(options);
    M = M.to(options);

    // Set active device
    const at::cuda::CUDAGuard device_guard(options.device());

    // Ensure contiguous and get data pointers
    G = G.contiguous();
    M = M.contiguous();
    // Pointing to device memory
    const float* d_G_ptr = G.data_ptr<float>();
    const float* d_M_ptr = M.data_ptr<float>();

    // Create output tensors that are float and on device
    torch::Tensor D = torch::empty({batch_size, 4, 4}, options);
    // Pointing to device memory
    float* d_D_ptr = D.data_ptr<float>();

    // Call CUDA function
    sinkhorn_knopp_n4_backward(
        d_G_ptr, d_M_ptr, d_D_ptr, batch_size, max_iter, cudaStreamPerThread
    );

    // Convert D when necessary using the input tensor options
    D = D.to(src_options);

    // Create result dictionary
    py::dict result;
    result["D"] = D;

    return result;
}

#endif  // #if defined(TORCH_BUILD)
