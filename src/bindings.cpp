#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// PyTorch interface declarations (only when PyTorch is available)
#if defined(TORCH_BUILD)
#include <torch/extension.h>

py::dict birkhoff_proj_n4(
    torch::Tensor R,
    float tol,
    const py::kwargs& kwargs
);

py::dict birkhoff_proj_n4_backward(
    torch::Tensor G,
    torch::Tensor T,
    const py::kwargs& kwargs
);

py::dict sinkhorn_knopp_n4(
    torch::Tensor M,
    int max_iter,
    const py::kwargs& kwargs
);

py::dict sinkhorn_knopp_n4_backward(
    torch::Tensor G,
    torch::Tensor M,
    int max_iter,
    const py::kwargs& kwargs
);
#endif


// PYBIND11 module definition
PYBIND11_MODULE(_internal, m)
{
    m.doc() = "Accelerated Birkhoff Projection for Manifold-Constrained Hyper-Connections";

    // PyTorch interface submodule (only when PyTorch is available)
#if defined(TORCH_BUILD)
    py::module m_torch = m.def_submodule("torch", "PyTorch interface");
    m_torch.def("birkhoff_proj_n4", &birkhoff_proj_n4,
        py::arg("R"), py::arg("tol") = 1e-6,
        "KL projection of 4x4 matrix to Birkhoff polytope");
    m_torch.def("birkhoff_proj_n4_backward", &birkhoff_proj_n4_backward,
        py::arg("G"), py::arg("T"),
        "KL projection of 4x4 matrix to Birkhoff polytope, backward pass");
    m_torch.def("sinkhorn_knopp_n4", &sinkhorn_knopp_n4,
        py::arg("M"), py::arg("max_iter") = 20,
        "Sinkhorn-Knopp algorithm for 4x4 matrix");
    m_torch.def("sinkhorn_knopp_n4_backward", &sinkhorn_knopp_n4_backward,
        py::arg("G"), py::arg("M"), py::arg("max_iter") = 20,
        "Sinkhorn-Knopp algorithm for 4x4 matrix, backward pass");
#endif
}
