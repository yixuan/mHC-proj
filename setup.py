import os
import sys
from glob import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.1.0"

# Use PyTorch's CUDAExtension
ext_modules = [
    CUDAExtension(
        name="mhc_proj._internal",
        sources=sorted(glob("src/*.cpp") + glob("src/*.cu")),
        include_dirs=[
            os.path.join(sys.exec_prefix, "include")
        ],
        library_dirs=[
            os.path.join(sys.exec_prefix, "lib64"),
            os.path.join(sys.exec_prefix, "lib")
        ],
        libraries=["cudart", "cuda"],
        define_macros=[("VERSION_INFO", __version__), ("TORCH_BUILD", None)],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc":
                [
                    "-O3",
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"
                ]
        }
    )
]

setup(
    name="mhc_proj",
    version=__version__,
    author="Yixuan Qiu",
    author_email="yixuanq@gmail.com",
    url="https://github.com/yixuan/mHC-proj",
    description="Accelerated Birkhoff Projection for Manifold-Constrained Hyper-Connections",
    packages=["mhc_proj"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.12"
)
