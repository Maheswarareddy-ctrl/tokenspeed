# TokenSpeed-Kernel-XPU

TokenSpeed-Kernel-XPU contains the Intel XPU-specific operators used by
TokenSpeed. The operators wrap the SYCL kernels from
[sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu). Keeping them
in a standalone package mirrors the AMD and NPU package layouts and keeps the
TokenSpeed runtime vendor-neutral.

TokenSpeed applications should continue to import operators from
`tokenspeed-kernel`; it owns registration and dispatch to this package.

For development from this repository, install sgl-kernel-xpu for your GPU
target first, then:

```bash
pip install -e tokenspeed-kernel-xpu
```

Run the operator correctness suite on a visible Intel GPU with:

```bash
pytest -q tokenspeed-kernel-xpu/test
```
