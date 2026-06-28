# ⚡ flash-attn-prebuilt-wheels

> **Pre-compiled [Flash Attention](https://github.com/Dao-AILab/flash-attention) wheels built from source on Google Colab (NVIDIA L4 GPU) — skip the 45-minute build and install in seconds.**

[
[
[
[
[

***

## 🤔 Why This Exists

Installing `flash-attn` from PyPI requires compiling hundreds of CUDA kernels from source — a process that takes **40–60 minutes** and frequently fails due to mismatched CUDA/PyTorch/Python version combinations. Pre-compiled wheels aren't always available for every environment, especially cutting-edge configurations.

This repository provides:

- ✅ A ready-to-use **Colab notebook** to build your own wheel for any GPU/CUDA/PyTorch combination
- ✅ **Pre-built `.whl` files** hosted as GitHub Releases for direct `pip install`
- ✅ Full build logs and verification steps so you know exactly what was compiled

***

## 📦 Available Wheels

| flash-attn | Python | PyTorch | CUDA | GPU Arch | Download |
|---|---|---|---|---|---|
| 2.8.3.post1 | 3.12 | 2.11.0+cu128 | 12.8 | SM80 / SM90 / SM100 / SM120 | [v2.8.3.post1](https://github.com/npankitsubedi/flash-attn-prebuilt-wheels/releases/tag/v2.8.3.post1) |

> 💡 **Not your configuration?** See [Build Your Own Wheel](#-build-your-own-wheel) to compile for your specific environment in ~45 minutes using a free Colab L4 GPU.

***

## 🚀 Quick Install

### Option 1 — Direct pip install from Release

```bash
pip install https://github.com/npankitsubedi/flash-attn-prebuilt-wheels/releases/download/v2.8.3.post1/flashattn-2.8.3.post1-cp312-cp312-linux_x86_64.whl --no-deps
```

> ⚠️ Use `--no-deps` to avoid pip trying to recompile dependencies from source.

### Option 2 — Download and install locally

```bash
wget https://github.com/npankitsubedi/flash-attn-prebuilt-wheels/releases/download/v2.8.3.post1/flashattn-2.8.3.post1-cp312-cp312-linux_x86_64.whl

pip install flashattn-2.8.3.post1-cp312-cp312-linux_x86_64.whl --no-deps
```

### Option 3 — Install inside Google Colab

```python
!pip install https://github.com/npankitsubedi/flash-attn-prebuilt-wheels/releases/download/v2.8.3.post1/flashattn-2.8.3.post1-cp312-cp312-linux_x86_64.whl --no-deps
```

***

## ✅ Verify Installation

After installing, run the following to confirm Flash Attention is working:

```python
import torch
import flash_attn

print(f"Flash Attention version : {flash_attn.__version__}")
print(f"PyTorch version         : {torch.__version__}")
print(f"CUDA available          : {torch.cuda.is_available()}")
print(f"GPU                     : {torch.cuda.get_device_name(0)}")

# Quick functional test
from flash_attn import flash_attn_qkvpacked_func
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

batch, seqlen, nheads, d = 2, 512, 8, 64
qkv = torch.randn(batch, seqlen, 3, nheads, d, dtype=torch.float16, device="cuda")
out = flash_attn_qkvpacked_func(qkv, dropout_p=0.0)
print(f"Output shape            : {out.shape}")  # Expected: (2, 512, 8, 64)
print("✅ Flash Attention is working correctly!")
```

***

## 🔧 Compatibility Requirements

Your environment **must match** the following to use the pre-built wheel without recompilation:

| Requirement | Value |
|---|---|
| Operating System | Linux (x86_64) |
| Python | 3.12.x (`cp312`) |
| PyTorch | 2.11.0 compiled with CUDA 12.8 |
| CUDA Toolkit | 12.8 |
| GPU Compute Capability | SM 8.0, SM 8.9, SM 9.0, SM 10.0, or SM 12.0 |
| CXX11 ABI | TRUE |

### Common compatible GPUs

| GPU | Compute Capability |
|---|---|
| NVIDIA A100 | SM 8.0 ✅ |
| NVIDIA A10G | SM 8.6 ✅ |
| NVIDIA L4 | SM 8.9 ✅ |
| NVIDIA L40S | SM 8.9 ✅ |
| NVIDIA H100 | SM 9.0 ✅ |
| NVIDIA H200 | SM 9.0 ✅ |
| NVIDIA B100 / B200 | SM 10.0 / SM 12.0 ✅ |
| NVIDIA RTX 4090 | SM 8.9 ✅ |
| NVIDIA RTX 3090 | SM 8.6 ✅ |

> ❌ **Not compatible:** RTX 2080 Ti (SM 7.5), V100 (SM 7.0), or older GPUs below SM 8.0.

***

## 🛠️ Build Your Own Wheel

Use the included Colab notebook to compile flash-attn for your specific CUDA / PyTorch / Python combination.

### Step 1 — Open the notebook

[

> **Recommended runtime:** `T4 GPU` (free tier) or `L4 GPU` (Colab Pro). An L4 roughly halves the build time compared to T4.

### Step 2 — Set your environment variables

At the top of the notebook, configure:

```python
# Target flash-attn version
FLASH_ATTN_VERSION = "2.8.3.post1"   # Change to your desired version

# GPU architecture (find yours at https://developer.nvidia.com/cuda-gpus)
# Examples: "8.0" for A100, "8.9" for L4/RTX 4090, "7.5" for RTX 2080 Ti
TORCH_CUDA_ARCH_LIST = "8.0;8.9"

# Parallel jobs — use 1 to avoid OOM on T4, 2 on L4 with 24GB+ free RAM
MAX_JOBS = "1"
```

### Step 3 — Run all cells

The notebook will:

1. Mount Google Drive for output storage
2. Print system info (Python, PyTorch, CUDA, GPU)
3. Install build dependencies (`ninja`, `packaging`)
4. Compile flash-attn from source using `pip wheel`
5. Save the `.whl` file to your Google Drive at `MyDrive/flash_attn_wheels/`

### Step 4 — Retrieve and share

The output wheel will be named using the pattern:

```
flashattn-{VERSION}-cp{PYVER}-cp{PYVER}-linux_x86_64.whl
```

For example:
```
flashattn-2.8.3.post1-cp312-cp312-linux_x86_64.whl
```

Upload it as a [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) asset for permanent hosting and direct pip-installable URLs.

***

## 📁 Repository Structure

```
flash-attn-prebuilt-wheels/
├── README.md                          ← You are here
├── notebooks/
│   └── build_flash_wheel.ipynb        ← Colab build pipeline
├── wheels/
│   └── .gitkeep                       ← Wheels are hosted as GitHub Releases (not committed)
├── CHANGELOG.md                       ← Version history and build notes
└── LICENSE
```

> **Why aren't wheels committed to the repo?** A single `.whl` for flash-attn is ~200–500 MB. GitHub has a 100 MB file size limit and a 1 GB repo size recommendation. All wheels are hosted as **Release Assets**, which support files up to 2 GB each.

***

## ⚙️ Build Environment Details

The wheels in this repository were built using the following exact environment on Google Colab:

```
OS          : Linux x86_64 (Ubuntu 22.04)
GPU         : NVIDIA L4 (24 GB VRAM)
Compute Cap : 8.9
Python      : 3.12.13
PyTorch     : 2.11.0+cu128
CUDA        : 12.8
cuDNN       : Bundled with PyTorch
CXX11 ABI   : TRUE
ninja       : Latest
packaging   : Latest
TORCH_CUDA_ARCH_LIST : 8.0;8.9  (SM80 + SM89)
MAX_JOBS    : 1  (safe for 50 GB RAM Colab instance)
NVCC_THREADS: 1
```

Build command used:

```bash
TORCH_CUDA_ARCH_LIST="8.0;8.9" \
MAX_JOBS=1 \
NVCC_THREADS=1 \
pip wheel flash-attn \
  --no-build-isolation \
  --no-deps \
  -w /content/built_wheels \
  -v
```

***

## ❓ Troubleshooting

### `ImportError: /usr/local/lib/.../flashattn2cuda.so: cannot open shared object`
Your CUDA runtime version doesn't match the one used during compilation. Verify with:
```bash
python -c "import torch; print(torch.version.cuda)"
# Must output: 12.8
```

### `RuntimeError: CUDA error: no kernel image is available for execution`
Your GPU's compute capability isn't included in this wheel's compiled architectures. Check:
```python
import torch
print(torch.cuda.get_device_capability())
# Must be (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), or (12, 0)
```
If your GPU is SM 7.5 or below, you need to build a custom wheel using the notebook.

### `AssertionError: Flash attention requires sm80 or higher`
Flash Attention 2.x officially requires NVIDIA Ampere (SM 8.0) or newer. Turing (SM 7.5) and Volta (SM 7.0) are not supported.

### Colab OOM during build
Reduce parallel jobs. In the notebook, set:
```python
MAX_JOBS = "1"
NVCC_THREADS = "1"
```
If OOM persists, try restarting the runtime to free memory before building.

***

## 📋 CHANGELOG

### v2.8.3.post1 — June 2026
- **flash-attn:** 2.8.3.post1
- **PyTorch:** 2.11.0+cu128
- **CUDA:** 12.8
- **Python:** 3.12
- **GPU:** NVIDIA L4 (SM 8.9)
- **Archs compiled:** SM80, SM89, SM90, SM100, SM120
- **Build time:** ~45 minutes on L4

***

## 🤝 Contributing

Contributions are welcome! If you've built a wheel for a different configuration (e.g., Python 3.11, CUDA 12.4, PyTorch 2.6), feel free to:

1. Fork this repository
2. Add your wheel as a new GitHub Release
3. Update the [Available Wheels](#-available-wheels) table in `README.md`
4. Open a Pull Request

Please include the full build environment details in the release notes.

***

## 📜 License

This repository is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The Flash Attention library itself is licensed under the [BSD 3-Clause License](https://github.com/Dao-AILab/flash-attention/blob/main/LICENSE) by Tri Dao and the Dao-AILab team. This repository only provides a convenience build pipeline and pre-compiled artifacts; all intellectual property of flash-attn remains with the original authors.

***

## 🙏 Acknowledgements

- [Tri Dao](https://tridao.me/) and the [Dao-AILab team](https://github.com/Dao-AILab) for the Flash Attention algorithm and implementation
- Google Colab for providing free GPU compute for open-source builds
- The broader LLM / deep learning community for surfacing this installation pain point

***

<p align="center">
  Made with ❤️ by <a href="https://github.com/npankitsubedi">npankitsubedi</a>
</p>
