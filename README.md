# Prerequisites

* Supported operating systems:
  * Linux
  * Mac OS X
  * Windows (via cross-compilation on Linux using MinGW)
* Supported GPUs:
  * AMD GPU with installed drivers
  * NVIDIA GPU with installed drivers
* Supported OpenCL SDKs:
  * AMD APP SDK: installation root exported in environment variable `AMDAPPSDKROOT `
  * NVIDIA CUDA Toolkit: installation root exported in environment variable `NVIDIA_SDK_INSTALL_PATH`

# Build instructions

The codebase contains a top-level Makefile that builds the library and the example, providing the following build targets:

* linux (default)
* macos
* win32
* win64
* clean

# Usage

* Library
* Example
