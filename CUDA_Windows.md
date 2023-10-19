# CUDA on Windows 11

# Check Version

Check whether is already installed. In this example, CUDA is not already insalled.

``` cmd
>where nvcc
INFO: Could not find files for the given pattern(s).

>nvcc --version
'nvcc' is not recognized as an internal or external command, operable program or batch file.
```

# Install Visual Studio

If a compatible version of Visual Studio is not found, the CUDA Toolkit installer will prompt to install it.

Install Visual Studio Community 2022.

https://visualstudio.microsoft.com/

I selected Multi-platform .NET and C++ packages.





# Install CUDA Toolkit

Install CUDA Toolkit 12.2 from 

* https://developer.nvidia.com/cuda-toolkit
* https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_537.13_windows.exe

Install to the default location.


