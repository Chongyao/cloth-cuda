cmake_minimum_required(VERSION 3.18)

# ---- CUDA detection (optional) ----
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    project(cuda_ms LANGUAGES C CXX CUDA)
    set(HAVE_CUDA TRUE)
else()
    project(cuda_ms LANGUAGES C CXX)
    set(HAVE_CUDA FALSE)
endif()

# ---- Standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(HAVE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Dependencies ----
find_package(Eigen3 REQUIRED NO_MODULE)
if(HAVE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ---- Subdirectories ----
add_subdirectory(src)
