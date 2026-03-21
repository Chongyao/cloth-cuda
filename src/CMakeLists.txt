# src/CMakeLists.txt

# ---- Shared static library ----
add_library(cuda_ms_core STATIC
    mesh.cpp mesh_generator.cpp constraints.cpp utils/obj_io.cpp)
target_include_directories(cuda_ms_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cuda_ms_core PUBLIC Eigen3::Eigen)
target_compile_options(cuda_ms_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

if(HAVE_CUDA)
    target_link_libraries(cuda_ms_core PUBLIC
        CUDA::cudart
        CUDA::cusparse
    )
    set_target_properties(cuda_ms_core PROPERTIES CUDA_ARCHITECTURES native)
    target_compile_definitions(cuda_ms_core PUBLIC CUDA_MS_HAVE_CUDA)
    target_compile_options(cuda_ms_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

# ---- Simulation main ----
add_executable(cuda_ms main.cpp)
target_link_libraries(cuda_ms PRIVATE cuda_ms_core)

# ---- Tools ----
add_executable(gen_cloth tools/gen_cloth.cpp)
target_link_libraries(gen_cloth PRIVATE cuda_ms_core)

# ---- Tests ----
foreach(t test_mesh test_topology test_constraints)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE cuda_ms_core)
endforeach()
