# src/CMakeLists.txt

# ---- Core CPU library ----
add_library(cuda_ms_core STATIC
    cloth_mesh.cpp
    mesh_topology.cpp
    sim_constraints.cpp
    mesh_generator.cpp
    constraints.cpp
)
target_include_directories(cuda_ms_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(cuda_ms_core PUBLIC Eigen3::Eigen)
target_compile_options(cuda_ms_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

if(HAVE_CUDA)
    target_link_libraries(cuda_ms_core PUBLIC CUDA::cudart)
    set_target_properties(cuda_ms_core PROPERTIES CUDA_ARCHITECTURES native)
    target_compile_definitions(cuda_ms_core PUBLIC CUDA_MS_HAVE_CUDA)
    target_compile_options(cuda_ms_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

# ---- FetchContent for viewer dependencies ----
include(FetchContent)

FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8 GIT_SHALLOW TRUE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(glm)

add_library(imgui_glfw STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_link_libraries(imgui_glfw PUBLIC glfw)
target_include_directories(imgui_glfw PUBLIC
    ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)

# ---- Viewer library ----
add_library(viewer STATIC
    viewer/viewer.cpp
    viewer/camera.cpp
    viewer/shader.cpp
    viewer/glad/glad.c
)
target_link_libraries(viewer PUBLIC cuda_ms_core glfw imgui_glfw glm)
target_include_directories(viewer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/viewer)
find_package(OpenGL REQUIRED)
target_link_libraries(viewer PUBLIC OpenGL::GL)

# ---- Executables ----
add_executable(cuda_ms main.cpp)
target_link_libraries(cuda_ms PRIVATE cuda_ms_core)

add_executable(gen_cloth tools/gen_cloth.cpp)
target_link_libraries(gen_cloth PRIVATE cuda_ms_core)

add_executable(view_cloth tools/view_cloth.cpp)
target_link_libraries(view_cloth PRIVATE viewer)

# ---- PD Solver (CUDA required) ----
if(HAVE_CUDA)
    add_library(pd_solver STATIC pd_solver.cu)
    target_link_libraries(pd_solver PUBLIC cuda_ms_core)
    set_target_properties(pd_solver PROPERTIES CUDA_ARCHITECTURES native)
    target_compile_options(pd_solver PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

    add_executable(sim_cloth tools/sim_cloth.cpp)
    target_link_libraries(sim_cloth PRIVATE pd_solver cuda_ms_core)
else()
    message(WARNING "CUDA not available — pd_solver and sim_cloth disabled")
endif()

# ---- Tests ----
foreach(t test_mesh test_topology test_constraints test_pd_constraints)
    add_executable(${t} tests/${t}.cpp)
    if(HAVE_CUDA)
        target_link_libraries(${t} PRIVATE cuda_ms_core pd_solver)
    else()
        target_link_libraries(${t} PRIVATE cuda_ms_core)
    endif()
endforeach()
