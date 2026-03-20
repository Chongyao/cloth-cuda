# src/CMakeLists.txt

set(SOURCES
    main.cpp
    mesh.cpp
    utils/obj_io.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

if(HAVE_CUDA)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        CUDA::cudart
        CUDA::cusparse
    )
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
    target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_MS_HAVE_CUDA)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)
