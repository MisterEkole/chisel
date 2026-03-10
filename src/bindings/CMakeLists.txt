# src/bindings/CMakeLists.txt

pybind11_add_module(_chisel_cpp bindings.cpp)

target_link_libraries(_chisel_cpp PRIVATE
    chisel_core
    chisel_geometry
    chisel_reconstruction
)

target_include_directories(_chisel_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)


set(PYTHON_DEST "${CMAKE_SOURCE_DIR}/python/chisel")

set_target_properties(_chisel_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYTHON_DEST}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PYTHON_DEST}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PYTHON_DEST}"
    
    PREFIX ""
)