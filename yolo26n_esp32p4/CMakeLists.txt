# Auto-generated YOLOv26n Inference Project
cmake_minimum_required(VERSION 3.16)

add_compile_options(-fdiagnostics-color=always)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

set(DEPENDENCIES_LOCK_FILE "${CMAKE_SOURCE_DIR}/dependencies.lock.${IDF_TARGET}")
if(EXISTS ${DEPENDENCIES_LOCK_FILE})
    idf_build_set_property(DEPENDENCIES_LOCK ${DEPENDENCIES_LOCK_FILE})
endif()

project(yolo26n_inference)
