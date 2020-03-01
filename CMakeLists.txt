cmake_minimum_required(VERSION 3.17)
set(target opencv-pyramid)
project(${target})
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
file(GLOB source ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.h)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${target} ${source})
target_precompile_headers(${target} PUBLIC pch.h)
target_link_libraries(${target} ${OpenCV_LIBS})
if(${CMAKE_CL_64} AND "${CMAKE_GENERATOR}" STREQUAL "Visual Studio 16 2019")
    set(opencv_dll ${OpenCV_INSTALL_PATH}/x64/vc15/bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH})
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<$<CONFIG:Debug>:${opencv_dll}d.dll>$<$<CONFIG:Release>:${opencv_dll}.dll>
            ${CMAKE_CURRENT_BINARY_DIR}/$<$<CONFIG:Debug>:Debug>$<$<CONFIG:Release>:Release>)
endif()
