cmake_minimum_required(VERSION 3.14)
project (CHMM_Examples)
 
set (CMAKE_CXX_STANDARD 20) #Sets C++ standard to C++20
set (SOURCES main.cpp) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include) #Find hmm.h

# Find the dynamic library
find_library(LIBCHMM CHMM
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../build
)
 
add_executable(a.out ${SOURCES})

target_link_libraries(a.out #hmm.h
    PRIVATE
        ${LIBCHMM}
)
target_link_libraries(a.out #Boost
    LINK_PUBLIC
        ${Boost_LIBRARIES}
)