cmake_minimum_required(VERSION 3.14)
project(CHMM)
set (CMAKE_CXX_STANDARD 20) #Sets C++ standard to C++20 

FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED ) #Need Boost
INCLUDE_DIRECTORIES(include) #Find the Headers

file(GLOB SOURCES "src/*.cpp") #Add .cpp files

add_library(CHMM SHARED ${SOURCES})

#TARGET_LINK_LIBRARIES(CHMM LINK_PUBLIC ${Boost_LIBRARIES} )