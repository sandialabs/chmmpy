cmake_minimum_required(VERSION 3.14)
project(CHMM)
set (CMAKE_CXX_STANDARD 20)

set(CMAKE_PREFIX_PATH = /opt/homebrew/)

FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Cbc_INCLUDE_DIR} )

add_executable(a.out main.cpp)

TARGET_LINK_LIBRARIES( a.out LINK_PUBLIC ${Boost_LIBRARIES} )