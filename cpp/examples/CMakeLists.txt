cmake_minimum_required(VERSION 3.14)
project (CHMM_Examples)
 
set (CMAKE_CXX_STANDARD 20) #Sets C++ standard to C++20
set (SOURCES simpleLearn.cpp) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include) #Find hmm.h

#Find the dyanmic library already generate
find_library(LIBCHMM CHMM
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../build
)

#Go through all files in examples and add an executable for each
file( GLOB EXAMPLES *.cpp )
foreach( exampleFile ${EXAMPLES} )
    # Cut off the file extension and directory path
    get_filename_component( exampleName ${exampleFile} NAME_WE )
    
    add_executable( ${exampleName} ${exampleFile} )
    
    target_link_libraries(${exampleName} #hmm.h
        PRIVATE
        ${LIBCHMM}
    )

endforeach( exampleFile ${EXAMPLES} )