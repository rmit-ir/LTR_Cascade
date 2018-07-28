cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(ExternalProject)

project(SIGIR17 CXX C)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release)
endif()
MESSAGE( STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )

externalproject_add(indri_proj
    SOURCE_DIR ${CMAKE_HOME_DIRECTORY}/external/WANDbl/external/indri
    CONFIGURE_COMMAND ./configure --prefix=${CMAKE_HOME_DIRECTORY}/external/local
    BUILD_IN_SOURCE 1
    BUILD_COMMAND make
    INSTALL_COMMAND make install
    )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11 -Wall -Wextra -pedantic -march=native")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++11 -Wall -Wextra -pedantic -DHAVE_EXT_ATOMICITY=1 -DP_NEEDS_GNU_CXX_NAMESPACE=1 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

add_library(indri STATIC IMPORTED)
set(INDRI_INCLUDE_DIR ${CMAKE_HOME_DIRECTORY}/external/local/include)
set(INDRI_LIB_DIR ${CMAKE_HOME_DIRECTORY}/external/local/lib)
set_target_properties(indri PROPERTIES IMPORTED_LOCATION ${INDRI_LIB_DIR}/libindri.a)

set(WANDBL_INCLUDE_DIR ${CMAKE_HOME_DIRECTORY}/external/WANDbl/include)

set(CEREAL_INCLUDE_DIR ${CMAKE_HOME_DIRECTORY}/external/cereal/include)

set(CLI11_INCLUDE_DIR ${CMAKE_HOME_DIRECTORY}/external/CLI11/include)


set(FASTPFOR_INCLUDE_DIR ${CMAKE_HOME_DIRECTORY}/external/FastPFor/headers)
add_library(FastPFor STATIC ${CMAKE_HOME_DIRECTORY}/external/FastPFor/src/bitpacking.cpp
                                ${CMAKE_HOME_DIRECTORY}/external/FastPFor/src/bitpacking.cpp
                                ${CMAKE_HOME_DIRECTORY}/external/FastPFor/src/bitpackingaligned.cpp
                                ${CMAKE_HOME_DIRECTORY}/external/FastPFor/src/bitpackingunaligned.cpp
                                ${CMAKE_HOME_DIRECTORY}/external/FastPFor/src/horizontalbitpacking.cpp
                                ${CMAKE_HOME_DIRECTORY}/external/FastPFor/src/simdunalignedbitpacking.cpp
                                ${CMAKE_HOME_DIRECTORY}/external/FastPFor/src/simdbitpacking.cpp
                                ${CMAKE_HOME_DIRECTORY}/external/FastPFor/src/varintdecode.c
                                ${CMAKE_HOME_DIRECTORY}/external/FastPFor/src/streamvbyte.c
                                ${FASTPFOR_INCLUDE_DIR}
                                )


include_directories(${INDRI_INCLUDE_DIR} ${WANDBL_INCLUDE_DIR} ${CEREAL_INCLUDE_DIR} ${CLI11_INCLUDE_DIR} ${FASTPFOR_INCLUDE_DIR} feature-gen/include)

# forward_index
add_executable(create_forward_index feature-gen/src/create_forward_index.cpp)
add_dependencies(create_forward_index indri_proj)
target_link_libraries(create_forward_index indri pthread z)

# lexicon
add_executable(create_lexicon feature-gen/src/create_lexicon.cpp)
add_dependencies(create_lexicon indri_proj)
target_link_libraries(create_lexicon indri pthread z)

# inverted_index
add_executable(create_inverted_index feature-gen/src/create_inverted_index.cpp)
add_dependencies(create_inverted_index indri_proj)
target_link_libraries(create_inverted_index indri FastPFor pthread z)

# create_bigram_inverted_index
add_executable(create_bigram_inverted_index feature-gen/src/create_bigram_inverted_index.cpp)
add_dependencies(create_bigram_inverted_index indri_proj)
target_link_libraries(create_bigram_inverted_index indri FastPFor pthread z)
target_include_directories(create_bigram_inverted_index PUBLIC ${CMAKE_HOME_DIRECTORY}/feature-gen/src)

# fgen_term
add_executable(fgen_term feature-gen/src/fgen_term.cpp)
target_link_libraries(fgen_term FastPFor m)

# pre-retrieval csv
add_executable(preret_csv feature-gen/src/preret_csv.cpp
    feature-gen/src/fgen_term_qry.c feature-gen/src/query_features.c
    feature-gen/src/fgen_bigram_qry.c
    feature-gen/src/strbuf.c)
add_dependencies(preret_csv indri_proj)
target_link_libraries(preret_csv indri pthread z)

# fgtrain
add_executable(fgtrain feature-gen/src/fgtrain.cpp
    feature-gen/src/fgen_term_qry.c feature-gen/src/query_features.c
    feature-gen/src/fgen_bigram_qry.c
    feature-gen/src/strbuf.c)
add_dependencies(fgtrain create_bigram_inverted_index indri_proj)
target_link_libraries(fgtrain indri pthread z)
