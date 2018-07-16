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

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11 -Wall -Wextra -pedantic")
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

include_directories(${INDRI_INCLUDE_DIR} ${WANDBL_INCLUDE_DIR} ${CEREAL_INCLUDE_DIR} feature-gen/include)

# forward_index
add_executable(forward_index feature-gen/src/forward_index.cpp)
add_dependencies(forward_index indri_proj)
target_link_libraries(forward_index indri pthread z)

# forward_index
add_executable(lexicon feature-gen/src/lexicon.cpp)
add_dependencies(lexicon indri_proj)
target_link_libraries(lexicon indri pthread z)

# dump_indri
add_executable(dump_indri feature-gen/src/dump_indri.cpp)
add_dependencies(dump_indri indri_proj)
target_link_libraries(dump_indri indri pthread z)

# dump_bigram
add_executable(dump_bigram feature-gen/src/dump_bigram.cpp
    feature-gen/src/w_scanner.cpp)
add_dependencies(dump_bigram indri_proj)
target_link_libraries(dump_bigram indri pthread z)
target_include_directories(dump_bigram PUBLIC ${CMAKE_HOME_DIRECTORY}/feature-gen/src)

# fgen_term
add_executable(fgen_term feature-gen/src/fgen_term.c)
target_link_libraries(fgen_term m)

# fgen_bigram
add_executable(fgen_bigram feature-gen/src/fgen_bigram.c)
target_link_libraries(fgen_bigram m)

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
add_dependencies(fgtrain dump_bigram indri_proj)
target_link_libraries(fgtrain indri pthread z)

# FIXME bench, unigram, query score,  bigram

# bench document feature
# add_executable(bench_docfeat feature-gen/src/bench_docfeat.cpp)
# add_dependencies(bench_docfeat indri_proj)
# target_link_libraries(bench_docfeat indri pthread z)
# target_include_directories(bench_docfeat PUBLIC ${CMAKE_HOME_DIRECTORY}/feature-gen/src)

# bench bigram ephemeral posting
add_executable(bench_bigram_ep feature-gen/src/bench/bench_bigram_ep.cpp feature-gen/src/w_scanner.cpp)
add_dependencies(bench_bigram_ep indri_proj)
target_link_libraries(bench_bigram_ep indri pthread z)
target_include_directories(bench_bigram_ep PUBLIC ${CMAKE_HOME_DIRECTORY}/feature-gen/src)
