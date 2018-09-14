
# forward_index
add_executable(create_forward_index feature-gen/src/create_forward_index.cpp)
add_dependencies(create_forward_index indri_proj)
set_target_properties(create_forward_index PROPERTIES COMPILE_FLAGS ${INDRI_DEP_FLAGS})
target_link_libraries(create_forward_index FastPFor indri lemur antlr pthread z)

# create_doc_lens
add_executable(create_doc_lens feature-gen/src/create_doc_lens.cpp)
add_dependencies(create_doc_lens indri_proj)
set_target_properties(create_doc_lens PROPERTIES COMPILE_FLAGS ${INDRI_DEP_FLAGS})
target_link_libraries(create_doc_lens indri lemur antlr pthread z)

# lexicon
add_executable(create_lexicon feature-gen/src/create_lexicon.cpp)
add_dependencies(create_lexicon indri_proj)
set_target_properties(create_lexicon PROPERTIES COMPILE_FLAGS ${INDRI_DEP_FLAGS})
target_link_libraries(create_lexicon indri lemur antlr pthread z)

# inverted_index
add_executable(create_inverted_index feature-gen/src/create_inverted_index.cpp)
add_dependencies(create_inverted_index indri_proj)
set_target_properties(create_inverted_index PROPERTIES COMPILE_FLAGS ${INDRI_DEP_FLAGS})
target_link_libraries(create_inverted_index indri lemur FastPFor pthread z)

# create_bigram_inverted_index
add_executable(create_bigram_inverted_index feature-gen/src/create_bigram_inverted_index.cpp)
add_dependencies(create_bigram_inverted_index indri_proj)
set_target_properties(create_bigram_inverted_index PROPERTIES COMPILE_FLAGS ${INDRI_DEP_FLAGS})
target_link_libraries(create_bigram_inverted_index indri lemur FastPFor pthread z)
target_include_directories(create_bigram_inverted_index PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/feature-gen/src)

# generate_term_features
add_executable(generate_term_features feature-gen/src/generate_term_features.cpp)
target_link_libraries(generate_term_features FastPFor m)

# pre-retrieval csv
add_executable(preret_csv feature-gen/src/preret_csv.cpp
    feature-gen/src/fgen_term_qry.c feature-gen/src/query_features.c
    feature-gen/src/fgen_bigram_qry.c
    feature-gen/src/strbuf.c)
add_dependencies(preret_csv indri_proj)
set_target_properties(preret_csv PROPERTIES COMPILE_FLAGS ${INDRI_DEP_FLAGS})
target_link_libraries(preret_csv indri lemur pthread z)

# fgtrain
add_executable(fgtrain feature-gen/src/fgtrain.cpp
    feature-gen/src/fgen_term_qry.c feature-gen/src/query_features.c
    feature-gen/src/fgen_bigram_qry.c
    feature-gen/src/strbuf.c)
add_dependencies(fgtrain create_bigram_inverted_index indri_proj)
set_target_properties(fgtrain PROPERTIES COMPILE_FLAGS ${INDRI_DEP_FLAGS})
target_link_libraries(fgtrain indri lemur antlr pthread FastPFor z)

# install
install(TARGETS
    create_forward_index
    create_doc_lens
    create_lexicon
    create_inverted_index
    create_bigram_inverted_index
    generate_term_features
    preret_csv
    fgtrain
    DESTINATION bin)