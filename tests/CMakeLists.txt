
project(GRNS_TEST LANGUAGES CUDA CXX)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

#####################################
# OpenMP
#####################################
find_package(OpenMP REQUIRED)

# Test for checking the correctness of the extended-range floating-point routines
add_executable(test_verify_extrange test_verify_extrange.cu)
target_compile_options(test_verify_extrange PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)

# Test for checking the correctness of the mixed-radix correction routines
add_executable(test_verify_mrc test_verify_mrc.cu)
target_compile_options(test_verify_mrc PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_verify_mrc gmp mpfr)

# Test for checking the algorithms that calculate the interval evaluation of an RNS number
add_executable(test_verify_rnseval test_verify_rnseval.cu)
target_compile_options(test_verify_rnseval PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_verify_rnseval gmp mpfr)

# Test for measure the performance of the algorithms that calculate the RNS interval evaluation
add_executable(test_perf_rnseval test_perf_rnseval.cu)
target_compile_options(test_perf_rnseval PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_perf_rnseval gmp mpfr)

# Test for checking the RNS division algorithms
add_executable(test_verify_rnsdiv test_verify_rnsdiv.cu)
target_compile_options(test_verify_rnsdiv PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_verify_rnsdiv gmp mpfr)

# Test for measure the performance of the RNS division algorithms
add_executable(test_perf_rnsdiv test_perf_rnsdiv.cu)
target_compile_options(test_perf_rnsdiv PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_perf_rnsdiv gmp mpfr)

# Test for measure the performance of the RNS magnitude comparison algorithms
add_executable(test_perf_rnscmp test_perf_rnscmp.cu)
target_compile_options(test_perf_rnscmp PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_perf_rnscmp gmp mpfr)