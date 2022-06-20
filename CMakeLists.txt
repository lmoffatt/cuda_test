cmake_minimum_required(VERSION 3.8)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.2/bin/nvcc)
project(cuda_test LANGUAGES CUDA CXX)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cuda_test main.cpp static_string.h cuda_test_functions.h cuda_test_transporter.h cuda_test_cpu.h cuda_test_gpu.h cuda_test_compile.h cuda_test_instantiate.h)
set_property(TARGET cuda_test PROPERTY CUDA_ARCHITECTURES 35 50 72)
set_property(TARGET cuda_test PROPERTY CUDA_STANDARD 17)

#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
#target_compile_options(cuda_test PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
#target_include_directories(cuda_test PUBLIC /usr/local/cuda-11.0/targets/x86_64-linux/include/cuda/std/detail/libcxx/include/)
target_compile_options(cuda_test PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17 -g -ftemplate-backtrace-limit=0 -ftemplate-depth=1024 -expt-relaxed-constexpr >)
#target_compile_options(cuda_test PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17 -ftemplate-backtrace-limit=0 -ftemplate-depth=1024 --gpu-architecture=compute_61 --gpu-code=sm_61>)
target_compile_options(cuda_test PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>)

#find_package(CUDA 11.0 REQUIRED)
set_target_properties(cuda_test PROPERTIES LINKER_LANGUAGE CXX)
#set_source_files_properties(main.cpp PROPERTIES LANGUAGE CXX)


