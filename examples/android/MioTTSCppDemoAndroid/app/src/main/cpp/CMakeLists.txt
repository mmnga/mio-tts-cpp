cmake_minimum_required(VERSION 3.22.1)

project("mio-tts-android" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED ANDROID_ABI)
    message(STATUS "Detected Android ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_OPENMP OFF)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_OPENMP OFF)
    else()
        message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
    endif()
endif()

# Keep Android build deterministic and avoid FetchContent failures in NDK CMake.
set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

set(PROJECT_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../../../../..)
set(LLAMA_SRC ${PROJECT_ROOT}/llama.cpp)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_SRC} build-llama)

add_library(mio-tts-lib STATIC
    ${PROJECT_ROOT}/src/mio-tts-lib.cpp
    ${PROJECT_ROOT}/src/miocodec-decoder.cpp
    ${PROJECT_ROOT}/src/wavlm-extractor.cpp)

target_compile_features(mio-tts-lib PRIVATE cxx_std_17)
target_include_directories(mio-tts-lib
    PUBLIC
        ${PROJECT_ROOT}/src
    PRIVATE
        ${LLAMA_SRC}
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/vendor
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src)
target_link_libraries(mio-tts-lib
    PUBLIC
        llama
        ggml
        ggml-base
        ggml-cpu
        android
        log)

add_library(${CMAKE_PROJECT_NAME} SHARED
    mio_tts_android_jni.cpp)

target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${PROJECT_ROOT}/src
    ${LLAMA_SRC}
    ${LLAMA_SRC}/include
    ${LLAMA_SRC}/vendor
    ${LLAMA_SRC}/ggml/include
    ${LLAMA_SRC}/ggml/src)
target_link_libraries(${CMAKE_PROJECT_NAME}
    mio-tts-lib
    llama
    ggml
    ggml-base
    ggml-cpu
    android
    log)
