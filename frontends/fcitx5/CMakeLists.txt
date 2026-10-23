# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of KIRAL. The SPDX lines name its copyright holder and its
# license, in the machine-readable form of the REUSE specification: the GNU
# General Public License, version 3 or any later version
# (LICENSES/GPL-3.0-or-later.txt).
#
# KIRAL's addon for Fcitx 5: kiral.cpp, built by Clang as C++26 against Fcitx
# 5's development files and KIRAL's C ABI, the static library that cargo builds
# here from the crate kiral-capi, into the build directory. Every warning Clang
# has is on, but those about what is not the code, as stable-diffusion.assembly
# builds its sources, and every warning stops the build (KIRAL_WERROR).
#
#     cmake -S frontends/fcitx5 -B build/fcitx5 -G Ninja -DCMAKE_CXX_COMPILER=clang++
#     cmake --build build/fcitx5 && ctest --test-dir build/fcitx5
#     cmake --install build/fcitx5

cmake_minimum_required(VERSION 3.29)
project(kiral-fcitx5 VERSION 0.1.0 LANGUAGES CXX)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "KIRAL's addon is built by Clang, not ${CMAKE_CXX_COMPILER_ID}: -DCMAKE_CXX_COMPILER=clang++")
endif ()

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_LINKER_TYPE LLD)

option(KIRAL_WERROR "Every warning stops the build" ON)
option(KIRAL_TESTS "Build the addon's tests, which ctest runs" ON)

find_package(Fcitx5Core 5.1 REQUIRED)
find_package(Fcitx5Utils 5.1 REQUIRED)
find_package(Threads REQUIRED)

set(KIRAL_WARNINGS
    -Weverything
    -Wno-c++98-compat-pedantic -Wno-pre-c++14-compat-pedantic -Wno-pre-c++17-compat-pedantic
    -Wno-pre-c++20-compat-pedantic -Wno-pre-c++23-compat-pedantic -Wno-pre-c++26-compat-pedantic
    -Wno-padded -Wno-switch-default)
if (KIRAL_WERROR)
    list(APPEND KIRAL_WARNINGS -Werror)
endif ()

# KIRAL's C ABI: the workspace's static library, built by cargo in a target
# directory of the build's own.
set(KIRAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "KIRAL's workspace")
find_program(KIRAL_CARGO cargo HINTS "$ENV{HOME}/.cargo/bin" REQUIRED)
set(KIRAL_CARGO_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo")
set(KIRAL_CAPI_LIBRARY "${KIRAL_CARGO_TARGET_DIR}/release/libkiral_capi.a")
add_custom_target(kiral-capi-build
    COMMAND "${KIRAL_CARGO}" build --release --locked -p kiral-capi --target-dir "${KIRAL_CARGO_TARGET_DIR}"
    WORKING_DIRECTORY "${KIRAL_SOURCE_DIR}"
    BYPRODUCTS "${KIRAL_CAPI_LIBRARY}"
    USES_TERMINAL
    COMMENT "Building KIRAL's C ABI with cargo")
add_library(kiral-capi STATIC IMPORTED)
set_target_properties(kiral-capi PROPERTIES
    IMPORTED_LOCATION "${KIRAL_CAPI_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${KIRAL_SOURCE_DIR}/crates/kiral-capi/include"
    # What rustc says a static library of Rust's std needs of the system's
    # (rustc --print native-static-libs: -lgcc_s -lutil -lrt -lpthread -lm -ldl
    # -lc), the C++ runtime bringing libgcc_s and libc.
    INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS};m;rt;util")
add_dependencies(kiral-capi kiral-capi-build)

add_library(kiral MODULE kiral.cpp)
set_target_properties(kiral PROPERTIES PREFIX "")
target_compile_options(kiral PRIVATE ${KIRAL_WARNINGS})
target_link_libraries(kiral PRIVATE Fcitx5::Core kiral-capi)
# The C ABI's symbols stay inside the addon, which exports Fcitx's factory
# alone; nothing the addon does not use is kept.
target_link_options(kiral PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--as-needed" "LINKER:--gc-sections")

configure_file(addon.conf.in "${CMAKE_CURRENT_BINARY_DIR}/addon/kiral.conf" @ONLY)
configure_file(inputmethod.conf.in "${CMAKE_CURRENT_BINARY_DIR}/inputmethod/kiral.conf" @ONLY)

install(TARGETS kiral DESTINATION "${FCITX_INSTALL_ADDONDIR}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/addon/kiral.conf" DESTINATION "${FCITX_INSTALL_PKGDATADIR}/addon")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/inputmethod/kiral.conf" DESTINATION "${FCITX_INSTALL_PKGDATADIR}/inputmethod")

if (KIRAL_TESTS)
    enable_testing()
    find_package(Fcitx5Module REQUIRED COMPONENTS TestFrontend)
    # Fcitx's test frontend and test UI, installed with its modules, beside
    # Fcitx's library, their addon files under its data directory's testing/.
    find_path(KIRAL_FCITX_TESTING_DIR addon/testfrontend.conf
        PATHS "${FCITX_SYS_INSTALL_PREFIX}/share/fcitx5/testing" "${FCITX_INSTALL_PKGDATADIR}/testing" REQUIRED NO_DEFAULT_PATH)
    add_executable(testkiral test/testkiral.cpp)
    target_compile_options(testkiral PRIVATE ${KIRAL_WARNINGS})
    # The test frontend's header, Fcitx's, warned of as a system header is.
    get_target_property(KIRAL_TEST_FRONTEND_INCLUDE Fcitx5::Module::TestFrontend INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(testkiral SYSTEM PRIVATE ${KIRAL_TEST_FRONTEND_INCLUDE})
    target_link_libraries(testkiral PRIVATE Fcitx5::Core)
    target_compile_definitions(testkiral PRIVATE
        "KIRAL_TEST_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\""
        "KIRAL_FCITX_ADDON_DIR=\"$<TARGET_FILE_DIR:Fcitx5::Core>/fcitx5\""
        "KIRAL_FCITX_TESTING_DIR=\"${KIRAL_FCITX_TESTING_DIR}\"")
    add_dependencies(testkiral kiral)
    # A data directory for each test: the addon's files, and the fixtures --
    # a dictionary, and for one a model -- that cargo writes.
    foreach (fixture dictionary model)
        set(data "${CMAKE_CURRENT_BINARY_DIR}/test/${fixture}")
        configure_file(addon.conf.in "${data}/addon/kiral.conf" @ONLY)
        configure_file(inputmethod.conf.in "${data}/inputmethod/kiral.conf" @ONLY)
        set(files "${data}/kiral/dictionary.kdic")
        if (fixture STREQUAL "model")
            list(APPEND files "${data}/kiral/model.kmdl")
        endif ()
        add_test(NAME fixtures-${fixture}
            COMMAND "${KIRAL_CARGO}" run --release --locked -q -p kiral-capi --example fixtures
                --target-dir "${KIRAL_CARGO_TARGET_DIR}" -- ${files}
            WORKING_DIRECTORY "${KIRAL_SOURCE_DIR}")
        set_tests_properties(fixtures-${fixture} PROPERTIES FIXTURES_SETUP kiral-${fixture})
        add_test(NAME ${fixture} COMMAND testkiral ${fixture})
        set_tests_properties(${fixture} PROPERTIES FIXTURES_REQUIRED kiral-${fixture} TIMEOUT 60)
    endforeach ()
endif ()
