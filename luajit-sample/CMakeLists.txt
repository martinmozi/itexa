cmake_minimum_required(VERSION 3.16)
project(luajit_aim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point this to your LuaJIT installation if auto-detection fails.
# Example:
#   cmake -S . -B build -DLUAJIT_ROOT="C:/DevTools/aimlibs/Debug/luajit-2.1"
set(LUAJIT_ROOT "C:/DevTools/aimlibs/Debug/luajit-2.1" CACHE PATH "Path to LuaJIT root")

if(LUAJIT_ROOT)
    find_path(LUAJIT_INCLUDE_DIR luajit.h
        PATHS
            "${LUAJIT_ROOT}/include/luajit-2.1"
            "${LUAJIT_ROOT}/include"
        NO_DEFAULT_PATH
    )

    find_library(LUAJIT_LIBRARY
        NAMES lua51 luajit-5.1 luajit
        PATHS
            "${LUAJIT_ROOT}"
            "${LUAJIT_ROOT}/lib"
            "${LUAJIT_ROOT}/src"
        NO_DEFAULT_PATH
    )
else()
    find_path(LUAJIT_INCLUDE_DIR luajit.h
        PATH_SUFFIXES luajit-2.1
    )

    find_library(LUAJIT_LIBRARY
        NAMES lua51 luajit-5.1 luajit
    )
endif()

if(NOT LUAJIT_INCLUDE_DIR OR NOT LUAJIT_LIBRARY)
    message(FATAL_ERROR "LuaJIT not found. Set -DLUAJIT_ROOT=<path>")
endif()

add_executable(luajit_embed
    src/main.cpp
    src/Inventory.cpp
    src/Account.cpp
    src/World.cpp
    src/HostFFI.cpp
)

# Export the C ABI symbols (world_*, inv_*, acc_*) from the executable so
# LuaJIT's ffi.C can resolve them at runtime. On Windows this produces an
# import lib and honors __declspec(dllexport); on Unix it adds -rdynamic.
set_target_properties(luajit_embed PROPERTIES ENABLE_EXPORTS ON)

target_include_directories(luajit_embed PRIVATE
    "${LUAJIT_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/include"
)
target_link_libraries(luajit_embed PRIVATE "${LUAJIT_LIBRARY}")

# Copy Lua modules next to the executable for package.path = './scripts/?.lua;...'
add_custom_command(TARGET luajit_embed POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/scripts"
            "$<TARGET_FILE_DIR:luajit_embed>/scripts"
)

# Copy the LuaJIT runtime DLL next to the executable on Windows.
if(WIN32)
    find_file(LUAJIT_DLL
        NAMES lua51.dll luajit.dll
        PATHS "${LUAJIT_ROOT}" "${LUAJIT_ROOT}/bin"
        NO_DEFAULT_PATH
    )
    if(LUAJIT_DLL)
        add_custom_command(TARGET luajit_embed POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${LUAJIT_DLL}"
                    "$<TARGET_FILE_DIR:luajit_embed>"
        )
    endif()
endif()
