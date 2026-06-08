#pragma once

// Pure C ABI exposed to LuaJIT's FFI. Each function performs an internal call
// into the matching C++ Inventory method. No Lua headers are needed here: the
// scripts declare matching prototypes via ffi.cdef and call these exported
// symbols directly (resolved from the host executable by ffi.C).

class Inventory; // real C++ type; opaque across the FFI boundary

#if defined(_WIN32)
#  define INV_FFI_API extern "C" __declspec(dllexport)
#else
#  define INV_FFI_API extern "C" __attribute__((visibility("default")))
#endif

INV_FFI_API void inv_add(Inventory* inv, const char* name, int qty);
INV_FFI_API int  inv_remove(Inventory* inv, const char* name, int qty);
INV_FFI_API int  inv_get(const Inventory* inv, const char* name);
INV_FFI_API int  inv_total(const Inventory* inv);
INV_FFI_API int  inv_size(const Inventory* inv);

// Writes the human-readable representation into `out` (NUL-terminated, capped
// at `cap` bytes) and returns the full length, like snprintf.
INV_FFI_API int  inv_tostring(const Inventory* inv, char* out, int cap);