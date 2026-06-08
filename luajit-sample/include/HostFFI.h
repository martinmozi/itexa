#pragma once

// Pure C ABI exposed to LuaJIT's FFI for every host object type. Each function
// performs an internal call into the matching C++ method. No Lua headers are
// needed here: the scripts declare matching prototypes via ffi.cdef and call
// these exported symbols directly (resolved from the host executable by ffi.C).
//
// The World functions are the registry entry points: given a World* they return
// typed handles (Inventory*/Account*) to other objects, so a script reaches any
// object from the single World handle it receives.

class World;     // real C++ types; opaque across the FFI boundary
class Inventory;
class Account;

#if defined(_WIN32)
#  define HOST_FFI_API extern "C" __declspec(dllexport)
#else
#  define HOST_FFI_API extern "C" __attribute__((visibility("default")))
#endif

// --- World registry: hand out typed handles (nullptr -> Lua nil) -----------
HOST_FFI_API Inventory* world_inventory(World* w, int id);
HOST_FFI_API Account*   world_account(World* w, int id);

// --- Inventory -------------------------------------------------------------
HOST_FFI_API void inv_add(Inventory* inv, const char* name, int qty);
HOST_FFI_API int  inv_remove(Inventory* inv, const char* name, int qty);
HOST_FFI_API int  inv_get(const Inventory* inv, const char* name);
HOST_FFI_API int  inv_total(const Inventory* inv);
HOST_FFI_API int  inv_size(const Inventory* inv);
HOST_FFI_API int  inv_tostring(const Inventory* inv, char* out, int cap);

// --- Account ---------------------------------------------------------------
HOST_FFI_API void acc_deposit(Account* acc, int amount);
HOST_FFI_API int  acc_withdraw(Account* acc, int amount);
HOST_FFI_API int  acc_balance(const Account* acc);
HOST_FFI_API int  acc_tostring(const Account* acc, char* out, int cap);
