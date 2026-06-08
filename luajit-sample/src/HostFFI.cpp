#include "HostFFI.h"

#include "Account.h"
#include "Inventory.h"
#include "World.h"

#include <cstring>
#include <string>

// Every shim is defensive on purpose. Scripts are trusted, but an honest typo
// must never crash the host, so each function:
//   * null-checks every pointer (a script may hold a missing/stale handle, or
//     pass nil where a string/object was expected -- FFI turns nil into NULL),
//   * never lets a C++ exception cross the FFI boundary (that is undefined
//     behavior in LuaJIT), by wrapping the body in try/catch.
// On a guard hit the function degrades to a safe default (no-op / 0 / "").

namespace {

// snprintf-like copy of `s` into the caller's buffer; returns the full length.
int write_string(const std::string& s, char* out, int cap) {
    if (out && cap > 0) {
        const int n = (static_cast<int>(s.size()) < cap - 1)
                          ? static_cast<int>(s.size())
                          : cap - 1;
        std::memcpy(out, s.data(), static_cast<std::size_t>(n));
        out[n] = '\0';
    }
    return static_cast<int>(s.size());
}

} // namespace

// --- World registry --------------------------------------------------------

HOST_FFI_API Inventory* world_inventory(World* w, int id) {
    if (!w) return nullptr;
    try { return w->inventory(id); } catch (...) { return nullptr; }
}

HOST_FFI_API Account* world_account(World* w, int id) {
    if (!w) return nullptr;
    try { return w->account(id); } catch (...) { return nullptr; }
}

// --- Inventory -------------------------------------------------------------

HOST_FFI_API void inv_add(Inventory* inv, const char* name, int qty) {
    if (!inv || !name) return;
    try { inv->add(name, qty); } catch (...) {}
}

HOST_FFI_API int inv_remove(Inventory* inv, const char* name, int qty) {
    if (!inv || !name) return 0;
    try { return inv->remove(name, qty); } catch (...) { return 0; }
}

HOST_FFI_API int inv_get(const Inventory* inv, const char* name) {
    if (!inv || !name) return 0;
    try { return inv->get(name); } catch (...) { return 0; }
}

HOST_FFI_API int inv_total(const Inventory* inv) {
    if (!inv) return 0;
    try { return inv->total(); } catch (...) { return 0; }
}

HOST_FFI_API int inv_size(const Inventory* inv) {
    if (!inv) return 0;
    try { return static_cast<int>(inv->size()); } catch (...) { return 0; }
}

HOST_FFI_API int inv_tostring(const Inventory* inv, char* out, int cap) {
    if (!inv) return write_string(std::string(), out, cap);
    try { return write_string(inv->toString(), out, cap); }
    catch (...) { return write_string(std::string(), out, cap); }
}

// --- Account ---------------------------------------------------------------

HOST_FFI_API void acc_deposit(Account* acc, int amount) {
    if (!acc) return;
    try { acc->deposit(amount); } catch (...) {}
}

HOST_FFI_API int acc_withdraw(Account* acc, int amount) {
    if (!acc) return 0;
    try { return acc->withdraw(amount); } catch (...) { return 0; }
}

HOST_FFI_API int acc_balance(const Account* acc) {
    if (!acc) return 0;
    try { return acc->balance(); } catch (...) { return 0; }
}

HOST_FFI_API int acc_tostring(const Account* acc, char* out, int cap) {
    if (!acc) return write_string(std::string(), out, cap);
    try { return write_string(acc->toString(), out, cap); }
    catch (...) { return write_string(std::string(), out, cap); }
}
