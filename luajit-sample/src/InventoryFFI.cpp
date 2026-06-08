#include "InventoryFFI.h"
#include "Inventory.h"

#include <cstring>
#include <string>

// Each exported C function is a thin shim: it takes the opaque Inventory*
// handle plus primitive/ABI-stable arguments and forwards to the C++ method.

INV_FFI_API void inv_add(Inventory* inv, const char* name, int qty) {
    inv->add(name, qty);
}

INV_FFI_API int inv_remove(Inventory* inv, const char* name, int qty) {
    return inv->remove(name, qty);
}

INV_FFI_API int inv_get(const Inventory* inv, const char* name) {
    return inv->get(name);
}

INV_FFI_API int inv_total(const Inventory* inv) {
    return inv->total();
}

INV_FFI_API int inv_size(const Inventory* inv) {
    return static_cast<int>(inv->size());
}

INV_FFI_API int inv_tostring(const Inventory* inv, char* out, int cap) {
    const std::string s = inv->toString();
    if (out && cap > 0) {
        const int n = (static_cast<int>(s.size()) < cap - 1)
                          ? static_cast<int>(s.size())
                          : cap - 1;
        std::memcpy(out, s.data(), static_cast<std::size_t>(n));
        out[n] = '\0';
    }
    return static_cast<int>(s.size());
}