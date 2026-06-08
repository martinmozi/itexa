-- Wraps the host's exported C ABI in a complete LuaJIT FFI object. The raw C
-- functions stay private to this module: gameplay scripts only ever see
-- `Inventory*` cdata that behaves like a full Lua object (obj:add(...),
-- tostring(obj), #obj, ...). Method calls JIT-compile straight into the C++
-- functions -- no metatable-on-userdata, no exposed C namespace.
--
-- Game installed with `make install` expects this to be named `inv_ffi`
-- (due to CMakeLists.txt and LUA_CPATH); if loading fails, check the
-- install location, CMake rules, and the LUA_CPATH environment variable.

local ffi = require('ffi')

ffi.cdef[[
typedef struct Inventory Inventory;   /* opaque handle; real type lives in C++ */

void inv_add     (Inventory* inv, const char* name, int qty);
int  inv_remove  (Inventory* inv, const char* name, int qty);
int  inv_get     (const Inventory* inv, const char* name);
int  inv_total   (const Inventory* inv);
int  inv_size    (const Inventory* inv);
int  inv_tostring(const Inventory* inv, char* out, int cap);
]]

-- Private handle to the exported symbols; never returned to callers.
local C = ffi.C

-- Reused scratch buffer for __tostring (avoids per-call allocation).
local TOSTRING_CAP = 256
local tostring_buf = ffi.new('char[?]', TOSTRING_CAP)

-- Methods reachable through __index. `self` is the Inventory* cdata, i.e. the
-- same C++ instance owned by the host.
local methods = {
    add = function(self, name, qty)
        C.inv_add(self, name, qty)
    end,

    remove = function(self, name, qty)
        return C.inv_remove(self, name, qty)
    end,

    get = function(self, name)
        return C.inv_get(self, name)
    end,

    total = function(self)
        return C.inv_total(self)
    end,

    size = function(self)
        return C.inv_size(self)
    end,
}

local Inventory_mt = {
    __index = methods,

    -- Mirror the old userdata rule: the object may only change via its methods,
    -- so a stray field write is rejected with a clear message.
    __newindex = function(_, key)
        error(("direct field assignment is not allowed (tried to set '%s'); use Inventory methods")
            :format(tostring(key)), 2)
    end,

    -- tostring(obj) -> "Inventory{...}", rendered by the C++ side.
    __tostring = function(self)
        local n = C.inv_tostring(self, tostring_buf, TOSTRING_CAP)
        if n >= TOSTRING_CAP then
            local big = ffi.new('char[?]', n + 1) -- grow once if it didn't fit
            C.inv_tostring(self, big, n + 1)
            return ffi.string(big, n)
        end
        return ffi.string(tostring_buf, n)
    end,

    -- #obj -> number of distinct item names.
    __len = function(self)
        return C.inv_size(self)
    end,
}

-- Associating the metatype with the *struct* makes every `Inventory*` cdata --
-- including the one produced by ffi.cast('Inventory*', ...) in the host -- a
-- complete object. Requiring this module is what installs the behavior; the
-- returned ctype is handy for ffi.istype('Inventory', x) checks.
return ffi.metatype('Inventory', Inventory_mt)