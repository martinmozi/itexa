-- Wraps the host's exported C ABI in complete LuaJIT FFI objects. The raw C
-- functions stay private to this module: processing scripts only ever see
-- typed cdata (World*/Inventory*/Account*) that behave like full Lua objects
-- (obj:method(...), tostring(obj), #obj, ...). Method calls JIT-compile straight
-- into the C++ functions -- no metatable-on-userdata, no exposed C namespace.
--
-- A script receives a World handle and reaches every other object from it:
--   local inv = world:inventory(1)   -- nil if no such id
--   if inv then inv:add("gold", 10) end
--
-- Installed with `make install` this is expected to be named `host_ffi`
-- (due to CMakeLists.txt and LUA_CPATH); if loading fails, check the
-- install location, CMake rules, and the LUA_CPATH environment variable.

local ffi = require('ffi')

ffi.cdef[[
typedef struct World     World;       /* opaque handles; real types live in C++ */
typedef struct Inventory Inventory;
typedef struct Account   Account;

/* World registry: returns a typed handle, or NULL for an unknown id. */
Inventory* world_inventory(World* w, int id);
Account*   world_account  (World* w, int id);

/* Inventory */
void inv_add     (Inventory* inv, const char* name, int qty);
int  inv_remove  (Inventory* inv, const char* name, int qty);
int  inv_get     (const Inventory* inv, const char* name);
int  inv_total   (const Inventory* inv);
int  inv_size    (const Inventory* inv);
int  inv_tostring(const Inventory* inv, char* out, int cap);

/* Account */
void acc_deposit (Account* acc, int amount);
int  acc_withdraw(Account* acc, int amount);
int  acc_balance (const Account* acc);
int  acc_tostring(const Account* acc, char* out, int cap);
]]

-- Private handle to the exported symbols; never returned to callers.
local C = ffi.C

-- Reused scratch buffer for __tostring (avoids per-call allocation).
local TOSTRING_CAP = 256
local tostring_buf = ffi.new('char[?]', TOSTRING_CAP)

-- Shared __tostring helper: call the C renderer into the scratch buffer, growing
-- once if the text didn't fit.
local function render(cfunc, self)
    local n = cfunc(self, tostring_buf, TOSTRING_CAP)
    if n >= TOSTRING_CAP then
        local big = ffi.new('char[?]', n + 1)
        cfunc(self, big, n + 1)
        return ffi.string(big, n)
    end
    return ffi.string(tostring_buf, n)
end

-- Factory for the __newindex guard: the object may only change via its methods,
-- so a stray field write is rejected with a clear, type-specific message.
local function readonly_newindex(label)
    return function(_, key)
        error(("direct field assignment is not allowed on %s (tried to set '%s'); use its methods")
            :format(label, tostring(key)), 2)
    end
end

-- Each C function's signature already starts with the receiver, so
-- `obj:method(args)` -> `c_func(obj, args)` lines up exactly -- the C function
-- IS the method, no Lua wrapper needed. (FFI C functions are first-class
-- callable cdata; const-ptr params accept the non-const receiver implicitly,
-- Lua strings convert to const char*.)

ffi.metatype('Inventory', {
    __index = {
        add    = C.inv_add,
        remove = C.inv_remove,
        get    = C.inv_get,
        total  = C.inv_total,
        size   = C.inv_size,
    },
    __newindex = readonly_newindex('Inventory'),
    __tostring = function(self) return render(C.inv_tostring, self) end,
    __len      = function(self) return C.inv_size(self) end,
})

ffi.metatype('Account', {
    __index = {
        deposit  = C.acc_deposit,
        withdraw = C.acc_withdraw,
        balance  = C.acc_balance,
    },
    __newindex = readonly_newindex('Account'),
    __tostring = function(self) return render(C.acc_tostring, self) end,
})

-- World accessors wrap the C lookups so a missing id comes back as Lua nil
-- (a NULL cdata is truthy, which would be a footgun for `if inv then`). The
-- returned non-nil handle is already a full Inventory/Account object.
return ffi.metatype('World', {
    __index = {
        inventory = function(self, id)
            local p = C.world_inventory(self, id)
            if p == nil then return nil end   -- NULL cdata compares equal to nil
            return p
        end,
        account = function(self, id)
            local p = C.world_account(self, id)
            if p == nil then return nil end
            return p
        end,
    },
    __newindex = readonly_newindex('World'),
})
