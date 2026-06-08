#include "InventoryLua.h"
#include "Inventory.h"

extern "C" {
#include <lauxlib.h>
}

namespace {

const char* INVENTORY_MT = "Cpp.Inventory";

// Unwrap the userdata back into the Inventory pointer (validates metatable).
Inventory* check_inventory(lua_State* L, int idx) {
    void* ud = luaL_checkudata(L, idx, INVENTORY_MT);
    return *static_cast<Inventory**>(ud);
}

// inv:add(name, qty)
int l_inv_add(lua_State* L) {
    Inventory* inv = check_inventory(L, 1);
    const char* name = luaL_checkstring(L, 2);
    int qty = static_cast<int>(luaL_checkinteger(L, 3));
    inv->add(name, qty);
    return 0;
}

// inv:remove(name, qty) -> removed
int l_inv_remove(lua_State* L) {
    Inventory* inv = check_inventory(L, 1);
    const char* name = luaL_checkstring(L, 2);
    int qty = static_cast<int>(luaL_checkinteger(L, 3));
    lua_pushinteger(L, inv->remove(name, qty));
    return 1;
}

// inv:get(name) -> qty
int l_inv_get(lua_State* L) {
    Inventory* inv = check_inventory(L, 1);
    const char* name = luaL_checkstring(L, 2);
    lua_pushinteger(L, inv->get(name));
    return 1;
}

// inv:total() -> sum
int l_inv_total(lua_State* L) {
    Inventory* inv = check_inventory(L, 1);
    lua_pushinteger(L, inv->total());
    return 1;
}

// inv:keys() -> { name1, name2, ... }
int l_inv_keys(lua_State* L) {
    Inventory* inv = check_inventory(L, 1);
    std::vector<std::string> keys = inv->keys();
    lua_createtable(L, static_cast<int>(keys.size()), 0);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        lua_pushstring(L, keys[i].c_str());
        lua_rawseti(L, -2, static_cast<int>(i + 1));
    }
    return 1;
}

// #inv -> number of distinct items
int l_inv_len(lua_State* L) {
    Inventory* inv = check_inventory(L, 1);
    lua_pushinteger(L, static_cast<lua_Integer>(inv->size()));
    return 1;
}

// tostring(inv)
int l_inv_tostring(lua_State* L) {
    Inventory* inv = check_inventory(L, 1);
    lua_pushstring(L, inv->toString().c_str());
    return 1;
}

// Block raw field writes; the object may only change via its methods.
int l_inv_newindex(lua_State* L) {
    return luaL_error(L,
        "direct field assignment is not allowed; use Inventory methods");
}

} // namespace

void register_inventory(lua_State* L) {
    luaL_newmetatable(L, INVENTORY_MT);

    lua_newtable(L); // method table used as __index
    lua_pushcfunction(L, l_inv_add);    lua_setfield(L, -2, "add");
    lua_pushcfunction(L, l_inv_remove); lua_setfield(L, -2, "remove");
    lua_pushcfunction(L, l_inv_get);    lua_setfield(L, -2, "get");
    lua_pushcfunction(L, l_inv_total);  lua_setfield(L, -2, "total");
    lua_pushcfunction(L, l_inv_keys);   lua_setfield(L, -2, "keys");
    lua_setfield(L, -2, "__index");

    lua_pushcfunction(L, l_inv_newindex); lua_setfield(L, -2, "__newindex");
    lua_pushcfunction(L, l_inv_tostring); lua_setfield(L, -2, "__tostring");
    lua_pushcfunction(L, l_inv_len);      lua_setfield(L, -2, "__len");

    lua_pop(L, 1); // pop metatable
}

void push_inventory(lua_State* L, Inventory* inv) {
    Inventory** ud = static_cast<Inventory**>(lua_newuserdata(L, sizeof(Inventory*)));
    *ud = inv;
    luaL_getmetatable(L, INVENTORY_MT);
    lua_setmetatable(L, -2);
}
