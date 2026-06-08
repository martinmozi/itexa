#pragma once

extern "C" {
#include <lua.h>
}

class Inventory;

// Registers the Inventory metatable in the given Lua state.
void register_inventory(lua_State* L);

// Pushes an existing Inventory pointer onto the Lua stack as userdata that
// carries the Inventory metatable. The C++ side keeps ownership.
void push_inventory(lua_State* L, Inventory* inv);
