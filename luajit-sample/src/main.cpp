#include <chrono>
#include <iostream>
#include <string>

#include "Inventory.h"

extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

static bool run_string(lua_State* L, const std::string& source, const char* chunk_name) {
    if (luaL_loadbuffer(L, source.c_str(), source.size(), chunk_name) != LUA_OK) {
        std::cerr << "[load error] " << lua_tostring(L, -1) << "\n";
        lua_pop(L, 1);
        return false;
    }

    if (lua_pcall(L, 0, LUA_MULTRET, 0) != LUA_OK) {
        std::cerr << "[runtime error] " << lua_tostring(L, -1) << "\n";
        lua_pop(L, 1);
        return false;
    }

    return true;
}

// Runs the precompiled main chunk (stored in the registry under `chunk_ref`)
// once, passing `inv` as its single argument. Returns the elapsed time in ms.
static double run_main_once(lua_State* L, int chunk_ref, Inventory* inv) {
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();

    // Push a fresh copy of the compiled chunk and its argument. The Inventory
    // pointer is handed over as a lightuserdata; the script turns it into a
    // typed `Inventory*` cdata via ffi.cast.
    lua_rawgeti(L, LUA_REGISTRYINDEX, chunk_ref);
    lua_pushlightuserdata(L, inv);

    if (lua_pcall(L, 1, 0, 0) != LUA_OK) {
        std::cerr << "[runtime error] " << lua_tostring(L, -1) << "\n";
        lua_pop(L, 1);
    }

    const auto end = clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {
    const bool debug_mode = (argc > 1 && std::string(argv[1]) == "--debug");

    lua_State* L = luaL_newstate();
    if (!L) {
        std::cerr << "Failed to create Lua state\n";
        return 1;
    }

    luaL_openlibs(L);

    const char* mode = debug_mode ? "true" : "false";

    // One-time setup script: JIT/debugger configuration + module search path.
    // This runs only once, before any timed execution of the main script.
    std::string setup_script =
        "local DEBUG_MODE = " + std::string(mode) + "\n" +
        R"lua(
if DEBUG_MODE then
  -- Disable the JIT compiler so the debugger can step line-by-line.
  local okjit, jit_mod = pcall(require, 'jit')
  if okjit and jit_mod and jit_mod.off then
    jit_mod.off()
    print('[debug] LuaJIT disabled via jit.off()')
  else
    print('[debug] jit module unavailable')
  end

  -- Allow loading the native EmmyLua debugger placed next to the executable.
  package.cpath = './?.dll;./?.so;' .. package.cpath

  -- emmy_core opens a TCP socket and waits for VS Code (EmmyLua extension) to attach.
  local ok, dbg = pcall(require, 'emmy_core')
  if ok then
    local host = os.getenv('EMMY_HOST') or 'localhost'
    local port = tonumber(os.getenv('EMMY_PORT') or '9966')
    dbg.tcpListen(host, port)
    print(('[debug] EmmyLua debugger listening on %s:%d'):format(host, port))
    print('[debug] waiting for VS Code to attach...')
    dbg.waitIDE()
    dbg.breakHere()
  else
    print('[debug] emmy_core not found; running without an attached debugger')
  end
end

package.path = './scripts/?.lua;' .. package.path

-- Pre-load the modules now (one-time disk read + compile, and the one-time
-- ffi.cdef inside inv_ffi) so that BOTH timed runs hit the in-memory
-- package.loaded cache and never touch disk.
require('inv_ffi')
require('mod_a')
require('mod_b')
)lua";

    if (!run_string(L, setup_script, "setup")) {
        lua_close(L);
        return 1;
    }

    // Main Lua script (built in C++): receives the target pointer as its only
    // argument, casts it to an Inventory* cdata once, and calls each module's
    // main(). Modules are already cached in memory, so this runs with zero disk
    // or console I/O -- it is the hot path we actually time. Compiled once,
    // executed multiple times.
    std::string main_script = R"lua(
local ffi = require('ffi')
require('inv_ffi') -- ensures the Inventory cdef exists before the cast
local obj = ffi.cast('Inventory*', ...)
local modules = { 'mod_a', 'mod_b' }
for _, name in ipairs(modules) do
  require(name).main(obj)
end
)lua";

    // Compile the main script once with luaL_loadstring.
    if (luaL_loadstring(L, main_script.c_str()) != LUA_OK) {
        std::cerr << "[load error] " << lua_tostring(L, -1) << "\n";
        lua_pop(L, 1);
        lua_close(L);
        return 1;
    }
    // Store the compiled chunk in the registry so we can run it repeatedly.
    const int chunk_ref = luaL_ref(L, LUA_REGISTRYINDEX);

    // First run over the first object (cold: JIT not warmed up yet).
    Inventory inventory_a;
    inventory_a.add("gold", 100);
    const double ms_first = run_main_once(L, chunk_ref, &inventory_a);

    // Second run over a different object (warm: JIT likely active now).
    Inventory inventory_b;
    inventory_b.add("silver", 50);
    const double ms_second = run_main_once(L, chunk_ref, &inventory_b);

    // Release the cached chunk.
    luaL_unref(L, LUA_REGISTRYINDEX, chunk_ref);

    // All console output happens here, AFTER the timed section.
    std::cout << "object A: " << inventory_a.toString() << "\n";
    std::cout << "object B: " << inventory_b.toString() << "\n";
    std::cout << "run #1 (cold): " << ms_first << " ms\n";
    std::cout << "run #2 (warm): " << ms_second << " ms\n";

    lua_close(L);
    return 0;
}
