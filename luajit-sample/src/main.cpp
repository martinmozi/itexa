#include <chrono>
#include <iostream>
#include <string>

#include "World.h"

extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

// How many times to run the hot path before measuring. LuaJIT only traces and
// compiles a function/loop after it crosses its hotcount threshold (~56 by
// default), so a couple of runs would still be interpreted. We run it many
// thousands of times so the JIT is fully warmed before timing.
constexpr int WARMUP_RUNS  = 100000;
constexpr int MEASURE_RUNS = 1000000;

// Kontrolovaný abort zo skriptu (return false, msg). Oddelený typ, aby ho
// host vedel odlíšiť od skutočného Lua crashu.
struct ScriptAbort : std::runtime_error {
    std::string script;
    ScriptAbort(std::string s, const std::string& msg)
        : std::runtime_error(s + ": " + msg), script(std::move(s)) {}
};

static void call_run(lua_State* L, int run_ref, World* world) {
    lua_rawgeti(L, LUA_REGISTRYINDEX, run_ref);
    lua_pushlightuserdata(L, world);

    // (B) Skutočná Lua chyba (raised error / bug): pcall vráti != LUA_OK.
    if (lua_pcall(L, 1, 2, 0) != LUA_OK) {
        const char* m = lua_tostring(L, -1);
        std::string msg = m ? m : "unknown Lua error";
        lua_pop(L, 1);
        throw std::runtime_error("script crashed: " + msg);
    }

    // (A) Kontrolovaný abort: run() vrátil (script, err); úspech => (nil, nil).
    if (!lua_isnil(L, -2)) {
        const char* s = lua_tostring(L, -2);
        const char* e = lua_tostring(L, -1);
        ScriptAbort ex(s ? s : "?", e ? e : "?");  // skopíruje do std::string...
        lua_pop(L, 2);                              // ...PRED popom (potom GC)
        throw ex;
    }
    lua_pop(L, 2);
}

// Calls the cached `reload` closure: it drops the cached script chunks and
// recompiles them from disk, so edited scripts take effect without restarting.
static bool call_reload(lua_State* L, int reload_ref) {
    lua_rawgeti(L, LUA_REGISTRYINDEX, reload_ref);
    if (lua_pcall(L, 0, 0, 0) != LUA_OK) {
        std::cerr << "[reload error] " << lua_tostring(L, -1) << "\n";
        lua_pop(L, 1);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    const bool debug_mode = (argc > 1 && std::string(argv[1]) == "--debug");

    lua_State* L = luaL_newstate();
    if (!L) {
        std::cerr << "Failed to create Lua state\n";
        return 1;
    }

    luaL_openlibs(L);

    // Load the bootstrap from disk and run it once, passing the debug flag. It
    // sets up package.path / JIT / debugger, resolves the FFI object and the
    // processing scripts, and returns the runner table { run, reload }. Loaded
    // by explicit path (relative to the working directory, where CMake copies
    // the scripts), so it is what sets up package.path for everything else.
    if (luaL_loadfile(L, "scripts/boot.lua") != LUA_OK) {
        std::cerr << "[load error] " << lua_tostring(L, -1) << "\n";
        lua_pop(L, 1);
        lua_close(L);
        return 1;
    }
    lua_pushboolean(L, debug_mode);
    if (lua_pcall(L, 1, 1, 0) != LUA_OK) { // expect one result: the runner table
        std::cerr << "[runtime error] " << lua_tostring(L, -1) << "\n";
        lua_pop(L, 1);
        lua_close(L);
        return 1;
    }

    // Pull run/reload out of the returned table into their own registry refs so
    // the hot loop never does a table lookup.
    lua_getfield(L, -1, "run");
    const int run_ref = luaL_ref(L, LUA_REGISTRYINDEX);
    lua_getfield(L, -1, "reload");
    const int reload_ref = luaL_ref(L, LUA_REGISTRYINDEX);
    lua_pop(L, 1); // pop the runner table

    using clock = std::chrono::steady_clock;

    // Populate the registry the scripts will operate on. Objects live for the
    // whole run, so the handles handed to Lua stay valid.
    World world;
    world.createInventory(1).add("gold", 100);
    world.createInventory(2);
    world.createAccount(1);

    // Cold: the very first call, fully interpreted (JIT not warmed yet).
    const auto cold_start = clock::now();
    call_run(L, run_ref, &world);
    const auto cold_end = clock::now();
    const double ms_cold = std::chrono::duration<double, std::milli>(cold_end - cold_start).count();

    // Warm-up: drive the hot path enough to trigger trace compilation.
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        call_run(L, run_ref, &world);
    }

    // Measure: time a large batch and report the per-call average. One call
    // here == processing one ISO message in the real system.
    const auto m_start = clock::now();
    for (int i = 0; i < MEASURE_RUNS; ++i) {
        call_run(L, run_ref, &world);
    }
    const auto m_end = clock::now();
    const double ms_total = std::chrono::duration<double, std::milli>(m_end - m_start).count();
    const double ns_per_call = (ms_total * 1e6) / MEASURE_RUNS;

    // Demonstrate hot reload: recompile the scripts from disk and run once more.
    // In production you would call this when a file watcher reports a change.
    const bool reloaded = call_reload(L, reload_ref);
    call_run(L, run_ref, &world);

    luaL_unref(L, LUA_REGISTRYINDEX, run_ref);
    luaL_unref(L, LUA_REGISTRYINDEX, reload_ref);

    // All console output happens here, after the timed section.
    std::cout << "inventory 1: " << world.inventory(1)->toString() << "\n";
    std::cout << "inventory 2: " << world.inventory(2)->toString() << "\n";
    std::cout << "account 1:   " << world.account(1)->toString() << "\n";
    std::cout << "reload: " << (reloaded ? "ok" : "FAILED") << "\n";
    std::cout << "cold call (interpreted): " << ms_cold << " ms\n";
    std::cout << "warm avg over " << MEASURE_RUNS << " calls: "
              << ns_per_call << " ns/call ("
              << ms_total << " ms total)\n";

    lua_close(L);
    return 0;
}
