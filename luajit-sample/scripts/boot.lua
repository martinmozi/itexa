-- One-time bootstrap, loaded by the C++ host via luaL_loadfile + pcall(1, 1):
-- it receives the debug flag as its single argument and returns the runner
-- table { run, reload }. Loaded by an explicit path (not require), so it is what
-- sets up package.path for everything that follows -- no chicken-and-egg.
--
-- All require()/path work and the ffi.cdef happen HERE, once. The returned
-- `run` is the per-message hot path and does none of it.

local DEBUG_MODE = ...

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

local ffi = require('ffi')
require('host_ffi')              -- installs the World/Inventory/Account metatypes
local cast = ffi.cast

-- Scripts that transform incoming messages; hot-reloadable on disk.
local SCRIPTS = { 'mod_a', 'mod_b' }
local n = #SCRIPTS

-- Resolved main() of each script and the single source of truth for the hot
-- path; only ever replaced atomically by load_scripts().
local mains = {}

-- Transactional (re)load: compile every script into a FRESH set and swap it in
-- only once ALL of them succeed. If any script fails to load (e.g. a syntax
-- error in an edited file), `mains` is left untouched -- the running set keeps
-- working on the old versions -- and the error is propagated to the caller
-- (the host reports it via its pcall, and the next run() still uses the old set).
local function load_scripts()
    local fresh = {}
    for i = 1, n do
        local name = SCRIPTS[i]
        package.loaded[name] = nil      -- force a recompile from disk...
        fresh[i] = require(name).main   -- ...into the fresh set (raises on error)
    end
    for i = 1, n do
        mains[i] = fresh[i]             -- all loaded: commit in one go
    end
end

load_scripts()

-- run(ptr): process one message. `ptr` is a lightuserdata World*; cast turns it
-- into a typed cdata registry from which each script reaches the objects it
-- needs. This is the hot path the JIT compiles after warm-up -- no require, no
-- allocation, no I/O.
local function run(ptr)
    local world = cast('World*', ptr)
    for i = 1, n do
        mains[i](world)
    end
end

return { run = run, reload = load_scripts }
