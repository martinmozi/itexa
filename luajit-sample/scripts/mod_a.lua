-- A hot-reloadable processing script. It receives the `world` registry and
-- reaches the objects it needs by id; every method (add/remove/deposit/...) is
-- installed by host_ffi via ffi.metatype and runs the matching C++ method
-- internally. The raw C functions are private to host_ffi, so this script never
-- touches the FFI namespace directly.
--
-- In the real system this is invoked once per incoming ISO message to mutate the
-- target objects. It must stay on the hot path: work only, no I/O, no error
-- raising. Missing ids come back as nil, so guard before use -- a bad id must
-- not crash the host. (Direct field writes like `inv.apple = 1` are rejected by
-- the object's __newindex.)
require('host_ffi') -- installs the metatypes (cached after first load)

local M = {}

function M.main(world)
    local inv = world:inventory(1)
    if not inv then
        return "inventory 1 missing"        -- error = abort; nil/nič = úspech
    end

    inv:add("apple", 3)
    inv:add("gold", 50)
    local removed = inv:remove("gold", 20)
    if removed < 20 then
        return "inventory 1: not enough gold"
    end

    -- happy path: prepadne na koniec => return nil => úspech
end

return M
