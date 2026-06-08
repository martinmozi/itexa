-- A second hot-reloadable processing script. It mutates a different inventory
-- and shows cross-object logic: read one object (an Account) to decide what to
-- write into another (an Inventory). Work only -- no console I/O, no error
-- raising on the hot path.
require('host_ffi') -- installs the metatypes (cached after first load)

local M = {}

function M.main(world)
    local inv2 = world:inventory(2)
    if inv2 then
        inv2:add("sword", 1)
    end

    -- Read account 1, mirror its balance into inventory 1 as a "coins" entry.
    local acc = world:account(1)
    local inv1 = world:inventory(1)
    if acc and inv1 then
        local coins = inv1:get("coins")
        inv1:add("coins", acc:balance() - coins) -- set coins := current balance
    end

    -- Honest mistake: an unknown id returns nil and is handled gracefully,
    -- instead of dereferencing a bad handle and crashing the host.
    local missing = world:inventory(999)
    if missing then
        missing:add("ghost", 1)
    end
end

return M
