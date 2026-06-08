-- Receives the same shared Inventory* object and keeps mutating it through the
-- methods installed by inv_ffi. Work only -- no console I/O on the hot path.
require('inv_ffi') -- installs the Inventory metatype (cached after first load)

local M = {}

function M.main(obj)
  obj:add("sword", 1)
  obj:add("apple", 2)
end

return M
