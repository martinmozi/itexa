-- `obj` is an Inventory* cdata that behaves like a complete Lua object: every
-- method (add/remove/get/total/...) is installed by inv_ffi via ffi.metatype and
-- runs the matching C++ method internally. The raw C functions are private to
-- inv_ffi, so this script never touches the FFI namespace directly.
require('inv_ffi') -- installs the Inventory metatype (cached after first load)

local M = {}

-- On the timed hot path, so it performs work only -- no I/O.
function M.main(obj)
  obj:add("apple", 3)
  obj:add("gold", 50)
  obj:remove("gold", 20)

  -- Direct field assignment is rejected by the object's __newindex.
  pcall(function() obj.apple = 999 end)
end

return M
