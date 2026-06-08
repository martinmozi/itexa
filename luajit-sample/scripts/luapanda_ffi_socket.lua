-- luapanda_ffi_socket.lua
-- Nahrada luasocket pre LuaJIT cez FFI (TCP klient).
-- Platformy: Windows (ws2_32), Linux, macOS.
-- Pouzitie: PRED require("LuaPanda") spusti:  require("luapanda_ffi_socket")
local ffi = require("ffi")
local WIN = (ffi.os == "Windows")

local lib
local AF_INET, SOCK_STREAM = 2, 1

if WIN then
  ffi.cdef[[
    typedef uintptr_t SOCKET;
    typedef struct { uint16_t family; uint16_t port; uint32_t addr; char zero[8]; } sockaddr_in;
    typedef struct { unsigned int fd_count; SOCKET fd_array[64]; } fd_set;
    struct timeval { long tv_sec; long tv_usec; };
    int WSAStartup(uint16_t v, void *data);
    SOCKET socket(int af, int type, int protocol);
    int connect(SOCKET s, const sockaddr_in *name, int namelen);
    int send(SOCKET s, const char *buf, int len, int flags);
    int recv(SOCKET s, char *buf, int len, int flags);
    int closesocket(SOCKET s);
    int select(int nfds, fd_set *r, fd_set *w, fd_set *e, struct timeval *t);
    unsigned short htons(unsigned short x);
    unsigned int inet_addr(const char *cp);
  ]]
  lib = ffi.load("ws2_32")
  lib.WSAStartup(0x0202, ffi.new("char[512]"))   -- MAKEWORD(2,2)
else
  ffi.cdef[[
    typedef struct { uint16_t family; uint16_t port; uint32_t addr; char zero[8]; } sockaddr_in;
    int socket(int domain, int type, int protocol);
    int connect(int fd, const sockaddr_in *addr, unsigned int len);
    long send(int fd, const void *buf, unsigned long n, int flags);
    long recv(int fd, void *buf, unsigned long n, int flags);
    int close(int fd);
    unsigned short htons(unsigned short x);
    unsigned int inet_addr(const char *cp);
  ]]
  lib = ffi.C
end

local MSG_PEEK     = 0x02
local MSG_DONTWAIT = (ffi.os == "OSX") and 0x80 or 0x40
local EAGAIN       = (ffi.os == "OSX") and 35   or 11
local closefd      = WIN and lib.closesocket or lib.close

local Sock = {}
Sock.__index = Sock

function Sock:settimeout(t) self.timeout = t end

function Sock:connect(host, port)
  if host == "localhost" then host = "127.0.0.1" end
  local sa = ffi.new("sockaddr_in")
  sa.family = AF_INET
  sa.port   = lib.htons(port)
  sa.addr   = lib.inet_addr(host)
  if lib.connect(self.fd, sa, ffi.sizeof("sockaddr_in")) ~= 0 then
    return nil, "connection refused"
  end
  return 1
end

function Sock:send(data)
  local n = lib.send(self.fd, data, #data, 0)
  if n < 0 then return nil, "closed" end
  return tonumber(n)
end

-- neblokujuca kontrola, ci su data pripravene na citanie
local function pending(self)
  if WIN then
    local fds = ffi.new("fd_set")
    fds.fd_count = 1
    fds.fd_array[0] = self.fd
    local tv = ffi.new("struct timeval", 0, 0)
    return lib.select(0, fds, nil, nil, tv) > 0     -- >0 = data/koniec, 0 = ziadne
  else
    local b = ffi.new("uint8_t[1]")
    local r = lib.recv(self.fd, b, 1, MSG_PEEK + MSG_DONTWAIT)
    if r > 0 then return true end
    if r < 0 and ffi.errno() == EAGAIN then return false end
    return nil                                       -- 0 alebo ina chyba = zatvorene
  end
end

local function readn(self, n)
  local buf, out, got = ffi.new("uint8_t[?]", n), {}, 0
  while got < n do
    local r = lib.recv(self.fd, buf, n - got, 0)
    if r <= 0 then return nil end
    out[#out + 1] = ffi.string(buf, r)
    got = got + r
  end
  return table.concat(out)
end

function Sock:receive(pattern)
  pattern = pattern or "*l"
  if self.timeout == 0 then                          -- neblokujuci poll (debug hook)
    local p = pending(self)
    if p == nil then return nil, "closed", "" end
    if p == false then return nil, "timeout", "" end
  end
  if type(pattern) == "number" then
    local d = readn(self, pattern)
    if not d then return nil, "closed", "" end
    return d
  end
  -- "*l": riadok ukonceny \n (LuaPanda posiela cele spravy naraz)
  local out, b = {}, ffi.new("uint8_t[1]")
  while true do
    local r = lib.recv(self.fd, b, 1, 0)
    if r <= 0 then return nil, "closed", table.concat(out) end
    local c = b[0]
    if c == 10 then return table.concat(out)
    elseif c ~= 13 then out[#out + 1] = string.char(c) end
  end
end

function Sock:close() closefd(self.fd) end

local function tcp()
  return setmetatable({ fd = lib.socket(AF_INET, SOCK_STREAM, 0), timeout = nil }, Sock)
end

local mod = { tcp = tcp }
package.loaded["socket.core"] = mod                  -- LuaPanda hlada toto ako prve
package.loaded["socket"]      = mod
return mod
