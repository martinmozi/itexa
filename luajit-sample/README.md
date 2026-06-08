# LuaJIT C++ CMake Minimal Example

This example demonstrates:

- Building a **main Lua script as a C++ string**.
- Optional debug path that simulates debugger loading with a print.
- Disabling LuaJIT in debug mode via `jit.off()`.
- Loading multiple module scripts (`mod_a`, `mod_b`) each with `main()`.
- Calling each module's `main()`.
- Running the main script from C++ using `luaL_loadstring` and `lua_pcall`.

## Files

- `src/main.cpp` - C++ host program
- `scripts/mod_a.lua` - Lua module A
- `scripts/mod_b.lua` - Lua module B
- `CMakeLists.txt` - Build setup

## Build

If LuaJIT is not discoverable globally, pass `LUAJIT_ROOT`.

```powershell
cmake -S . -B build -DLUAJIT_ROOT="C:/LuaJIT"
cmake --build build --config Release
```

## Run

Normal run:

```powershell
./build/Release/luajit_embed.exe
```

Debug mode (simulated debugger load + `jit.off()`):

```powershell
./build/Release/luajit_embed.exe --debug
```

Expected output includes module main calls and debug prints in debug mode.

## Debugging from VS Code (EmmyLua, listens on a port)

The `--debug` path loads the native **EmmyLua** debugger (`emmy_core`), which
opens a TCP socket and waits for the IDE to attach. VS Code then connects to
that port.

Steps:

1. Install the **EmmyLua** extension in VS Code (publisher: `tangzx`).
2. Download the matching `emmy_core` native module from
   [EmmyLuaDebugger releases](https://github.com/EmmyLua/EmmyLuaDebugger/releases)
   (use the LuaJIT / Lua 5.1 build for your platform/arch) and place it next to
   the built executable:
   - Windows: `emmy_core.dll`
   - Linux: `emmy_core.so`
3. Build and start the app in debug mode (it will block on `waitIDE()`):

   ```powershell
   ./build/Release/luajit_embed.exe --debug
   ```

4. In VS Code press **F5** and pick **"Attach to LuaJIT (EmmyLua :9966)"**
   (see `.vscode/launch.json`). Set breakpoints in `scripts/mod_a.lua` /
   `scripts/mod_b.lua` and step through.

Optional overrides via environment variables before launching the app:

```powershell
$env:EMMY_HOST = "localhost"
$env:EMMY_PORT = "9966"
```

If `emmy_core` is not found, the app prints a notice and runs normally without
a debugger.

