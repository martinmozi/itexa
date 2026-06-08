# LuaJIT C++ host: modifikovanie C++ objektov zo skriptov cez FFI objekt

C++ host vkladá (embeduje) LuaJIT a umožňuje **dôveryhodným** Lua skriptom čítať
a modifikovať C++ objekty, ale výhradne cez verejné metódy daného objektu.
Objekty sa nikdy nevystavujú ako surové tabuľky či userdata: skripty vidia
typované FFI `cdata`, ktoré sa *správa* ako bežný Lua objekt (`obj:metoda(...)`,
`tostring(obj)`, `#obj`), a každé volanie metódy sa JIT-skompiluje priamo do
príslušnej C++ funkcie.

Hot path (spracovanie jednej správy) sa zostaví raz a spustí mnohokrát, takže
JIT sa naplno zahreje ešte predtým, než sa meria.

## Ako to funguje

### Vrstvy

```
C++ objekty            Inventory, Account            (reálny stav, verejné metódy)
      |  vlastní
World (registry)        World                         (vydáva objekty podľa id)
      |  vystavené cez
C ABI shimy            src/HostFFI.cpp                world_*, inv_*, acc_*
      |  deklarované + zabalené v
FFI objekt modul       scripts/host_ffi.lua           ffi.cdef + ffi.metatype
      |  používané v
Spracovacie skripty    scripts/mod_a.lua, mod_b.lua   main(world)
```

1. **C++ objekty** (`Inventory`, `Account`) držia reálny stav a vystavujú obyčajné
   verejné metódy. O Lue nevedia nič.
2. **`World`** je registry / context objekt. Vlastní ostatné objekty a vyhľadáva
   ich podľa id. Je to jediný handle, ktorý skript dostane.
3. **`src/HostFFI.cpp`** exportuje čisté C ABI (`world_inventory`, `inv_add`,
   `acc_deposit`, ...). Každá funkcia je tenký shim, ktorý preposiela volanie do
   C++ metódy. `ENABLE_EXPORTS` v CMake zariadi, že tieto symboly sú z executable
   dostupné cez LuaJIT `ffi.C`.
4. **`scripts/host_ffi.lua`** deklaruje C prototypy cez `ffi.cdef` a každý opaque
   struct naviaže na metatabuľku cez `ffi.metatype`. Keďže prijímateľ (receiver)
   je prvý parameter každej C funkcie, **C funkcia priamo JE metóda** — netreba
   žiadny obalový Lua closure (`obj:add(n, q)` -> `inv_add(obj, n, q)`).
5. **`scripts/mod_a.lua` / `mod_b.lua`** sú hot-reloadovateľné skripty. Každý
   dostane `world` handle a siahne na objekty, ktoré potrebuje.

### Registry pattern (prístup k viacerým objektom)

Skript dostane jeden `World` handle a z neho si vytiahne typované handle:

```lua
local inv = world:inventory(1)     -- Inventory*  (nil ak také id neexistuje)
local acc = world:account(1)       -- Account*    (nil ak také id neexistuje)
if inv and acc then
    inv:add("coins", acc:balance())
end
```

Vrátený handle je už plnohodnotný objekt (má nainštalovaný svoj metatype), takže
`world:inventory(1):add("gold", 5)` funguje priamo. Pridanie nového typu objektu
je prevažne mechanické — viď **Rozširovanie** nižšie.

### Bootstrap: zostav raz, spúšťaj mnohokrát (`scripts/boot.lua`)

C++ host načíta `scripts/boot.lua` raz cez `luaL_loadfile`, odovzdá debug flag a
dostane späť runner tabuľku `{ run, reload }`. Všetka jednorazová práca —
`package.path`, `require`, `ffi.cdef`, vyriešenie `main` každého skriptu — sa deje
v `boot.lua`. Vrátený `run(world_ptr)` je per-message hot path a nič z toho už
nerobí: pretypuje pointer na `World*` a zavolá `main` každého skriptu.

`main.cpp` si `run` aj `reload` vytiahne do vlastných registry referencií, takže
hot loop nikdy nerobí lookup do tabuľky.

Debug flag tečie z `main.cpp` do `boot.lua` ako jeho jediný argument: keď je
nastavený, `boot.lua` vypne JIT (`jit.off()`) a spustí EmmyLua debugger *predtým*,
než sa rozbehne ktorýkoľvek skript, takže breakpointy v hot-reloadovateľných
skriptoch zaberú hneď pri prvom volaní. Viď **Debugovanie** nižšie.

### Hot reload (transakčný)

`reload` zahodí zacachované chunky skriptov (`package.loaded[name] = nil`) a znova
ich `require`-ne, takže editovaný `mod_a.lua` / `mod_b.lua` sa prejaví bez
reštartu. V produkcii by si ho zavolal, keď file watcher nahlási zmenu.
(`host_ffi` ani C ABI sa **nereloaduje** — len spracovacie skripty.)

Reload je **transakčný — všetko alebo nič**: všetky skripty sa najprv skompilujú
do čerstvej sady a do `mains` sa prehodia až vtedy, keď **všetky** uspejú. Ak
ktorýkoľvek zlyhá (napr. syntax chyba v editovanom súbore), `mains` ostane
nedotknuté — bežiaca sada ďalej funguje na starých verziách — a chyba sa
propaguje volajúcemu (host ju ohlási cez svoj `pcall`). Jeden zlý edit teda
nezhodí bežiacu sadu.

> Pozn.: transakčný reload rieši **atomicitu výmeny**, nie súbežnosť. Jeden
> `lua_State` nie je thread-safe — ak budeš správy spracúvať paralelne, potrebuješ
> aj **mutex** (read pri `run`, write pri `reload`), alebo radšej `lua_State` na
> worker (hot path bez zámku, reload koordinovaný cez všetky states).

### Zahriatie JIT a meranie

LuaJIT začne trasovať a kompilovať funkciu/slučku až keď prekročí svoj hotcount
prah (default ~56), takže pár behov by stále bežalo v interpreteri. `main.cpp`
preto:

- odmeria jedno **studené** (cold) volanie (interpretované),
- spustí `WARMUP_RUNS` volaní bez merania, aby naštartoval kompiláciu trás,
- odmeria dávku `MEASURE_RUNS` volaní a vypíše **priemer na volanie**.

Jedno volanie `run()` == spracovanie jednej správy v reálnom systéme.

### Bezpečnostný model (dôveryhodné skripty, no žiadne pády z poctivej chyby)

Skripty sú dôveryhodné (FFI dáva plný natívny prístup — skript, ktorý sa dostane
k `ffi`, ber ako ekvivalent natívneho kódu; toto **nie je** sandbox pre
nedôveryhodný kód). Cieľom je, aby poctivý preklep nezhodil host:

- **C++ shimy sú defenzívne.** Každý shim v `HostFFI.cpp` null-checkne svoje
  pointre (aj string argumenty — FFI premení `nil` na `NULL`) a obalí telo do
  `try/catch`, takže žiadny zlý dereferencing ani žiadna C++ výnimka nikdy
  neprejde cez FFI hranicu (to druhé je v LuaJIT undefined behavior). Pri zásahu
  guardu sa volanie zvrhne na bezpečný default (no-op / 0 / "").
- **Neznáme id vráti `nil`.** `world:inventory(id)` vráti Lua `nil` (nie `NULL`
  cdata, ktoré je truthy), takže skript vie pred použitím spraviť `if inv then`.
- **Objekty sú read-only okrem cez metódy.** `__newindex` odmietne priamy zápis
  do poľa (`inv.apple = 1`) jasnou, typovo špecifickou chybou.
- **Host chráni každé volanie** cez `lua_pcall`, takže Lua-level chyba (zlá
  konverzia argumentu, indexovanie nil) sa zaloguje, nie je fatálna.

Poznámka k životnosti: v tomto prototype `World` vlastní každý objekt počas celého
behu, takže surové pointre odovzdané do Lua ostávajú platné. V produkčnom systéme,
kde objekty vznikajú a zanikajú, vydávaj radšej **validované celočíselné handle s
generation counterom** namiesto surových pointrov, aby sa zastaraný handle
detegoval a nie dereferencoval.

## Súbory

| Súbor | Úloha |
| --- | --- |
| `src/main.cpp` | C++ host: načíta `boot.lua`, naplní `World`, warm-up + meranie, demo reloadu |
| `include/Inventory.h`, `src/Inventory.cpp` | Prvý typ C++ objektu |
| `include/Account.h`, `src/Account.cpp` | Druhý typ C++ objektu |
| `include/World.h`, `src/World.cpp` | Registry, ktorý vlastní objekty a vyhľadáva ich podľa id |
| `include/HostFFI.h`, `src/HostFFI.cpp` | Exportované C ABI shimy (`world_*`, `inv_*`, `acc_*`) s guardmi |
| `scripts/boot.lua` | Jednorazový bootstrap; vracia `{ run, reload }` |
| `scripts/host_ffi.lua` | `ffi.cdef` + `ffi.metatype` pre každý typ objektu |
| `scripts/mod_a.lua`, `scripts/mod_b.lua` | Hot-reloadovateľné spracovacie skripty |
| `CMakeLists.txt` | Build; kopíruje `scripts/` vedľa executable |

## Build

Ak LuaJIT nie je globálne nájditeľný, zadaj `LUAJIT_ROOT`.

```powershell
cmake -S . -B build -DLUAJIT_ROOT="C:/cesta/k/luajit"
cmake --build build --config Release
```

(Priložený `CMakeSettings.json` umožní Visual Studiu konfigurovať/buildovať priamo
so správnym MSVC prostredím.)

## Spustenie

```powershell
./build/Release/luajit_embed.exe
```

Očakávaný výstup (čísla závisia od počtu behov):

```
inventory 1: Inventory{apple=..., coins=..., gold=...}
inventory 2: Inventory{sword=...}
account 1:   Account{balance=...}
reload: ok
cold call (interpreted): 0.0x ms
warm avg over 1000000 calls: <ns> ns/call (<ms> ms total)
```

Porovnaj teplé `ns/call` (po JIT-e) so studeným volaním a uvidíš efekt JIT-u.

## Debugovanie z VS Code (LuaPanda)

Debug režim vypne JIT (`jit.off`, aby debugger vedel krokovať riadok po riadku)
a načíta čisto-Lua debugger **LuaPanda** (`LuaPanda.lua`). Na rozdiel od EmmyLuy
**nepočúva** a nečaká na IDE — naopak sa **pripojí** k debug serveru bežiacemu vo
VS Code a `BP()` hneď zastaví beh na breakpointe.

> Poradie je preto opačné ako pri EmmyLue: najprv spusti počúvanie vo VS Code
> (**F5**), až potom aplikáciu s `--debug`.

Kroky:
`
1. Nainštaluj rozšírenie **LuaPanda** vo VS Code (vyhľadaj „LuaPanda" v Extensions;
   projekt [Tencent/LuaPanda](https://github.com/Tencent/LuaPanda)).
2. Stiahni `LuaPanda.lua` z
   [LuaPanda repozitára](https://github.com/Tencent/LuaPanda) (súbor
   `Debugger/LuaPanda.lua`) a polož ho do priečinka `scripts/`. Žiadna zmena v
   `CMakeLists.txt` netreba — `copy_directory` celý `scripts/` skopíruje vedľa
   executable, takže ho `require` nájde cez `package.path`.
3. Vo VS Code priprav LuaPanda konfiguráciu v `.vscode/launch.json` a stlač **F5**
   (VS Code začne počúvať na porte 8818):

   ```json
   {
       "version": "0.2.0",
       "configurations": [
           {
               "type": "lua",
               "request": "launch",
               "name": "LuaPanda",
               "cwd": "${workspaceFolder}",
               "connectionPort": 8818,
               "stopOnEntry": true
           }
       ]
   }
   ```

4. Spusti aplikáciu v debug režime — pripojí sa k VS Code a zastaví na `BP()`:

   ```powershell
   ./build/Release/luajit_embed.exe --debug
   ```

5. Daj breakpointy do `scripts/mod_a.lua` / `mod_b.lua` a krokuj.

Voliteľné prepísanie hostiteľa/portu pred spustením (port musí sedieť s
`connectionPort` v `launch.json`):

```powershell
$env:LUAPANDA_HOST = "localhost"
$env:LUAPANDA_PORT = "8818"
```

Ak sa `LuaPanda` nenájde, aplikácia vypíše oznam a beží normálne. Ak nechceš
commitovať súbor tretej strany, pridaj `scripts/LuaPanda.lua` do `.gitignore`.

## Rozširovanie: pridanie nového typu C++ objektu

Pre vystavenie nového typu `Foo`, na ktorý skripty siahnu cez registry:

1. **C++ objekt** — pridaj `include/Foo.h` / `src/Foo.cpp` s verejnými metódami.
2. **C ABI shimy** — do `HostFFI.h` / `HostFFI.cpp` pridaj funkcie `foo_*`
   preposielajúce do metód `Foo`, každú s null-check + `try/catch` guardmi. Pridaj
   `World` accessor `world_foo(World*, int id)` vracajúci `Foo*`.
3. **Registry** — pridaj `std::map<int, Foo>` do `World` plus `foo(int id)` a
   `createFoo(int id)`.
4. **FFI modul** — v `host_ffi.lua` pridaj prototypy `Foo` do `ffi.cdef`, väzbu
   `ffi.metatype('Foo', ...)` s jeho metódami a accessor `world:foo(id)`, ktorý
   mapuje `NULL` na `nil`.
5. **CMake** — pridaj `src/Foo.cpp` do zoznamu `add_executable`.
6. **Skripty** — `world:foo(id)` je odteraz dostupné v `mod_*.lua`.

Signatúra každej metódy sa musí zhodovať na troch miestach (C++ metóda, C shim,
`ffi.cdef`); drž ich zladené. Pri mnohých typoch zváž makro na generovanie shimov
alebo jeden spoločný bindings súbor.
