#pragma once

#include <map>

#include "Account.h"
#include "Inventory.h"

// The registry / context object handed to scripts. It owns the other host
// objects and is the single entry point a script receives: from a World handle a
// script reaches any Inventory or Account by id. Using std::map keeps the stored
// objects at stable addresses, so handles returned to Lua stay valid for the
// object's lifetime.
//
// Accessors return nullptr for an unknown id; the FFI layer turns that into a
// Lua nil so scripts can check before use. (For a production system where
// objects come and go, hand out validated integer handles with a generation
// counter instead of raw pointers -- this prototype owns everything for the
// whole run, so raw pointers are safe here.)
class World {
public:
    Inventory* inventory(int id);
    Account*   account(int id);

    // Create-or-get helpers used by the host to populate the world up front.
    Inventory& createInventory(int id);
    Account&   createAccount(int id);

private:
    std::map<int, Inventory> inventories_;
    std::map<int, Account>   accounts_;
};
