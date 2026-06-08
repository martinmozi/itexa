#include "World.h"

Inventory* World::inventory(int id) {
    auto it = inventories_.find(id);
    return (it == inventories_.end()) ? nullptr : &it->second;
}

Account* World::account(int id) {
    auto it = accounts_.find(id);
    return (it == accounts_.end()) ? nullptr : &it->second;
}

Inventory& World::createInventory(int id) {
    return inventories_[id];
}

Account& World::createAccount(int id) {
    return accounts_[id];
}
