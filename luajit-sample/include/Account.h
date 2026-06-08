#pragma once

#include <string>

// A second host object type, exposed to Lua alongside Inventory to show that the
// registry pattern hands out more than one kind of object. Same rule applies:
// scripts may only change it through its public methods.
class Account {
public:
    // Add `amount` to the balance (ignored if <= 0).
    void deposit(int amount);

    // Remove up to `amount` from the balance. Returns the amount withdrawn.
    int withdraw(int amount);

    // Current balance.
    int balance() const;

    // Human readable representation, e.g. "Account{balance=70}".
    std::string toString() const;

private:
    int balance_ = 0;
};
