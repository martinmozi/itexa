#pragma once

#include <map>
#include <string>
#include <vector>

// A small C++ class with internal state (a std::map). Lua scripts will modify
// instances of this class, but only through its public interface, which is
// exposed to Lua via a metatable.
class Inventory {
public:
    // Add `qty` units of `name` (qty may be negative to subtract).
    void add(const std::string& name, int qty);

    // Remove up to `qty` units of `name`. Returns the amount actually removed.
    int remove(const std::string& name, int qty);

    // Current quantity for `name` (0 if absent).
    int get(const std::string& name) const;

    // Sum of all quantities.
    int total() const;

    // Number of distinct item names currently stored.
    std::size_t size() const;

    // All item names currently stored (sorted, since std::map is ordered).
    std::vector<std::string> keys() const;

    // Human readable representation, e.g. "Inventory{apple=3, gold=10}".
    std::string toString() const;

private:
    std::map<std::string, int> items_;
};
