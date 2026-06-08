#include "Inventory.h"

#include <sstream>

void Inventory::add(const std::string& name, int qty) {
    int next = items_[name] + qty;
    if (next <= 0) {
        items_.erase(name);
    } else {
        items_[name] = next;
    }
}

int Inventory::remove(const std::string& name, int qty) {
    if (qty <= 0) {
        return 0;
    }
    auto it = items_.find(name);
    if (it == items_.end()) {
        return 0;
    }
    int removed = (it->second < qty) ? it->second : qty;
    it->second -= removed;
    if (it->second <= 0) {
        items_.erase(it);
    }
    return removed;
}

int Inventory::get(const std::string& name) const {
    auto it = items_.find(name);
    return (it == items_.end()) ? 0 : it->second;
}

int Inventory::total() const {
    int sum = 0;
    for (const auto& kv : items_) {
        sum += kv.second;
    }
    return sum;
}

std::size_t Inventory::size() const {
    return items_.size();
}

std::vector<std::string> Inventory::keys() const {
    std::vector<std::string> result;
    result.reserve(items_.size());
    for (const auto& kv : items_) {
        result.push_back(kv.first);
    }
    return result;
}

std::string Inventory::toString() const {
    std::ostringstream oss;
    oss << "Inventory{";
    bool first = true;
    for (const auto& kv : items_) {
        if (!first) {
            oss << ", ";
        }
        oss << kv.first << "=" << kv.second;
        first = false;
    }
    oss << "}";
    return oss.str();
}
