#include "Account.h"

#include <sstream>

void Account::deposit(int amount) {
    if (amount > 0) {
        balance_ += amount;
    }
}

int Account::withdraw(int amount) {
    if (amount <= 0) {
        return 0;
    }
    int taken = (amount < balance_) ? amount : balance_;
    balance_ -= taken;
    return taken;
}

int Account::balance() const {
    return balance_;
}

std::string Account::toString() const {
    std::ostringstream oss;
    oss << "Account{balance=" << balance_ << "}";
    return oss.str();
}
