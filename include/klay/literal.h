#pragma once

#include <iostream>
using namespace std;

typedef int Var;

/**
 * A Literal
 *
 * The internal representation uses (var << 1) + sign;
 * There is an implicit constructor, but it is not advised.
 * Instead, we recommend to use either of the following
 * instantiation methods (per example):
 * ```
 * Lit ex1 = Lit(5, false) // 5
 * Lit ex2 = Lit(5, true) // -5
 *
 * int i = -5;
 * Lit ex3 = Lit::fromInt(i); // -5
 * assert i == ex3.toInt();
 *
 * Lit(Lit(-5)); // -5
 * ```
 */
class Lit {

public:
    int m_val;

    /**
     * Create the corresponding Literal. Sign true indicates negative literal.
     */
    Lit(Var v, bool sign) : m_val((v << 1) + sign) {}

    /**
     * Create a copy of the given literal.
     */
    Lit(const Lit& l) : m_val(l.m_val) {}

    /**
     * Do not use! It expects as input the internal representation val.
     * Created for implicit conversions.
     */
    Lit(int val) : m_val(val) {}



    // static inline Lit fromInt(int i) { return Lit(std::abs(i), i < 0); }
    static inline Lit fromInt(int i);
    inline int toInt() const { return sign() ? -var() : var(); }

    inline Var var() const { return m_val >> 1; }
    inline bool sign() const { return m_val & 0x01; }
    inline int internal_val() const { return m_val; }

    inline Lit negation() const { return {m_val ^ 0x01};  }
    Lit operator~() { return negation(); }
    bool operator==(Lit p) const { return m_val == p.m_val; }
    bool operator!=(Lit p) const { return m_val != p.m_val; };

};

// std::ostream& operator<<(std::ostream &os, const Lit& l) { os << l.toInt(); return os; };

Lit Lit::fromInt(int i) {
    return Lit(std::abs(i), i < 0);
}