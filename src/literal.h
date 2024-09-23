/*
MIT License

Copyright (c) 2024 Anonymized

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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