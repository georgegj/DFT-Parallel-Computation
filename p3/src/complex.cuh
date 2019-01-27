//
// Created by brian on 11/20/18.
//

#pragma once

#include <iostream>

class Complex {
public:
    __device__ __host__ Complex();
    __device__ __host__ Complex(float r, float i);
    __device__ __host__ Complex(float r);
    __device__ __host__ Complex operator+(const Complex& b) const;
    __device__ __host__ Complex operator-(const Complex& b) const;
    __device__ __host__ Complex operator*(const Complex& b) const;

    __device__ __host__ Complex mag() const;
    __device__ __host__ Complex angle() const;
    __device__ __host__ Complex conj() const;

    float real;
    float imag;
};

 std::ostream& operator<<(std::ostream& os, const Complex& rhs);

