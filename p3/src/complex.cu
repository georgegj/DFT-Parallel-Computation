//
// Created by brian on 11/20/18.
//

#include "complex.cuh"

#include <cmath>

const float PI = 3.14159265358979f;

__device__ __host__ Complex::Complex() : real(0.0f), imag(0.0f) {}

__device__ __host__ Complex::Complex(float r) : real(r), imag(0.0f) {}

__device__ __host__ Complex::Complex(float r, float i) : real(r), imag(i) {}

__device__ __host__ Complex Complex::operator+(const Complex &b) const {
Complex a;
a.real = b.real + this->real;
a.imag = b.imag + this->imag;
return a;
}

__device__ __host__ Complex Complex::operator-(const Complex &b) const {
Complex a;
a.real = this->real - b.real  ;
a.imag = this->imag - b.imag  ;
return a;
}

__device__ __host__ Complex Complex::operator*(const Complex &b) const {
Complex a;
a.real = b.real * this->real - b.imag*this->imag;
a.imag = b.imag * this->real + b.real*this->imag;
return a;
}

__device__ __host__ Complex Complex::mag() const {
return Complex(sqrt(real*real + imag*imag));
}

__device__ __host__ Complex Complex::angle() const {
return Complex(atan2(imag,real)*360/(2*M_PI));
}

__device__ __host__ Complex Complex::conj() const {
return Complex(real,-imag);
}

 std::ostream& operator<< (std::ostream& os, const Complex& rhs) {
    Complex c(rhs);
    if(fabsf(rhs.imag) < 1e-10) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-10) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}
