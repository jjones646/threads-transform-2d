/*
 * Complex Number Class implementation
 * Jonathan Jones
 */

#include <iostream>
#include <string>

#include <math.h>

#include "Complex.h"

using namespace std;

// Constructors
Complex::Complex()
  : real(0), imag(0)
{
}

Complex::Complex(double r)
  : real(r), imag(0)
{
}

Complex::Complex(double r, double i)
  : real(r), imag(i)
{
}

// Operators
Complex Complex::operator+ (const Complex& b) const
{
  return Complex(real + b.real, imag + b.imag);
}

Complex Complex::operator- (const Complex& b) const
{
  return Complex(real - b.real, imag - b.imag);
}

Complex Complex::operator* (const Complex& b) const
{
  return Complex(real * b.real - imag * b.imag,
                 real * b.imag + imag * b.real);
}

bool Complex::operator< (const Complex& b) const
{
  return get_mag() < b.get_mag();
}

bool Complex::operator< (double mag) const
{
  return get_mag() < mag;
}

bool Complex::operator== (const Complex& b) const
{
  return (real == b.real) && (imag == b.imag);
}

// Member functions
double Complex::get_mag() const
{
  return sqrt(real * real + imag * imag);
}

Complex Complex::Mag() const
{
  return Complex(sqrt(real * real + imag * imag));
}

Complex Complex::Angle() const
{
  return Complex(atan2(imag, real) * 360 / (2 * M_PI));
}

Complex Complex::Conj() const
{ // Return to complex conjugate
  return Complex(real, -imag);
}

void Complex::Print() const
{
  if (imag == 0)
  { // just real part with no parens
    cout << real;
  }
  else
  {
    cout << '(' << real << "," << imag << ')';
  }
}

// Global function to output a Complex value
std::ostream& operator<< (std::ostream &os, const Complex& c)
{
  if (c.imag == 0)
  { // just real part with no parens
    os << c.real;
  }
  else
  {
    os << '(' << c.real << "," << c.imag << ')';
  }
  return os;
}
