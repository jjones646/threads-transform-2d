#pragma once

/*
 * Class declaration for Complex number class
 */

#include <iostream>
#include <string>

class Complex {
public:
  //Constructors;
  Complex();
  Complex(double r, double i); // Real and imaginary parts
  Complex(double r);           // Real part only, imag = 0.
  // Overloaded operators
  Complex operator+ (const Complex& b) const;
  Complex operator- (const Complex& b) const;
  Complex operator* (const Complex& b) const;
  bool    operator< (const Complex& b) const;
  bool    operator< (double mag)       const;
  bool    operator==(const Complex& b) const;
  // Required member functions
  Complex Mag()   const; //Returns the magnitude of the complex number
  Complex Angle() const; //returns the angle of the complex number
  Complex Conj()  const; //returns the complex conjugate of the Complex number
  void    Print() const; // Print the complex value
  double  get_mag() const;
  // Data Members
private:
  double real;
  double imag;

  friend std::ostream& operator<< (std::ostream &os, const Complex& c);
};

// Global function to output a Complex value
std::ostream& operator << (std::ostream &os, const Complex& c);
