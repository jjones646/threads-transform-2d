// Threaded two-dimensional Discrete FFT transform
// Jonathan Jones
// ECE8893 Project 2

#include <iostream>
#include <string>
#include <math.h>
#include <cstdlib>

#include "Complex.h"
#include "InputImage.h"

using namespace std;

// Global variables visible to all threads
pthread_mutex_t   startCountMutex;
pthread_mutex_t   exitMutex;
pthread_mutex_t   elementCountMutex;
pthread_cond_t    exitCond;
int               startCount;
// Global variables for the image
Complex*          ImageData;
int               ImageWidth;
int               ImageHeight;
unsigned          N;

// Function to reverse bits in an unsigned integer
// This assumes there is a global variable N that is the
// number of points in the 1D transform.
unsigned ReverseBits(unsigned v)
{ //  Provided to students
  unsigned n = N; // Size of array (which is even 2 power k value)
  unsigned r = 0; // Return value

  for (--n; n > 0; n >>= 1)
  {
    r <<= 1;        // Shift return value
    r |= (v & 0x1); // Merge in next bit
    v >>= 1;        // Shift reversal value
  }
  return r;
}

// Call MyBarrier_Init once in main
void MyBarrier_Init()// you will likely need some parameters)
{

}

// Each thread calls MyBarrier after completing the row-wise DFT
void MyBarrier() // Again likely need parameters
{

}

void Transform1D(Complex* h, int N)
{
  // Implement the efficient Danielson-Lanczos DFT here.
  // "h" is an input/output parameter
  // "N" is the size of the array (assume even power of 2)
}

void* Transform2DThread(void* const arg)
{
  // we are passed our thread number
  unsigned long threadID = (unsigned long)arg;

  // Calculate 1d DFT for assigned rows
  // wait for all to complete
  // Calculate 1d DFT for assigned columns
  // Decrement active count and signal main if all complete

  // -- mutex example --
  // pthread_mutex_lock(&elementCountMutex);
  // elementCount += localCount;
  // pthread_mutex_unlock(&elementCountMutex);

  pthread_mutex_lock(&startCountMutex);
  startCount--;
  if (startCount == 0) {
    // Last to exit, notify main
    pthread_mutex_unlock(&startCountMutex);
    pthread_mutex_lock(&exitMutex);
    pthread_cond_signal(&exitCond);
    pthread_mutex_unlock(&exitMutex);
  }
  else {
    pthread_mutex_unlock(&startCountMutex);
  }

  return 0;
}

void Transform2D(const char* inputFN, int nThreads)
{
  InputImage image(inputFN);  // Create the helper object for reading the image

  // Create the global pointer to the image array data
// same image data array as well as width/height
  ImageData   = image.GetImageData();
  ImageWidth  = image.GetWidth();
  ImageHeight = image.GetHeight();

  // All mutex and condition variables must be "initialized"
  pthread_mutex_init(&exitMutex, 0);
  pthread_mutex_init(&startCountMutex, 0);
  // Program ThreadedCount.cc (continued)
  pthread_mutex_init(&elementCountMutex, 0);
  pthread_cond_init(&exitCond, 0);
  // Main holds the exit mutex until waiting for exitCond condition
  pthread_mutex_lock(&exitMutex);

  // Create 16 threads
  for (unsigned int i = 0; i < static_cast<unsigned int>(nThreads); ++i) {
    // Now create the thread
    pthread_t pt; // pThread variable (output param from create)
    // Third param is the thread starting function
    // Fourth param is passed to the thread starting function
    pthread_create(&pt, 0, Transform2DThread, (void*)i) ;

    cout << "Thread " << i << " launched" << endl;
  }

  // Main program now waits until all child threads completed
  pthread_cond_wait(&exitCond, &exitMutex);

  cout << "Done!" << endl;

  // Write the transformed data


}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  int nThreads = 16;  // default to 16 threads

  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  if (argc > 2) nThreads = atoi(argv[2]);   // number of threads to run is the 2nd cmd line opt

  // die if the void cast will be a different size of memory
  if (sizeof(void*) != sizeof(unsigned long)) {
    exit(EXIT_FAILURE);
  }

  startCount = nThreads;

  Transform2D(fn.c_str(), nThreads); // Perform the transform.
  // At this point all thread have completed
}



