// Threaded two-dimensional Discrete FFT transform
// Jonathan Jones
// ECE8893 Project 2

#include <iostream>
#include <cstdlib>
#include <valarray>
#include <cstring>

#include <math.h>

#include "pthread.h"

#include "Complex.h"
#include "InputImage.h"
#include "myBarrier.h"


using namespace std;


// Global variables visible to all threads
pthread_mutex_t   startCountMutex,  exitMutex,    elementCountMutex,    NMutex,     row_per_threadMutex;
pthread_cond_t    exitCond;
Complex*          ImageData;
myBarrier*        barrier;
int               startCount;
unsigned int      N, rows_per_thread;

pthread_mutex_t   clogMutex;

// Function to reverse bits in an unsigned integer
// This assumes there is a global variable N that is the
// number of points in the 1D transform.
unsigned ReverseBits(const unsigned int v)
{ //  Provided to students
  size_t n = N; // Size of array (which is even 2 power k value)
  unsigned r = 0; // Return value
  unsigned v_tmp = v;

  for (--n; n > 0; n >>= 1)
  {
    r <<= 1;        // Shift return value
    r |= (v_tmp & 0x1); // Merge in next bit
    v_tmp >>= 1;        // Shift reversal value
  }

  return r;
}


// Recursively take the fft using the Cooley–Tukey algorithm
void fft(std::valarray<Complex>& h)
{
  // Implement the efficient Danielson-Lanczos DFT here.
  const size_t N = h.size();

  // stop once we are at the base case
  if (N <= 1) return;

  // create temporary vectors for storing the even & odd values
  std::valarray<Complex> h_e = h[std::slice(0, N / 2, 2)];
  std::valarray<Complex> h_o = h[std::slice(1, N / 2, 2)];

  // take the fft of the even & odd values
  fft(h_e); fft(h_o);

  // combine the values to get the final fft & resort
  // the values to their correct return locations
  for (size_t n = 0; n < N / 2; ++n) {
    Complex W = Complex( (cos(2 * M_PI * n / N)), (-1 * sin(2 * M_PI * n / N)));
    h[n]          = h[2 * n] + W * h[2 * n + 1];
    h[n + N / 2]  = h[2 * n] - W * h[2 * n + 1];
  }
}


// Wrapper function for taking the fft of a given
// number of values & length
void Transform1D(Complex* h, const size_t N)
{
  // Construct a std::valarray of the numbers we will compute the fft with
  std::valarray<Complex> h_vals = std::valarray<Complex>(h, N);

  // Take the fft!
  fft(h_vals);

  // Now we copy the computed values into their correct locations
  for (size_t i = 0; i < N; ++i)
    h[i] = h_vals[i];
}


void* Transform2DThread(void* const arg)
{
  // we are passed our thread number
  const unsigned long threadID = (unsigned long)arg;

  // find out the dimension so we can find our starting location
  pthread_mutex_lock(&NMutex);
  const size_t NN = N;
  pthread_mutex_unlock(&NMutex);

  // set how many threads we are working with
  pthread_mutex_lock(&row_per_threadMutex);
  const size_t thread_num_rows = rows_per_thread;
  pthread_mutex_unlock(&row_per_threadMutex);

  // now we can find our starting location with the new info
  const size_t thread_start_loc = thread_num_rows * threadID;

  pthread_mutex_lock(&clogMutex);
  clog << "-- thread " << threadID << " starting at index " << thread_start_loc << endl;
  pthread_mutex_unlock(&clogMutex);

  // Do the individual 1D transforms on the rows assigned to this thread
  for (size_t row = 0; row < thread_num_rows; ++row) {
    const size_t row_offset = row * NN;
    Transform1D(&ImageData[thread_start_loc * NN + row_offset], NN);
  }

  // Wait for all to complete
  barrier->enter(threadID);

  // Calculate 1d DFT for assigned columns

  pthread_mutex_lock(&startCountMutex);
  startCount--;

  if (startCount == 0) {
    // Last to exit, notify main
    // pthread_mutex_unlock(&startCountMutex);

    pthread_mutex_lock(&exitMutex);
    pthread_cond_signal(&exitCond);
    pthread_mutex_unlock(&exitMutex);
  }
  // else {
  //   // release mutex
  pthread_mutex_unlock(&startCountMutex);
  // }

  return 0;
}


void Transform2D(const char* filename, size_t nThreads)
{
  // Create the helper object for reading the image
  InputImage image(filename);
  startCount = nThreads;

  // Store image data array as well as width/height
  ImageData       = image.GetImageData();
  int ImageWidth  = image.GetWidth();
  int ImageHeight = image.GetHeight();

  // set the global variable for number of rows/thread
  rows_per_thread = ImageWidth / nThreads;

  if (ImageHeight == ImageWidth) {
    // Set the global size variable
    N = ImageHeight;
  } else {
    cerr << "Image height/width mismatch (" << ImageHeight << " != " << ImageWidth << "). Quitting" << endl;
    exit(EXIT_FAILURE);
  }

  // All mutex/condition variables must be "initialized"
  pthread_mutex_init(&exitMutex, 0);
  pthread_mutex_init(&startCountMutex, 0);
  pthread_mutex_init(&elementCountMutex, 0);
  pthread_mutex_init(&NMutex, 0);
  pthread_mutex_init(&row_per_threadMutex, 0);
  pthread_mutex_init(&clogMutex, 0);
  pthread_cond_init(&exitCond, 0);

  // Assign the barrier object pointer
  barrier = new myBarrier(nThreads);

  // Main holds the exit mutex until waiting for exitCond condition
  pthread_mutex_lock(&exitMutex);

  // Create the correct number of threads
  for (unsigned int i = 0; i < static_cast<unsigned int>(nThreads); ++i) {
    // Now create the thread & start it
    pthread_t pt;
    pthread_create(&pt, 0, Transform2DThread, (void*)i) ;
  }

  // Main program now waits until all child threads completed
  pthread_cond_wait(&exitCond, &exitMutex);
  clog << "-- done!" << endl;

  // Write the transformed data
  image.SaveImageData("MyAfterInverse.txt", ImageData, ImageWidth, ImageHeight);
}


int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  size_t nThreads = 16;  // default to 16 threads

  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  if (argc > 2) nThreads = atoi(argv[2]);   // number of threads to run is the 2nd cmd line opt

  // die if the void cast will be a different size of memory
  if (sizeof(void*) != sizeof(unsigned long)) {
    exit(EXIT_FAILURE);
  }

  Transform2D(fn.c_str(), nThreads); // Perform the transform.
  // At this point all thread have completed
}



