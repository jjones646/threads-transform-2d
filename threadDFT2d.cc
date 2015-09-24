// Threaded two-dimensional Discrete FFT transform
// Jonathan Jones
// ECE8893 Project 2

#include <iostream>
#include <cstdlib>
#include <valarray>
#include <cstring>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>
#include <deque>

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

pthread_mutex_t*  wMutexes;
Complex*          weights;

// Function to reverse bits in an unsigned integer
unsigned int rev_bits(const unsigned int v, const size_t N)
{
  // N = size of array
  size_t n = N;
  unsigned r = 0;
  unsigned v_tmp = v;

  for (--n; n > 0; n >>= 1)
  {
    r <<= 1;        // Shift return value
    r |= (v_tmp & 0x1); // Merge in next bit
    v_tmp >>= 1;        // Shift reversal value
  }

  return r;
}

// Millisecond clock function
int get_clk_ms(void)
{
  timeval tv;
  static bool first = true;
  static int startSec = 0;

  gettimeofday(&tv, 0);

  if (first == true) {
    startSec = tv.tv_sec;
    first = false;
  }

  // Time in milliseconds
  return (tv.tv_sec - startSec) * 1000 + tv.tv_usec / 1000;
}


// Recursively take the fft using the Cooley–Tukey algorithm
void fft(std::valarray<Complex>& h)
{
  // Implement the efficient Danielson-Lanczos DFT here.
  const size_t sz = h.size();

  // stop once we are at the base case
  if (sz < 2) return;

  //bool locked = false;

  // create temporary valarrays for storing the even & odd values
  std::valarray<Complex> h_e = h[std::slice(0, sz / 2, 2)];
  std::valarray<Complex> h_o = h[std::slice(1, sz / 2, 2)];

  // if ((pthread_mutex_trylock(&clogMutex) == 0)) {
  //   locked = true;

  //   if (h_e.size() > 64) {
  //     clog << "--  size:\t[" << h_e.size() << "," << h_o.size() << "] -> " << sz << endl;
  //   }
  // }

  // take the fft of the even & odd values
  fft(h_e); fft(h_o);

  // combine the values to get the final fft & resort
  // the values to sizetheir correct return locations
  // for (size_t n = 0; n < sqrt(sz); ++n) {
  //   pthread_mutex_lock(&wMutexes[n]);
  //   Complex weights_local[n] =
  //   pthread_mutex_unlock(&wMutexes[n]);
  // }

  for (size_t n = 0; n < (sz / 2); ++n) {

    Complex W = Complex( cos(2 * M_PI * n / sz), -1 * sin(2 * M_PI * n / sz) );

    h[n]            = h_e[n] + W * h_o[n];
    h[n + sz / 2]   = h_e[n] - W * h_o[n];
  }

  // if (locked) {
  //   pthread_mutex_unlock(&clogMutex);
  // }
}


// Wrapper function for taking the fft of a given
// number of values & length
void Transform1D(Complex * h, const size_t sz)
{
  // Construct a std::valarray of the numbers we will compute the fft with
  std::valarray<Complex> h_vals = std::valarray<Complex>(h, sz);

  // Take the fft!
  fft(h_vals);

  // Now we copy the computed values into their correct locations
  for (size_t i = 0; i < sz; ++i)
    h[i] = Complex(h_vals[i]);
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

  // pthread_mutex_lock(&clogMutex);

  // Do the individual 1D transforms on the rows assigned to this thread
  for (size_t row = 0; row < thread_num_rows; ++row) {
    const size_t row_offset = row * NN;

    // if (threadID == 0) {
    //   clog << &ImageData[thread_start_loc * NN + row_offset] - &ImageData[thread_start_loc * NN + ((row - 1) * NN)] << NN << endl;
    // }

    Transform1D(&ImageData[thread_start_loc * NN + row_offset], NN);
  }

  // if (threadID == 0) {
  //   clog << endl << endl;
  // }

  // pthread_mutex_unlock(&clogMutex);

// Wait for all to complete
  barrier->enter(threadID);

// Calculate 1d DFT for assigned columns

  pthread_mutex_lock(&startCountMutex);
  startCount--;

  if (startCount <= 0) {
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

  // 1c) Check and make sure the given matrix dimensions are powers of 2
  if ((N & (N - 1)) != 0) {
    cerr << "Image dimensions not power of 2 (" << ImageHeight << " != " << ImageWidth << "). Quitting" << endl;
    exit(EXIT_FAILURE);
  }
  // else {
  //   wMutexes = new pthread_mutex_t[sqrt(N)];
  //   for (size_t n = 0; n < sizeof(wMutexes / sizeof(pthread_mutex_t)); ++n) {
  //     pthread_mutex_init(&wMutexes[n], 0);
  //   }
  // }

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

  // Get elapsed milliseconds (starting time after image loaded)
  get_clk_ms();

  // Precompute an array of weights
  // for (size_t n = 0; n < sizeof(wMutexes / sizeof(pthread_mutex_t)); ++n) {
  //   weights[n] = Complex( cos(2 * M_PI * n / nThreads), -1 * sin(2 * M_PI * n / nThreads) );
  // }

  // Create the correct number of threads
  for (size_t i = 0; i < nThreads; ++i) {
    // Now create the thread & start it
    pthread_t pt;
    pthread_create(&pt, 0, Transform2DThread, (void*)i) ;
  }

  // Main program now waits until all child threads completed
  pthread_cond_wait(&exitCond, &exitMutex);

  clog << "--  rows/thd:\t" << rows_per_thread << endl;
  clog << "--  vals_t:\t" << ImageWidth * ImageHeight << endl;

  // Show how long it took to run everything
  clog << "--  runtime:\t" << get_clk_ms() / 1000.0 << " s" << endl;

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



