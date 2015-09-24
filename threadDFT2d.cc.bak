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

#include "Complex.h"
#include "InputImage.h"
#include "myBarrier.h"


using namespace std;


// the names of the output files
const std::string outfile_2d          = "Tower-DFT2D.txt";
const std::string outfile_2d_inv      = "MyAfterInverse.txt";
const std::string outfile_2d_inv_alt  = "TowerInverse.txt";


// Global variables visible to all threads
pthread_mutex_t   startCountMutex,  exitMutex,    elementCountMutex,
                  NMutex,           colMutex,     row_per_threadMutex;
pthread_cond_t    exitCond,         colCond;
Complex*          ImageData;
Complex*          Weights;
myBarrier*        barrier;
myBarrier*        barrier_begin_inv;
int               startCount;
unsigned int      N,                rows_per_thread;


// Takes the transpose of a matrix from the given width and height
void transpose(Complex* m, const size_t w, const size_t h)
{
  for (size_t i = 0; i < h; ++i)
    for (size_t j = i + 1; j < w; ++j)
      std::swap(m[j * w + i], m[i * w + j]);
}


// Flips a matrix along the X axis
void flip_horz(Complex*m, const size_t w, const size_t h)
{
  for (size_t i = 0; i < h; ++i)
    for (size_t j = i + 1; j < w / 2; ++j)
      std::swap(m[i * w + j], m[i * w + j + w / 2]);
}


// Flips a matrix along the Y axis
void flip_vert(Complex*m, const size_t w, const size_t h)
{
  transpose(m, w, h);
  flip_horz(m, w, h);
  transpose(m, w, h);
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

  // get the size of the weights array
  pthread_mutex_lock(&NMutex);
  const size_t NN = N;
  pthread_mutex_unlock(&NMutex);

  // create temporary valarrays for storing the even & odd values
  std::valarray<Complex> h_e = h[std::slice(0, sz / 2, 2)];
  std::valarray<Complex> h_o = h[std::slice(1, sz / 2, 2)];

  // take the fft of the even & odd values
  fft(h_e); fft(h_o);

  for (size_t n = 0; n < (sz / 2); ++n) {
    //Complex W = Complex( cos(2 * M_PI * n / sz), -1 * sin(2 * M_PI * n / sz) );
    Complex W = Weights[n * (NN / sz)];
    //W = W.Conj();

    h[n]            = h_e[n] + W * h_o[n];
    h[n + sz / 2]   = h_e[n] - W * h_o[n];
  }
}


// Wrapper function for taking the fft of a given
// number of values & length
void Transform1D(Complex * h, const size_t sz, const bool inverse = false)
{
  // Construct a std::valarray of the numbers we will compute the fft with
  std::valarray<Complex> h_vals = std::valarray<Complex>(h, sz);

  // Take the fft!
  fft(h_vals);

  // Now we copy the computed values into their correct locations
  for (size_t i = 0; i < sz; ++i)
    h[i] = h_vals[i] * (inverse ? (1.0 / N) : 1.0);
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

  // Do the individual 1D transforms on the rows assigned to this thread
  for (size_t row = 0; row < thread_num_rows; ++row) {
    const size_t row_offset = row * NN;
    Transform1D( &ImageData[thread_start_loc * NN + row_offset], NN );
  }

  // Wait for all to complete
  barrier->enter(threadID);

  // Wait for main to take the transpose
  barrier->enter(threadID);

  // Calculate 1d DFT for assigned columns
  for (size_t row = 0; row < thread_num_rows; ++row) {
    const size_t row_offset = row * NN;
    Transform1D( &ImageData[thread_start_loc * NN + row_offset], NN );
  }

  // determine if all other threads are complete so we can signal to main
  pthread_mutex_lock(&startCountMutex);
  startCount--;
  if (startCount <= 0) {
    // Last to finish our calculations, notify main
    pthread_mutex_lock(&exitMutex);
    pthread_cond_signal(&exitCond);
    pthread_mutex_unlock(&exitMutex);
  }
  pthread_mutex_unlock(&startCountMutex);


  // wait until main barriers after saving the 2d transform before we take the inverse
  barrier_begin_inv->enter(threadID);

  // Now we are working with the inverse transform
  for (size_t row = 0; row < thread_num_rows; ++row) {
    const size_t row_offset = row * NN;
    Transform1D( &ImageData[thread_start_loc * NN + row_offset], NN, true );
  }

  // entering twice for the same reasons we did before
  barrier->enter(threadID);
  barrier->enter(threadID);

  // back to the original rows
  for (size_t row = 0; row < thread_num_rows; ++row) {
    const size_t row_offset = row * NN;
    Transform1D( &ImageData[thread_start_loc * NN + row_offset], NN, true );
  }

  // determine if all other threads are complete so we can signal to main
  pthread_mutex_lock(&startCountMutex);
  startCount--;
  if (startCount <= 0) {
    // Last to exit, notify main
    pthread_mutex_lock(&exitMutex);
    pthread_cond_signal(&exitCond);
    pthread_mutex_unlock(&exitMutex);
  }
  pthread_mutex_unlock(&startCountMutex);

  return 0;
}


void Transform2D(const char* filename, size_t nThreads)
{
  // Create the helper object for reading the image
  InputImage image(filename);
  startCount = nThreads;

  // Store image data array as well as width/height
  ImageData       = image.GetImageData();
  size_t ImageWidth  = image.GetWidth();
  size_t ImageHeight = image.GetHeight();

  // set the global variable for number of rows/thread
  rows_per_thread = ImageWidth / nThreads;

  if (ImageHeight == ImageWidth) {
    // Set the global size variable
    N = ImageHeight;
  } else {
    cerr << "Image dimension mismatch (" << ImageHeight << " != " << ImageWidth << "). Exiting!" << endl;
    exit(EXIT_FAILURE);
  }

  // Check and make sure the given matrix dimensions are powers of 2
  if ((N & (N - 1)) != 0) {
    cerr << "Image dimensions not a power of 2 (" << ImageHeight << " != " << ImageWidth << "). Exiting!" << endl;
    exit(EXIT_FAILURE);
  }

  // All mutex/condition variables must be "initialized"
  pthread_mutex_init(&exitMutex, 0);
  pthread_mutex_init(&startCountMutex, 0);
  pthread_mutex_init(&elementCountMutex, 0);
  pthread_mutex_init(&NMutex, 0);
  pthread_mutex_init(&row_per_threadMutex, 0);
  pthread_cond_init(&exitCond, 0);

  // Assign the barrier object pointer
  barrier = new myBarrier(nThreads + 1);
  barrier_begin_inv = new myBarrier(nThreads + 1);

  // Precompute the weight values
  Weights = new Complex[N];
  for (size_t n = 0; n < N; ++n)
    Weights[n] = Complex( cos(2 * M_PI * n / N), -1 * sin(2 * M_PI * n / N) );

  // Main holds the exit mutex until waiting for exitCond condition
  pthread_mutex_lock(&exitMutex);

  // Get elapsed milliseconds (starting time after image loaded)
  get_clk_ms();

  // Create the correct number of threads
  for (size_t i = 0; i < nThreads; ++i) {
    // Now create the thread & start it
    pthread_t pt;
    pthread_create(&pt, 0, Transform2DThread, (void*)i) ;
  }

  // enter the barrier until the 1d transform is complete
  barrier->enter(nThreads);

  // take the transpose after all threads have completed their rows
  transpose(ImageData, ImageWidth, ImageHeight);

  // enter again so all of the threads will be released to start again
  barrier->enter(nThreads);

  // wait for the signal condition that the last thread is finished
  pthread_cond_wait(&exitCond, &exitMutex);

  // Write the transformed data
  image.SaveImageData(outfile_2d.c_str(), ImageData, ImageWidth, ImageHeight);

  // Reset the start count
  startCount = nThreads;

  // =====  START OF INVERSE =====

  // enter the barrier so it will be released for the threads to begin the ifft
  barrier_begin_inv->enter(nThreads);

  // now we enter back immediately so that we wait for the 1d inverse transpose to finish
  barrier->enter(nThreads);

  transpose(ImageData, ImageWidth, ImageHeight);

  barrier->enter(nThreads);

  pthread_cond_wait(&exitCond, &exitMutex);

  //for (size_t i = 0; i < h; ++i)
  for (size_t i = 0; i < (ImageWidth * ImageHeight) / 2; ++i)
    std::swap(ImageData[i], ImageData[ImageWidth * ImageHeight - i]);

  // Save the inverse data (and its alternative filename)
  image.SaveImageData(outfile_2d_inv.c_str(), ImageData, ImageWidth, ImageHeight);
  image.SaveImageData(outfile_2d_inv_alt.c_str(), ImageData, ImageWidth, ImageHeight);

  // Show how long it took to run everything
  clog << "--  runtime:\t" << get_clk_ms() / 1000.0 << " s" << endl;
}


int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  size_t nThreads = 16;  // default to 16 threads

  if (argc > 1) nThreads = atoi(argv[1]);   // number of threads to run is the 2nd cmd line opt
  if (argc > 2) fn = string(argv[2]);       // if name specified on cmd line

  // die if the void cast will be a different size of memory
  if (sizeof(void*) != sizeof(unsigned long)) {
    exit(EXIT_FAILURE);
  }

  // Perform the transforms
  Transform2D(fn.c_str(), nThreads);

  // At this point all thread have completed & there's nothing left to do
}
