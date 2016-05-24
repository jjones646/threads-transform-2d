// Threaded two-dimensional Discrete FFT transform
// Jonathan Jones
// ECE8893 Project 2

#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <valarray>
#include <vector>
#include <numeric>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>

#include "Complex.h"
#include "InputImage.h"
#include "myBarrier.h"

using namespace std;

// the names of the output files
const std::string outfile_2d          = "Tower-DFT2D.txt";
const std::string outfile_2d_inv      = "Tower-IDFT2D.txt";
// testbench values to check the results against
const std::string correct_2d          = "after2d-correct.txt";

const std::string DEFAULT_IN_FILENAME = "Tower.txt";
const size_t DEFAULT_NUM_THREADS      = 16;
const bool VERIFY_VALUES              = false;

// Global variables visible to all threads
pthread_mutex_t   start_cnt_M   = PTHREAD_MUTEX_INITIALIZER,
                  exit_M        = PTHREAD_MUTEX_INITIALIZER,
                  elem_cnt_M    = PTHREAD_MUTEX_INITIALIZER,
                  dim_M         = PTHREAD_MUTEX_INITIALIZER,
                  row_per_M     = PTHREAD_MUTEX_INITIALIZER,
                  run_tm_M      = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t    exit_C        = PTHREAD_COND_INITIALIZER;
Complex*          ImageData;
Complex*          Weights;
myBarrier*        barrier;
myBarrier*        barrier_begin_inv;
size_t            start_cnt,
                  N,
                  rows_per_thread;
// vector that holds all thread's final runtime
std::vector<int>  runtimes;


// Verify two Complex arrays against each other - returns the percent different
double verify(const Complex* correct_vals, const Complex* check_vals, const size_t sz)
{
  double err_t = 0.0;
  std::valarray<double> err(sz);

  for (size_t i = 0; i < sz; ++i)
    err[i] = std::abs(correct_vals[i].get_mag() - check_vals[i].get_mag()) / correct_vals[i].get_mag();

  err_t = err.sum() / static_cast<double>(sz);

  if (err_t < 1.0e-10)
    err_t = 0.0;

  if ( (err.max() < 1.0e-4) && (isnormal(err_t)) )
    err_t = 0.0;

  return (isnormal(err_t) ? err_t : 0.0);
}

// Takes the transpose of a matrix from the given width and height
void transpose(Complex* m, const size_t w, const size_t h) {
  for (size_t i = 0; i < h; ++i)
    for (size_t j = i + 1; j < w; ++j)
      std::swap(m[j * w + i], m[i * w + j]);
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
  pthread_mutex_lock(&dim_M);
  const size_t NN = N;
  pthread_mutex_unlock(&dim_M);

  // create temporary valarrays for storing the even & odd values
  std::valarray<Complex> h_e = h[std::slice(0, sz / 2, 2)];
  std::valarray<Complex> h_o = h[std::slice(1, sz / 2, 2)];

  // take the fft of the even & odd values
  fft(h_e); fft(h_o);

  for (size_t n = 0; n < (sz / 2); ++n) {
    // lookup what weight we are using
    Complex W = Weights[n * (NN / sz)];

    h[n]            = h_e[n] + W * h_o[n];
    h[n + sz / 2]   = h_e[n] - W * h_o[n];
  }
}

// Wrapper function for taking the fft of a given
// number of values & length
void Transform1D(Complex * h, const size_t sz, bool inverse = false)
{
  // Construct a std::valarray of the numbers we will compute the fft with
  std::valarray<Complex> h_vals = std::valarray<Complex>(h, sz);

  // Take the conjugate if we're taking the inverse
  if (inverse) {
    for (size_t i = 0; i < sz; ++i)
      h_vals[i] = h_vals[i].Conj();
  }

  // Take the fft!
  fft(h_vals);

  // Now we copy the computed values into their correct locations
  // and fixup the values if we're taking the IFFT
  if (inverse) {
    for (size_t i = 0; i < sz; ++i) {
      h[i] = h_vals[i].Conj() * (1.0 / sz);

      if (h_vals[i] < 1.0e-10)
        h[i] = Complex(0);
    }
  }
  else {
    for (size_t i = 0; i < sz; ++i) {
      h[i] = h_vals[i];

      if (h_vals[i] < 1.0e-10)
        h[i] = Complex(0);
    }
  }
}

// Each thread's operation function
void* Transform2DThread(void* const arg)
{
  // we are passed our thread number
  const unsigned long threadID = (unsigned long)arg;

  // find out the dimension so we can find our starting location
  pthread_mutex_lock(&dim_M);
  const size_t NN = N;
  pthread_mutex_unlock(&dim_M);

  // set how many threads we are working with
  pthread_mutex_lock(&row_per_M);
  const size_t thread_num_rows = rows_per_thread;
  pthread_mutex_unlock(&row_per_M);

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
  pthread_mutex_lock(&start_cnt_M);
  start_cnt--;

  if (start_cnt <= 0) {
    // Last to finish our calculations, notify main
    pthread_mutex_lock(&exit_M);
    pthread_cond_signal(&exit_C);
    pthread_mutex_unlock(&exit_M);
  }
  pthread_mutex_unlock(&start_cnt_M);

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

  // save how long this thread took to compute everything
  pthread_mutex_lock(&run_tm_M);
  runtimes.push_back(get_clk_ms());
  pthread_mutex_unlock(&run_tm_M);

  // Don't exit until we know all of the computations are complete from other threads
  barrier->enter(threadID);

  return 0;
}

void Transform2D(const char* filename, size_t nThreads, bool check_results = false)
{
  // Create the helper object for reading the image
  InputImage image(filename);
  start_cnt = nThreads;

  // Store image data array as well as width/height
  ImageData           = image.GetImageData();
  size_t ImageWidth   = image.GetWidth();
  size_t ImageHeight  = image.GetHeight();

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

  // Assign the barrier object pointer
  barrier = new myBarrier(nThreads + 1);
  barrier_begin_inv = new myBarrier(nThreads + 1);

  // Precompute the weight values
  Weights = new Complex[N];
  for (size_t n = 0; n < N; ++n)
    Weights[n] = Complex( cos(2 * M_PI * n / N), -1 * sin(2 * M_PI * n / N) );

  // Main holds the exit mutex until waiting for exit_C condition
  pthread_mutex_lock(&exit_M);

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
  pthread_cond_wait(&exit_C, &exit_M);

  // take the transpose to complete the 2d transform
  transpose(ImageData, ImageWidth, ImageHeight);

  // Write the transformed data
  image.SaveImageData(outfile_2d.c_str(), ImageData, ImageWidth, ImageHeight);
  clog << "results written to " << outfile_2d << endl;

  if (check_results == true) {
    // Check the results against the testbench values
    InputImage  testbench(correct_2d.c_str());
    Complex*    testbench_data  = testbench.GetImageData();
    size_t      testbench_N     = testbench.GetWidth();
    double perc_diff = verify(ImageData, testbench_data, testbench_N * testbench_N);
    cout << "--  FFT 2D results:\t" << ((perc_diff == 0.0) ? "PASS" : "FAIL")
         << "\t(" << std::fixed << std::setprecision(6) << ((1.0 - perc_diff) * 100.0) << "% similar)" << endl;
    // cleanup since we do the inverse now
    delete[] testbench_data;
  }

  // =====  START OF INVERSE =====

  // enter the barrier so it will be released for the threads to begin the ifft
  barrier_begin_inv->enter(nThreads);

  // now we enter back immediately so that we wait for the 1d inverse transpose to finish
  barrier->enter(nThreads);

  // flip the columns into the rows
  transpose(ImageData, ImageWidth, ImageHeight);

  // enter into the barrier so that the threads will be released
  barrier->enter(nThreads);

  // wait for all of the threads to compute the rows
  barrier->enter(nThreads);

  // flip the matrix so we are back to our original placements
  transpose(ImageData, ImageWidth, ImageHeight);

  // save the results
  image.SaveImageData(outfile_2d_inv.c_str(), ImageData, ImageWidth, ImageHeight);
  clog << "inverse results written to " << outfile_2d_inv << endl;

  // Check the results against the testbench values
  if (check_results == true) {
    InputImage  testbench2(DEFAULT_IN_FILENAME.c_str());
    Complex*    testbench2_data  = testbench2.GetImageData();
    size_t      testbench2_N     = testbench2.GetWidth();
    double perc_diff2 = verify(ImageData, testbench2_data, testbench2_N * testbench2_N);
    cout << "--  IFFT 2D results:\t" << ((perc_diff2 == 0.0) ? "PASS" : "FAIL")
         << "\t(" << std::fixed << std::setprecision(6) << ((1.0 - perc_diff2) * 100.0) << "% similar)" << endl;
    // cleanup since we do the inverse now
    delete[] testbench2_data;
  }

  std::vector<int>::iterator min_time = std::min_element(runtimes.begin(), runtimes.end());
  std::vector<int>::iterator max_time = std::max_element(runtimes.begin(), runtimes.end());
  int avg_time = std::accumulate(runtimes.begin(), runtimes.end(), 0) / runtimes.size();

  // Show how long it took to run everything
  clog << "===== RUNTIME REPORT =====\t" << std::fixed << std::setprecision(4) << endl
       << "  Min:  \t" << (*min_time) / 1000.0 << "s" << endl
       << "  Avg:  \t" << avg_time / 1000.0 << "s" << endl
       << "  Max:  \t" << (*max_time) / 1000.0 << "s" << endl
       << "  Total:\t" << get_clk_ms() / 1000.0 << "s" << endl
       << "  Threads:\t" << nThreads << endl
       << "==========================\t" << endl;
}

int main(int argc, char** argv)
{
  string fn(DEFAULT_IN_FILENAME); // default file name
  size_t nThreads = DEFAULT_NUM_THREADS;   // default to 16 threads

  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <input-matrix> [num-threads]" << endl;
    exit(EXIT_FAILURE);
  }

  // if name specified on cmd line
  if (argc > 1) fn        = string(argv[1]);

  // number of threads to run is the 2nd cmd line opt
  if (argc > 2) nThreads  = atoi  (argv[2]);

  // die if the void cast will be a different size of memory
  if ( sizeof(void*) != sizeof(size_t) ) {
    exit(EXIT_FAILURE);
  }

  // Perform the transform
  Transform2D(fn.c_str(), nThreads, VERIFY_VALUES);
}



