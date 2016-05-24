/*
 * This is how a barrier is implemented
 */

#include "myBarrier.h"

#include <iostream>
#include <pthread.h>

using namespace std;

myBarrier::myBarrier(int P0)
	: P(P0), count(P0)
{
	// Initialize the mutex used for FetchAndIncrement
	pthread_mutex_init(&countMutex, 0);

	// Create and initialize the localSense arrar, 1 entry per thread
	localSense = new bool[P];

	for (int i = 0; i < P; ++i) localSense[i] = true;

	// Initialize global sense
	globalSense = true;
}

void myBarrier::enter(int myId)
{
	// Toggle private sense variable
	localSense[myId] = !localSense[myId];

	if ( fetch_n_decrement() == 1 ) {
		// All threads here, reset count and toggle global sense
		count = P;
		globalSense = localSense[myId];
	}
	else {
		while (globalSense != localSense[myId]) { } // wait here for the others
	}
}

int myBarrier::fetch_n_decrement(void)
{
	int myCount;

	// We don’t have an atomic FetchAndDecrement, but we can get the
	// same behavior by using a mutex
	pthread_mutex_lock(&countMutex);
	myCount = count--;
	pthread_mutex_unlock(&countMutex);

	return myCount;
}
