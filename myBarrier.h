#pragma once

/*
 * Class declaration for a barrier class
 */

#include "pthread.h"

class myBarrier {
public:
    myBarrier(int P0); // P is the total number of threads
    void enter(int myId); // Enter the barrier, don’t exit till alll there

private:
    int fetch_n_decrement(void);

    pthread_mutex_t countMutex;
    bool* localSense; // We will create an array of bools, one per threadj
    bool globalSense; // Global sense
    int P;
    int count; // Number of threads presently in the barrier
};
