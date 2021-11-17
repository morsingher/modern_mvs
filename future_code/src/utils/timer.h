#ifndef MVS_TIMER_H
#define MVS_TIMER_H

#include "common.h"

using namespace std::chrono;

class Timer {
 public:
  Timer(const std::string& name) : func(name), started(true) { start_time = high_resolution_clock::now(); };
  inline void stop()
  {
    if (!started) {
      std::cout << "Timer never started" << std::endl;
    }
    else {
      const auto end_time = high_resolution_clock::now();
      const auto duration = duration_cast<nanoseconds>(end_time - start_time);
      const double elapsed = duration.count() / 1e9;
      std::cout << std::endl << "Elapsed time for timer " << func << ": " << elapsed << " s" << std::endl << std::endl;
    }
  }

 private:
  std::string func;
  bool started;
  high_resolution_clock::time_point start_time;
};

#endif