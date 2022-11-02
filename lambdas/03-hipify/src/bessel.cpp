/*
 * HAVE_DEF is set during compile time
 * and determines which accelerator backend is used
 * by including the respective header file
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <climits>
#include <random>
#include <stdio.h>

#ifdef HAVE_MATPLOT
  #include <matplot/matplot.h>
#endif

// Namespaces "comms" and "devices" declared here
#include "comms.h"

// Set problem dimensions
#define N_ITER 10000
#define N_POPU 10000
#define N_SAMPLE 50

int main(int argc, char *argv []){

  // Initialize processes and devices
  comms::init_procs(&argc, &argv);
  const unsigned int my_rank = comms::get_rank();

  // Set spacing and range for beta
  const unsigned int n_beta = 40;
  const float range_beta = 4.0f;

  // Set timer
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  // Memory allocation
  float* mse_stdev = (float*)devices::allocate(n_beta * sizeof(float));
  float* mse_var = (float*)devices::allocate(n_beta * sizeof(float));

  // Use a non-deterministic random number generator for the master seed value
  std::random_device rd;

  // Use 64 bit Mersenne Twister 19937 generator
  std::mt19937_64 mt(rd());

  // Get a random unsigned long long from a uniform integer distribution (srand requires 32b uint)
  std::uniform_int_distribution<unsigned long long> dist(0, UINT_MAX);

  // Get the non-deterministic random master seed value
  unsigned long long seed = dist(mt);

  // Initialize the mean error array
  devices::parallel_for(n_beta, 
    DEVICE_LAMBDA(const int j) {
      mse_stdev[j] = 0.0f;
      mse_var[j] = 0.0f;
    }
  );

  // Run the loop over iterations
  devices::parallel_for(N_ITER, 
    DEVICE_LAMBDA(const int iter) {

      float p_mean = 0.0f;
      float s_mean = 0.0f;
      
      for(int i = 0; i < N_POPU; ++i){
        unsigned long long seq = ((unsigned long long)iter * (unsigned long long)N_POPU) + (unsigned long long)i;
        float rnd_val = devices::random_float(seed, seq, i, 100.0f, 15.0f);
        p_mean += rnd_val;
        if(i < N_SAMPLE) s_mean += rnd_val;
        if(iter == 0 && i < 3) printf("Rank %u, rnd_val[%d]: %.5f \n", my_rank, i, rnd_val);
      }
      
      p_mean /= N_POPU;
      s_mean /= N_SAMPLE;
      
      float b_var[n_beta];
      float b_sum = 0.0f;
      float p_var = 0.0f;
      
      for(int i = 0; i < N_POPU; ++i){
        unsigned long long seq = ((unsigned long long)iter * (unsigned long long)N_POPU) + (unsigned long long)i;
        float rnd_val = devices::random_float(seed, seq, i, 100.0f, 15.0f);
        float p_diff = rnd_val - p_mean;
        p_var += p_diff * p_diff;
        if(i < N_SAMPLE){
          float b_diff = rnd_val - s_mean;
          b_sum += b_diff * b_diff;   
        }
        //if(iter == 0 && i < 3) printf("Rank %u, rnd_val[%d]: %.5f? \n", my_rank, i, rnd_val);
      }
      p_var /= N_POPU;
      //printf("p_var: %f\n",p_var);
      
      for(int j = 0; j < n_beta; ++j){
        float sub = j * (range_beta / n_beta) - range_beta / 2.0f;
        b_var[j] = b_sum / (N_SAMPLE - sub);
        float diff_stdev = sqrtf(p_var) - sqrtf(b_var[j]);
        float diff_var = p_var - b_var[j];
        //printf("b_var[%d]: %f, error[iter: %d][sub: %f]: %f\n", j, b_var[j], iter, sub, sqrt(diff_var * diff_var));  

        // Sum the errors of each iteration
        devices::atomic_add(&mse_stdev[j], diff_stdev * diff_stdev);
        devices::atomic_add(&mse_var[j], diff_var * diff_var);
      }     
    }
  );

  // Each process sends its values to reduction, root process collects the results
  comms::reduce_procs(mse_stdev, n_beta);
  comms::reduce_procs(mse_var, n_beta);

#ifdef HAVE_MATPLOT
  // Define vectors for matplot  
  std::vector<float> x;
  std::vector<float> y1;
  std::vector<float> y2;
#endif

  // Divide the error sums to find the averaged errors for each tested beta value
  if(my_rank == 0){
    for(int j = 0; j < n_beta; ++j){
      mse_stdev[j] /= (comms::get_procs() * N_ITER);
      mse_var[j] /= (comms::get_procs() * N_ITER);
      float rmse_stdev = sqrtf(mse_stdev[j]);
      float rmse_var = sqrtf(mse_var[j]);
      float sub = j * (range_beta / n_beta) - range_beta / 2.0f;
      printf("Beta = %.2f: RMSE for stdev = %.5f and var = %.5f\n", sub, rmse_stdev, rmse_var);
#ifdef HAVE_MATPLOT     
      // Add data for matplot 
      x.push_back(sub);
      y1.push_back(rmse_stdev);
      y2.push_back(rmse_var);
#endif
    }
  }

  // Memory deallocations
  devices::free((void*)mse_stdev);
  devices::free((void*)mse_var);

  // Finalize processes and devices
  comms::finalize_procs();

  // Print timing
  if(my_rank == 0){
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
  }

#ifdef HAVE_MATPLOT   
  // Plot the standard deviation error (1st line) 
  auto p1 = matplot::plot(x, y1);
  p1->display_name("stdev");
  matplot::hold(matplot::on);

  // Plot the variance error (2nd line)
  auto p2 = matplot::plot(x, y2);
  p2->use_y2(true).display_name("variance");
  matplot::hold(matplot::off);

  // Create legend
  auto l = matplot::legend();
  l->location(matplot::legend::general_alignment::topright);
  
  // Set labels and style
  matplot::title("Root mean squared error (RMSE) for Bessel's correction");
  matplot::xlabel("Beta");
  matplot::ylabel("RMSE");
  matplot::grid(matplot::on);

  // Show plot
  matplot::show();
#endif

  return 0;
}
