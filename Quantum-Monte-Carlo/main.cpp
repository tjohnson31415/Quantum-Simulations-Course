#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <vector>
#include <utility>
#include <string>

#include <cuda_runtime.h>

#include "MetropolisSampler.h"
#include "ThrustHistogram.h"

#include "my_helper.h"

#define MIDSTREAM_OUTPUTS 1

// Parameters for the simulation and data collection
static const unsigned int num_warm_up = 100000;
static const unsigned int num_paths = 1000; // number of paths to generate and analyze
static const unsigned int print_path_every = num_paths/20; // how often to print sample paths to file
static const unsigned int print_corr_every = num_paths/20; // how often to print sample paths to file
static const unsigned int num_skip = 100; // distributions skipped between analyzed paths

// The two paratemeters below determine the time step epsilon
static const float period = 1024.0f; // the period of the path to be generated
static const unsigned int num_time_steps = 1024; // needs to be a multiple of 256

static const float max_jump = 1.0f; // maximum displacement in a single step

// Parameters for the correlation calculation
static const unsigned int max_tau = 16;
static const unsigned int max_corr_power = 3;
static const unsigned int num_correlations = max_corr_power * (max_corr_power + 1) / 2;

// Parameters to describe the histogram
static const float histogram_range = 10.0f;
static const float histogram_binwidth = 0.05f;

// Variables and parameters for the statistics to be generated
static double sum_correlations[num_correlations][max_tau] = {0};
// Holds the current path on the host
static float current_path[num_time_steps];

double computeCorrelation( unsigned int pow1, unsigned int pow2, unsigned int tau )
{
    double corr_sum = 0;
    for( unsigned int t = 0; t < num_time_steps; ++t) {
         corr_sum += pow(current_path[t], pow1) * pow(current_path[(t+tau)%num_time_steps], pow2);
    }
    return corr_sum/num_time_steps;
}

void printCorrelations(std::ostream &ostream)
{
    ostream << std::setprecision(12);
    unsigned int col_width = 24;
    for( unsigned int tau = 0; tau < max_tau; ++tau) {
        for( unsigned int corr = 0; corr < num_correlations; ++corr) {
            double correlations_average = sum_correlations[corr][tau] / num_paths;
            ostream << std::setw(col_width) << correlations_average;
        }
        ostream << std::endl;
    }
}

int main(int argc, char* argv[])
{
    /////////////////////////////////////////////////////////
    /////      Initializations and Set up
    /////////////////////////////////////////////////////////

    // Output files for the simulation results
    std::ofstream file_paths;
    std::ofstream file_correlations;
    std::ofstream file_wavefunction;

    file_paths.open("outputs/paths.dat");
    file_correlations.open("outputs/correlations.dat");
    file_wavefunction.open("outputs/wavefunction.dat");

    // Create instances of the classes that do all the work
    //MetropolisSampler sampler(period, num_time_steps, max_jump, std::clock() );
    MetropolisSampler sampler(period, num_time_steps, max_jump, 1337 );
    ThrustHistogram hist( -histogram_range, histogram_range, histogram_binwidth );

    // Pointers to arrarys that will exist on the cpu side to copy the gpu
    // simulation data to for output
    float* sample_energies = new float[sampler.m_path_length];
    float* x_grid = new float[hist.m_numbins];
    unsigned int grid_size = 0;
    unsigned int path_length = 0;

    // Construct the pairs of powers for the correlation matrix calculations
    std::vector< std::pair<unsigned,unsigned> > powers;
    for( unsigned int i = 0; i < max_corr_power; ++i)
        for( unsigned int j = i; j < max_corr_power; ++j)
            powers.push_back(std::make_pair(i+1,j+1));

    // Turn off the acceptance rate calculation for efficiency
    sampler.doRecordAcceptance(false);
    // Warm up the Markov Engine so that it can find the desired probability
    // distribution
    std::cout << "Warming up the Markov Machine." << std::endl;
    sampler.generateSamplePaths(num_warm_up);
    // Reset the acceptance rate counters
    sampler.getAcceptanceRate();

    /////////////////////////////////////////////////////////
    /////      Main Monte Carlo loop
    /////////////////////////////////////////////////////////
    sampler.doRecordAcceptance(true);
    std::cout << "Entering the main loop." << std::endl;

    for(unsigned int i = 0; i < num_paths; i++) {
        // Do all the outputting of information first in hopes that the path generation can be done
        //   concurrently with the generation of new statistics

        // Copy out the current_path
        sampler.fillWithPath(current_path, path_length);

        // Add the current path to the histogram
        hist.addData( sampler.getPathPtr(), sampler.m_path_length );

        // Output sample paths if desired
#if MIDSTREAM_OUTPUTS
        if( (i % print_path_every) == 0 ) {
            //std::cout << "Outputting a sample path for iteration " << i << std::endl;
            sampler.fillWithEnergies(sample_energies, path_length);
            for(unsigned int j = 0; j < path_length; ++j) {
                file_paths << std::setw(16) << current_path[j];
                file_paths << std::setw(16) << sample_energies[j];
                file_paths << std::endl;
            }
            // Skip two lines to separate data sets for gnuplot
            file_paths << std::endl;
            file_paths << std::endl;
        }
#endif

        if( (i % print_corr_every) == 0 && i != 0 ) {
            printCorrelations(file_correlations);
        }
        // Tell the sampler to begin generating new paths
        sampler.generateSamplePaths(num_skip);

        // Compute our correlation matricies
#pragma omp parallel for schedule(static) shared(powers, sum_correlations) num_threads(num_correlations)
        for( unsigned int corr = 0; corr < num_correlations; ++corr) {
            for( unsigned int tau = 0; tau < max_tau; ++tau) {
                std::pair<unsigned, unsigned> power_pair = powers[corr];
                sum_correlations[corr][tau] += computeCorrelation(power_pair.first, power_pair.second, tau);
            }
        }

        // Get a check of the acceptance rate from the fist 10 paths (* num_skip)
        if( i == 10 ) {
            std::cout << "Acceptance Rate: " << sampler.getAcceptanceRate() << std::endl;
            sampler.doRecordAcceptance(false);
        }

        // Progress bar
        if( i% ((int)ceil(num_paths/20.0)) == 0 && i != 0) {
            //std::cout << float(i)/num_paths*100 << "% Complete." << std::endl;
            std::cout << "===";
            std::cout.flush();
        }
    }
    std::cout << std::endl; // new line after the progress bar
    std::cout << "Simulation Complete. Finalizing." << std::endl;

    /////////////////////////////////////////////////////////
    /////      Output the results of the simulation
    /////////////////////////////////////////////////////////

    // Print the normalized histogram to the results file
    hist.printHistogramNormalized(file_wavefunction);
    // Print two blank lines for gnuplot
    file_wavefunction << std::endl << std::endl;
    // Get the values of the center of the bins from the histogram
    hist.getGridArray(x_grid, grid_size);
    // Pass these into the sampler to print the potential at those points
    sampler.printPotential( x_grid, grid_size, file_wavefunction);
    // Print two blank lines for gnuplot
    file_wavefunction << std::endl << std::endl;

    // A check on the histogram to see how many points weren't in the bins
    hist.printUnbinned( std::cout );

    // Output the correlation matrix elements
    printCorrelations(file_correlations);

    /////////////////////////////////////////////////////////
    /////      Clean up time
    /////////////////////////////////////////////////////////

    file_paths.close();
    file_correlations.close();
    file_wavefunction.close();

    delete[] sample_energies;
    delete[] x_grid;

    return 0;
}

// vim: fdm=syntax : tags+=~/.vim/tags/cudacomplete
