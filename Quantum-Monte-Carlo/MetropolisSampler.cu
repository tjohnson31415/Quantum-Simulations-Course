#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include "my_helper.h"
#include "MetropolisSampler.h"

// Multiple updates per point per path; not fully implemented yet so keep at 1
#define MULTIUPDATE ((unsigned int) 1)

__host__ __device__ float potentialEnergy(float x)
{
    // Harmonic Oscillator Potential
    const float a2 = .15*.15;

    return .5 * a2 * x*x;

    // Quartic Potential 1
/*
 *    const double a2 = -4.0;
 *    const double a4 = 1.0;
 *    //offset so that the potential minimum is equal to zero
 *    const double offset = .25 * a2 * a2 / a4;
 *
 *    return .5 * a2 * x*x + .25 * a4 * x*x*x*x + offset;
 */

    // Quartic Potential 2
/*
 *    const double lambda = 0.0004;
 *    const double nu = 6.5;
 *
 *    return lambda * pow((x*x - nu*nu),2);
 */
    // Double SHO potential
/*
 *    const double omega = 0.15L;
 *    const double minima = 3.5L;
 *    const int ispositive = x >= 0;
 *
 *    return ispositive*.5*omega*omega*(x-minima)*(x-minima) + (1-ispositive)*.5*omega*omega*(x+minima)*(x+minima);
 */

}

/*
 * Computes the term in the total energy due to just a single grid point.
 * The sum of all elements in this array will produce twice the total energy of
 * the system, since each grid point shows up in two places.
 */
__device__ float perPositionEnergy(float const* grid_point_d, float epsT)
{
    float pos = *(grid_point_d);
    return .5f/epsT*(pow( pos - *(grid_point_d-1), 2) + pow( *(grid_point_d+1) - pos, 2)
                + epsT * potentialEnergy(pos));
}

__device__ float perPositionEnergy(float left, float center, float right, float epsT)
{
    return .5f/epsT*(pow( center - left, 2) + pow( right - center, 2) + epsT* potentialEnergy(center));
}

/*
 * Initializes the random number generators populating the inputted device array
 * with the inial states. Uses the default XORWOW pseudo-RNG.
 */
__global__ void initializeRNGsKernel(curandState* global_state, unsigned int seed)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread creates a RNG with the same seed but different sequence numbers
    curand_init( seed, tid, 0, &global_state[tid]);
}

/*
 * Generates the inital path for the algorithm.
 * TODO choose a better starting path than a constant
 */
__global__ void initializePathKernel(float* positions)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    positions[tid] = 0;
}

/*
 * Fills the energy vector by calling the perPositionEnergy function.
 * Also requires the halo to be set up which is why there is alot going on.
 * The algorithm for filling the shared memory is the same that is used in
 * updatePositionsKernel.
 */
__global__ void computeEnergiesKernel(float* __restrict__ energies, const float* __restrict__ positions, float epsilon)
{
    // The first step is to copy the needed data into shared memory for efficient access
    //__shared__ float shared_pos[512+2];
    extern __shared__ float shared_pos[];
    unsigned int num_threads = gridDim.x * blockDim.x;
    unsigned int global_tid  = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int block_tid   = threadIdx.x;

    // If thread 0 in the block, then copy in the halo points from the global array as well
    if( block_tid == 0 ) {
        if( blockIdx.x == 0 ) {
            // If this is the first block, the we take the left halo point from the end of the global array
            shared_pos[0] = *(positions + num_threads - 1);
        } else {
            shared_pos[0] = *(positions + global_tid - 1);
        }
        // Also need to get the right halo point
        if( blockIdx.x == gridDim.x - 1) {
            // If this is the last block, the we take the right halo point from the start of the global array
            shared_pos[blockDim.x+1] = *(positions);
        } else {
            shared_pos[blockDim.x+1] = *(positions + global_tid + blockDim.x);
        }
    }
    // All threads fill the main part of the shared memory with the positions from the grid
    shared_pos[block_tid+1] = positions[global_tid];

    __syncthreads();

    // Now that we have all of the data, we compute the energies
    //float* shPosition = &shared_pos[block_tid+1];
    //energies[global_tid] = perPositionEnergy( shPosition[-1], shPosition[0], shPostion[1], epsilon );
    energies[global_tid] = perPositionEnergy( shared_pos[block_tid], shared_pos[block_tid+1], shared_pos[block_tid+2], epsilon );
}
/*
 * Fills the energy vector by calling the perPositionEnergy function.
 * Also requires the halo to be set up which is why there is alot going on.
 * The algorithm for filling the shared memory is the same that is used in
 * updatePositionsKernel.
 */

/*
 *__global__ void computeEnergiesKernel(float* __restrict__ energies, const float* __restrict__ positions, float epsilon)
 *{
 *    // The first step is to copy the needed data into shared memory for efficient access
 *    unsigned int array_size = gridDim.x * blockDim.x;
 *    unsigned int global_tid  = threadIdx.x + blockIdx.x * blockDim.x;
 *
 *    float center_pos = positions[global_tid];
 *    float left_pos, right_pos;
 *
 *    // Need to choose the left and right positions for the endpoints specially
 *    if( global_tid == 0 ) {
 *        left_pos  = positions[array_size-1];
 *        right_pos = positions[global_tid+1];
 *    } else if( global_tid == array_size-1 ) {
 *        left_pos  = positions[global_tid-1];
 *        right_pos = positions[0];
 *    } else {
 *        left_pos  = positions[global_tid-1];
 *        right_pos = positions[global_tid+1];
 *    }
 *    // Now that we have all of the data, we compute the energies
 *    energies[global_tid] = perPositionEnergy( left_pos, center_pos, right_pos, epsilon );
 *}
 */
/*
 * Updates the grid using the Metropolis Monte Carlo algorithm.
 * TODO Just got the code so that it would work, I'm sure a better algorithm is possible
 */

__global__ void updatePositionsKernel( float* __restrict__ positions, float* __restrict__ energies,
                                       unsigned int* __restrict__ accepted_count,
                                       curandState* __restrict__ global_rng_state, float max_jump,
                                       float epsT,
                                       bool do_odds, bool do_acceptance)
{
    unsigned int array_size  = 2 * gridDim.x * blockDim.x;
    unsigned int global_tid  = threadIdx.x + blockIdx.x * blockDim.x;
    // offset index into the global arrays; we access every other element
    unsigned int index = do_odds + 2*global_tid;

    __shared__ unsigned int local_accepted_count;
    // Initialize the shared variable once
    if( threadIdx.x == 1 ) {
        local_accepted_count = 0;
    }

    // Sync so that the shared memory is initialized
    __syncthreads();

    float current_energy = energies[index];
    curandState rng_state = global_rng_state[index];

    float center_pos = positions[index];
    float left_pos, right_pos;

    // Need to choose the left and right positions for the endpoints specially
    if( threadIdx.x == 0 && blockIdx.x == 0 && !do_odds ) {
        left_pos  = positions[array_size - 1];
        right_pos = positions[index+1];
    } else if( threadIdx.x == blockDim.x-1 && blockIdx.x == gridDim.x-1 && do_odds ) {
        left_pos  = positions[index-1];
        right_pos = positions[0];
    } else {
        left_pos  = positions[index-1];
        right_pos = positions[index+1];
    }

//#pragma unroll TODO get the multi-update working
//for(unsigned int loop = 0; loop < MULTIUPDATE; loop += 1 ) {
    // Evaluate the proposed new position and compute its change in energy
    float delta_pos = max_jump * 2.0f*(.5f-curand_uniform(&rng_state));
    center_pos += delta_pos;

    float delta_energy = perPositionEnergy( left_pos, center_pos+delta_pos, right_pos, epsT ) - current_energy;

    // Check if the proposed jump will be accpeted
    //float accept_probability = exp( -delta_energy );
    bool is_accepted = (curand_uniform(&rng_state) < exp( -delta_energy ) );

    // Update the global arrays with the new values if there is a change
    if( is_accepted ) {
        positions[index] += delta_pos;
        if( do_acceptance )
            atomicAdd(&local_accepted_count, 1);
    }
//}

    // Update the global position
    //positions[index] = center_pos;
    // Update the global rng state
    global_rng_state[index] = rng_state;
    // Have thread 0 report the total number of accepted moves
    if( do_acceptance && threadIdx.x == 0 ) {
        atomicAdd(accepted_count, local_accepted_count);
    }
}

/*
__global__ void analyzePathKernel( float* result, float* path )
{
    float x_zero = path[0];
    float x_tau = path[tau_step];

    *correlation_sum += x_zero*x_tau;
    *x_zero_sum += x_zero;
    *x_tau_sum += x_tau;
}
*/


 /*
  * Consturctor for the sampler.
  */
MetropolisSampler::MetropolisSampler(float period, unsigned int path_length, float max_jump, unsigned int RNG_seed)
{
    m_period = period;
    m_path_length = path_length;
    if( (path_length % 256) != 0 ) {
        std::cerr << "MetropolisSampler: Error path_length must be a multiple of 256.";
        std::cerr << std::endl;
        exit(1);
    }
    m_num_paths_generated = 0;

    m_blockDim1D = 256;
    m_gridDim1D = m_path_length/m_blockDim1D;

    // Shared memory is 2 entries larger to store the halo data as well
    m_shared_mem_size = (m_blockDim1D + 2) * sizeof(float);

    m_max_jump = max_jump;
    m_epsilon_time = m_period/m_path_length;

    m_record_acceptance = true;
    m_num_paths_acceptance = 0;

    // Create the cuda stream to use for all asynchronous calls
    CUDA_CALL(cudaStreamCreate( &m_cuda_stream ) );
    // Create memory for the acceptance rate exponential moving average.
    CUDA_CALL(cudaMalloc(&m_accepted_counts_d, sizeof(unsigned int) ) );
    CUDA_CALL(cudaMemset(m_accepted_counts_d, 0, sizeof(unsigned int) ) );

    // Initialize the random number generators
    CUDA_CALL(cudaMalloc( &RNG_state_d, m_path_length * sizeof(curandState) ));
    initializeRNGsKernel<<<m_gridDim1D, m_blockDim1D, 0, m_cuda_stream>>> (RNG_state_d, RNG_seed);

    // Initialize the positions
    CUDA_CALL(cudaMalloc( &m_positions_d, m_path_length * sizeof(float) ));
    initializePathKernel<<<m_gridDim1D, m_blockDim1D, 0, m_cuda_stream>>> (m_positions_d);

    // Initialize the energies
    CUDA_CALL(cudaMalloc( &m_energies_d, m_path_length * sizeof(float) ));
    computeEnergiesKernel<<<m_gridDim1D, m_blockDim1D, m_shared_mem_size, m_cuda_stream>>>
        (m_energies_d, m_positions_d, m_epsilon_time);

    cudaStreamSynchronize( m_cuda_stream );
}

MetropolisSampler::~MetropolisSampler()
{
    cudaStreamSynchronize( m_cuda_stream );

    CUDA_CALL(cudaStreamDestroy( m_cuda_stream ) );

    CUDA_CALL(cudaFree( RNG_state_d ) );
    CUDA_CALL(cudaFree( m_positions_d ) );
    CUDA_CALL(cudaFree( m_energies_d ) );
    CUDA_CALL(cudaFree( m_accepted_counts_d ) );
}

// Generates a sample path according to the Metropolis algorithm
void MetropolisSampler::generateSamplePaths(int num)
{
    for( int i=0; i < num; i++) {
        // update the even positions
        updatePositionsKernel<<<m_gridDim1D, m_blockDim1D/2, 0, m_cuda_stream>>>
            (m_positions_d, m_energies_d,
             m_accepted_counts_d,
             RNG_state_d, m_max_jump, m_epsilon_time, 0, m_record_acceptance );

        // update the energies for all positions
        computeEnergiesKernel<<<m_gridDim1D, m_blockDim1D, m_shared_mem_size, m_cuda_stream>>>
            (m_energies_d, m_positions_d, m_epsilon_time);

        // update the odd positions
        updatePositionsKernel<<<m_gridDim1D, m_blockDim1D/2, 0, m_cuda_stream>>>
            (m_positions_d, m_energies_d,
             m_accepted_counts_d,
             RNG_state_d, m_max_jump, m_epsilon_time, 1, m_record_acceptance);

        // update the energies for all positions
        computeEnergiesKernel<<<m_gridDim1D, m_blockDim1D, m_shared_mem_size, m_cuda_stream>>>
            (m_energies_d, m_positions_d, m_epsilon_time);
    }
    m_num_paths_generated += num;
    if( m_record_acceptance ) {
        m_num_paths_acceptance += num;
    }

}


void MetropolisSampler::synchronize()
{
    cudaStreamSynchronize( m_cuda_stream );
}

float MetropolisSampler::getAcceptanceRate()
{
    cudaStreamSynchronize( m_cuda_stream );

    if( !m_record_acceptance ) {
        //std::cerr << "Acceptance rate is not being recorded, call doRecordAcceptance(true)." << std::endl;
        return -1;
    }
    // Quick exit if no new acceptance rate
    if( m_num_paths_generated == 0 ) {
        std::cerr << "No samples generated since last call to getAcceptanceRate()." << std::endl;
        return 0;
    }

    unsigned int accepted_counts;
    CUDA_CALL(cudaMemcpy(&accepted_counts, m_accepted_counts_d, sizeof(unsigned int), cudaMemcpyDeviceToHost ));
    float acceptance_rate = (float)accepted_counts / (m_path_length * m_num_paths_acceptance * MULTIUPDATE);

    // Reset the counts to compute a new acceptance rate
    //CUDA_CALL(cudaMemset(m_accepted_counts_d, 0x00, sizeof(unsigned int) ) );

    return acceptance_rate;
}

void MetropolisSampler::fillWithPath(float *path, unsigned int &num_filled)
{
    cudaStreamSynchronize( m_cuda_stream );
    // The number of elements written into the array
    num_filled = m_path_length;
    if( path == NULL ) {
        std::cerr << "MetropolisSampler::fillWithPath : Error - path array not allocated." << std::endl;
    }

    CUDA_CALL(cudaMemcpy(path, m_positions_d, num_filled * sizeof(float), cudaMemcpyDeviceToHost ));
}

void MetropolisSampler::fillWithEnergies(float *energies, unsigned int &num_filled)
{
    cudaStreamSynchronize( m_cuda_stream );
    // The number of elements written into the array
    num_filled = m_path_length;
    if( energies == NULL ) {
        std::cerr << "MetropolisSampler::fillWithEnergies : Error - energy array not allocated." << std::endl;
    }

    CUDA_CALL(cudaMemcpy(energies, m_energies_d, num_filled * sizeof(float), cudaMemcpyDeviceToHost ));
}

/*
// Returns the total action of the current path
float MetropolisSampler::totalEnergy()
{
    thrust::device_ptr<float> energies_ptr( m_energies_d );

    float energies_sum = thrust::reduce( energies_ptr, energies_ptr + m_path_length,
                          0.0f, thrust::plus<float>() );

    // The factor of two comes from the sum of the energies vector being twice
    // the total energy of the path
    return energies_sum/(2*m_path_length);
}
*/

// Returns the mean position squared of the current path
float MetropolisSampler::meanPositionSquared()
{
    cudaStreamSynchronize( m_cuda_stream );
    thrust::device_ptr<float> positions_ptr( m_positions_d );

    // For use in the lambda expression used below
    using namespace thrust::placeholders;

    float reduction = thrust::transform_reduce( positions_ptr, positions_ptr + m_path_length,
                                                  (_1 * _1), 0.0f, thrust::plus<float>() );

    return reduction/m_path_length;

}

void MetropolisSampler::printPotential(const float* x_grid, unsigned int grid_size, std::ostream& ostream )
{
    for(unsigned int i = 0; i < grid_size; i++) {
        ostream << std::setw(16) << x_grid[i];
        ostream << std::setw(16) << potentialEnergy(x_grid[i]) << std::endl;
    }
}

// Prints out the positions and energies for each timeslice on the path
void MetropolisSampler::printData(std::ostream& ostream)
{
    cudaStreamSynchronize( m_cuda_stream );

    //float* positions = (float*) malloc( m_path_length * sizeof(float) );
    float *positions = new float[m_path_length];
    //float* energies = (float*) malloc( m_path_length * sizeof(float) );
    float *energies = new float[m_path_length];

    cudaMemcpy( positions, m_positions_d, m_path_length*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy( energies, m_energies_d, m_path_length*sizeof(float), cudaMemcpyDeviceToHost);

    ostream << std::setprecision(8);
    for(unsigned int i = 0; i < m_path_length; i++) {
        ostream << std::setw(16);
        ostream << positions[i];
        ostream << std::setw(16);
        ostream << energies[i];
        ostream << std::endl;
    }
    ostream << std::endl;
    ostream << std::endl;

    //free( positions );
    //free( energies );
    delete[] positions;
    delete[] energies;
}

// vim: fdm=syntax : tags+=~/.vim/tags/cudacomplete
