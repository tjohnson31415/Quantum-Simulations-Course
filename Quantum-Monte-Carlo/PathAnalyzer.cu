#include <iostream>
#include <iomanip>
#include <cstring> // for memset

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "my_helper.h"
#include "PathAnalyzer.h"

/*
 * Constructor
 */
PathAnalyzer::PathAnalyzer(unsigned int path_length, unsigned int max_tau, unsigned int max_power )
    :   m_max_power(max_power)
    ,   m_max_tau(max_tau)
    ,   m_path_length(path_length)
    ,   m_num_paths_analyzed(0)
{
    CUDA_CALL(cudaMalloc(&m_path_d, (m_path_length + m_max_tau) * sizeof(m_path_d[0])) );
    m_path_devptr = thrust::device_ptr<float>( m_path_d );

    // Allocate the memory for the correlators
    m_num_correlations = m_max_power * (m_max_power + 1) / 2;
    m_correlations = new double*[m_num_correlations];
    for( unsigned int i = 0; i < m_num_correlations; ++i) {
        m_correlations[i] = new double[m_max_tau];
        // Ensure that the values are set to zero since we will just be adding to them
        memset( m_correlations[i], 0, m_max_tau*sizeof(m_correlations[0][0]) );
    }

    // Create the functors used to compute the correlations
    m_functors = new correlation_functor[m_num_correlations];
    unsigned int index = 0;
    // The functors are stored to match the symmetric correlations
    // matrix being stored in triangular form
    for( unsigned int i = 0; i < m_max_power; ++i) {
        for( unsigned int j = i; j < m_max_power; ++j) {
            m_functors[index] = correlation_functor(i+1, j+1);
            index += 1;
        }
    }

    // Allocate memory for the temporary device vector that will store the
    // intermediate results of the transform reduce operation.
    temp_dvec.resize(m_path_length, 0.0f);
}

/*
 * Destructor
 */
PathAnalyzer::~PathAnalyzer() {
    // delete m_correlations in reverse order of creation
    for( unsigned int i = 0; i < m_num_correlations; ++i) {
        delete[] m_correlations[i];
    }
    // delete the array of pointers
    delete[] m_correlations;

    delete[] m_functors;
    //CUDA_CALL(cudaFree(m_path_d) );
}

/*
 * Computes the correlation matrix for the current path
 */
void PathAnalyzer::computeCorrelations()
{
    for( unsigned int corr = 0; corr < m_num_correlations; ++corr) {
        for( unsigned int tau = 0; tau < m_max_tau; ++tau) {
            // First transform the data with the correlator functors
            thrust::transform(m_path_devptr, m_path_devptr + m_path_length,
                                m_path_devptr + tau,
                                temp_dvec.begin(), m_functors[corr]);
            // Now do the reduction and save the result
            //     tau-1 since we let tau run from 1 to m_max_tau inclusive
            //std::cerr << temp_dvec[1];
            m_correlations[corr][tau] += (double)thrust::reduce(temp_dvec.begin(), temp_dvec.end(),
                                                          0.0f, thrust::plus<float>())
                                                        / m_path_length;
        }
    }
    m_num_paths_analyzed += 1;

    //thrust::copy( temp_dvec.begin(), temp_dvec.end(), std::ostream_iterator<float>(std::cout, "\n") );
    //std::cout << std::endl;
}

/*
 *
 */
void PathAnalyzer::printCorrelations(std::ostream &ostream)
{
    ostream << std::setprecision(16);
    for( unsigned int tau = 0; tau < m_max_tau; ++tau) {
        for( unsigned int corr = 0; corr < m_num_correlations; ++corr) {
            ostream << std::setw(24)
                    << m_correlations[corr][tau]/m_num_paths_analyzed;
        }
        ostream << std::endl;
    }
}

/*
 *void createFunctors()
 *{
 *    unsigned int index = 0;
 *    for( unsigned int i = 0; i < m_max_power; ++i) {
 *        for( unsigned int j = i; j < m_max_power; ++j) {
 *            m_functors[index++] = correlation_functor(i+1, j+1);
 *        }
 *    }
 *}
 */

void PathAnalyzer::setPathData(float *path_d, unsigned int path_length)
{
    // Copy in the data
    CUDA_CALL(cudaMemcpy( m_path_d, path_d, path_length * sizeof(path_d[0]), cudaMemcpyDeviceToDevice ) );
    // Add the tail for easy correlation calculations
    CUDA_CALL(cudaMemcpy( m_path_d + path_length, path_d, m_max_tau * sizeof(path_d[0]), cudaMemcpyDeviceToDevice ) );
}

// vim: fdm=syntax : tags+=~/.vim/tags/cudacomplete
