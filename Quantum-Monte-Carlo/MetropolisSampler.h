#pragma once

#include <iostream>
#include <curand_kernel.h>

class MetropolisSampler
{
    public:
        MetropolisSampler(float period = 50.0f, unsigned int path_length = 512, float max_jump = 1.0f, unsigned int RNG_seed = 1337);
        ~MetropolisSampler();

        void doRecordAcceptance(bool b) {m_record_acceptance = b;};
        void generateSamplePaths(int num);
        void printData(std::ostream& ostream = std::cout);
        void printPotential(const float* x_grid, unsigned int grid_size, std::ostream& ostream = std::cout );
        void synchronize();

        float getAcceptanceRate();

        float* getPathPtr() const {cudaStreamSynchronize(m_cuda_stream); return m_positions_d; };
        float* getEnergiesPtr() const {cudaStreamSynchronize(m_cuda_stream); return m_energies_d; };

        void fillWithPath(float *path, unsigned int &num_filled);
        void fillWithEnergies(float *energies, unsigned int &num_filled);

        float meanPositionSquared();

    public: // private
        // Class parameters
        unsigned int m_blockDim1D, m_gridDim1D;
        size_t m_shared_mem_size;

        unsigned int m_path_length;
        unsigned int m_num_paths_generated;

        float m_period;
        float m_max_jump;
        float m_epsilon_time;

        curandState* RNG_state_d;
        cudaStream_t m_cuda_stream;

        float* m_positions_d;
        float* m_energies_d;

        unsigned int* m_accepted_counts_d;
        unsigned int m_num_paths_acceptance;

        bool m_record_acceptance;
};
