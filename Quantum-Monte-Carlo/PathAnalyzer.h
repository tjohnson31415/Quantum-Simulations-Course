#pragma once

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/functional.h>

class PathAnalyzer
{
    public:
        PathAnalyzer(unsigned int path_length, unsigned int max_tau, unsigned int max_power);
        ~PathAnalyzer();

        void computeCorrelations();
        void printCorrelations(std::ostream &ostream = std::cout);

    public: //private
        void setPathData(float *path_d, unsigned int path_length);

        struct correlation_functor : public thrust::binary_function<float,float,float>
        {
            int m_pow1;
            int m_pow2;

            correlation_functor(int pow1 = 1, int pow2 = 1)
                : m_pow1(pow1)
                , m_pow2(pow2)
                  {}

            __host__ __device__
            float operator()(const float &val1, const float &val2) const
            {
                return pow(val1, m_pow1) * pow(val2, m_pow2);
            }
        };

    // Data memebers
    public: //private
        float *m_path_d;
        thrust::device_ptr<float> m_path_devptr;

        thrust::device_vector<float> temp_dvec;

        // An array of pointers to different correlations as a function of time
        double **m_correlations;
        unsigned int m_num_correlations;

        correlation_functor *m_functors;

        // Analyzer parameters
        unsigned int m_max_power;
        unsigned int m_max_tau;
        unsigned int m_path_length;
        unsigned int m_num_paths_analyzed;
};
