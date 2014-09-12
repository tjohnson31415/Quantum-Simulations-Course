#pragma once

#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

/*
 * Class for a histogram that is stored and computed on the GPU with the thrust
 * library. The histogram will be evenly spaced.
 */
class ThrustHistogram
{
    public:
        ThrustHistogram(float minimum, float maximum, float binwidth);
        ~ThrustHistogram() {};

        //void setPathData(float* data_d, unsigned int num_elements);
        void addData(float* data_d, unsigned int num_elements);

        void getGridArray(float *grid_array, unsigned int &num_elements);

        void printHistogram(std::ostream& ostream = std::cout);
        void printHistogramNormalized(std::ostream& ostream = std::cout);

        void printUnbinned(std::ostream& ostream = std::cout);

    public:
        //float* m_path_data_d;

        float m_minimum;
        float m_maximum;
        float m_binwidth;
        unsigned int m_numbins;

        // Record the number of missed data values below and above the range given
        unsigned int m_num_unbinned_low;
        unsigned int m_num_unbinned_high;

        thrust::device_vector<unsigned int> m_histogram;
};
