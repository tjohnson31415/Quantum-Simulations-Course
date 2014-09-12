#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "ThrustHistogram.h"

ThrustHistogram::ThrustHistogram(float minimum = -1.0f,
                                 float maximum = 1.0f,
                                 float binwidth = 0.05f)
{
    m_minimum = minimum;
    m_binwidth = binwidth;

    m_numbins = ceil((maximum - minimum)/binwidth);
    //m_numbins = numbins;
    m_maximum = m_minimum + m_numbins * binwidth;

    m_num_unbinned_low = 0;
    m_num_unbinned_high = 0;

    // create the thrust device_vector that will store the count data
    // and initialize all values to zero.
    m_histogram.resize(m_numbins, 0);
}

/*
 * Adds the passed in data to the current histogram, the data is copied
 * before any calculations are done so it does not change.
 * Loosely follows the algorithm used in the thrust histogram example:
 *    https://code.google.com/p/thrust/source/browse/examples/histogram.cu
 */
void ThrustHistogram::addData(/*const*/ float* data_d, unsigned int num_elements)
{
    // For use in the lambda expression below
    using namespace thrust::placeholders;

    // First we copy the data from the pointer into a device vector.
    thrust::device_vector<float> new_data(num_elements);
    thrust::copy( thrust::device_ptr<float>(data_d),
                  thrust::device_ptr<float>(data_d + num_elements),
                  new_data.begin());

    // The "remove" algorithm doesn't actually take elements out, it reorders
    // the container so that all the important (non-"removed") elements are
    // accessible by the iterator before the others. Therefore, the remove
    // function returns a new iterator that gives the ending point for the
    // imporant elements. That will be stored in new_end.
    thrust::device_vector<float>::iterator new_end_1;
    thrust::device_vector<float>::iterator new_end;
    // Now remove those uneeded data points
    new_end_1 = thrust::remove_if( new_data.begin(), new_data.end(),
                            (_1 < m_minimum)  );
    // We can calculate the number of removed elements from the new iterator
    m_num_unbinned_low += new_data.end() - new_end_1;

    new_end = thrust::remove_if( new_data.begin(), new_end_1,
                            (_1 > m_maximum)  );

    m_num_unbinned_high += new_end_1 - new_end;

    /*// Rescale the data array according to our histogram*/
    /*thrust::transform(data_ptr, data_ptr + num_elements,*/
                      /*[>output<] data_ptr,*/
                      /*((_1 - m_minimum)/m_binwidth) );*/

    /*// Reassign negative values to a value outside of the range but positive*/
    /*// before attempting to convert from the float to an unsigned int to avoid*/
    /*// possible conversion issues*/
    /*thrust::transform_if(data_ptr, data_ptr + num_elements,*/
                      /*[>output<] new_data.begin(),*/
                      /*((_1 - m_minimum)/m_binwidth) );*/

    // Now we convert the data into a new histogram
    thrust::device_vector<unsigned int> new_histogram(m_numbins);

    // Transform its range so that the integer part is its bin assignment
    thrust::transform(new_data.begin(), new_end,
                      /*output*/ new_data.begin(),
                      ((_1 - m_minimum)/m_binwidth) );

    thrust::sort(new_data.begin(), new_end);

    // Find the end of each bin of values, essentially yielding a cummulative
    // histogram.
    thrust::counting_iterator<unsigned int> search_begin(0);
    thrust::upper_bound(new_data.begin(), new_end,//new_data.end(),
                        search_begin, search_begin + m_numbins,
                        new_histogram.begin());

    // Compute the histogram by taking differences of the cumulative histogram
    thrust::adjacent_difference(new_histogram.begin(), new_histogram.end(),
                                new_histogram.begin());

    // Add the new histogram to the complete histogram
    thrust::transform( m_histogram.begin(), m_histogram.end(),
                       /*input*/new_histogram.begin(),
                       /*output*/m_histogram.begin(),
                       thrust::plus<unsigned int>() );
}

void ThrustHistogram::getGridArray(float *grid_array, unsigned int &num_elements)
{
    if( grid_array == NULL ) {
        std::cerr << "ThrustHistogram::getGridArray : Error - sufficient memory needs to be allocated external to this function";
        std::cerr << std::endl;
    }

    // Fills the grid vector with the bin centers
    for(unsigned int i = 0; i < m_numbins; i++) {
        grid_array[i] = m_minimum + (i+.5f) * m_binwidth;
    }
    num_elements = m_numbins;
}

void ThrustHistogram::printHistogram(std::ostream& ostream)
{
    ostream << std::setprecision(8);
    for(unsigned int i = 0; i < m_numbins; i++) {
        ostream << std::setw(16);
        ostream << m_minimum + (i+.5f) * m_binwidth;
        ostream << std::setw(16);
        ostream << m_histogram[i];
        ostream << std::endl;
    }
    ostream << std::endl;
}

void ThrustHistogram::printHistogramNormalized(std::ostream& ostream)
{
    unsigned int total_counts = thrust::reduce(m_histogram.begin(), m_histogram.end(),
                                               0, thrust::plus<unsigned int>());

    ostream << std::setprecision(8);
    for(unsigned int i = 0; i < m_numbins; i++) {
        ostream << std::setw(16);
        ostream << m_minimum + (i+.5f) * m_binwidth;
        ostream << std::setw(16);
        ostream << float(m_histogram[i])/(total_counts * m_binwidth);
        ostream << std::endl;
    }
    ostream << std::endl;
}

void ThrustHistogram::printUnbinned(std::ostream& ostream)
{
    // Report that nothing has been missed
    if( m_num_unbinned_low + m_num_unbinned_high == 0 ) {
        ostream << "No unbinned histogram data." << std::endl;
        return;
    }
    // Report how much was unbinned and where
    ostream << "Unbinned histogram data: " << std::endl;
    ostream << "  Too low:  " << m_num_unbinned_low << std::endl;
    ostream << "  Too high: " << m_num_unbinned_high << std::endl;
    ostream << "  Total:    " << m_num_unbinned_low + m_num_unbinned_high << std::endl;
    ostream << std::endl;
}

// vim: fdm=syntax : tags+=~/.vim/tags/cudacomplete
