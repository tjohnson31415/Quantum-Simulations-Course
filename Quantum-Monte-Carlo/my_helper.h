#pragma once

#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/functional.h>


#define CUDA_CALL(x) if((x)!=cudaSuccess) { \
    printf("Error(%d) at %s:%d\n",x,__FILE__,__LINE__); \
    }

#define CUDA_ERRCHK printf(cudaGetErrorString(cudaGetLastError()));\
                    printf("\n");

/* Error checking code for calls to the cuda runtime
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
*/

// Used to get a strided range from an array
// Code copied directly from https://github.com/thrust/thrust/blob/master/examples/strided_range.cu
template <typename Iterator>
class strided_range
{
    public:
        typedef typename thrust::iterator_difference<Iterator>::type difference_type;

        struct stride_functor : public thrust::unary_function<difference_type,difference_type>
        {
            difference_type stride;

            stride_functor(difference_type stride)
                : stride(stride) {}

            __host__ __device__
            difference_type operator()(const difference_type& i) const
            {
                return stride * i;
            }
        };

        typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
        typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
        typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

        // type of the strided_range iterator
        typedef PermutationIterator iterator;

        // construct strided_range for the range [first,last)
        strided_range(Iterator first, Iterator last, difference_type stride)

            : first(first), last(last), stride(stride) {}
        iterator begin(void) const
        {
            return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
        }

        iterator end(void) const
        {
            return begin() + ((last - first) + (stride - 1)) / stride;
        }

    private:
        Iterator first;
        Iterator last;
        difference_type stride;
};

// Prints out a vector stored in device memory to the stream given
template <typename T>
void fprintCudaVector(std::ostream& ostream, T* data_d, size_t num_elements, const std::string& name = "")
{
    T* data_h = (T*) malloc( num_elements * sizeof(T) );
    cudaMemcpy( data_h, data_d, num_elements*sizeof(T), cudaMemcpyDeviceToHost);
    if( name.length() != 0 ) {
        ostream << name << std::endl;
    }
    ostream << std::setprecision(8);
    for(int i = 0; i < num_elements; i++) {
      ostream << std::setw(16);
      ostream << data_h[i] << std::endl;
    }
    ostream << std::endl;

    free( data_h );
}

// Prints out a matrix that is stored in column major format in memory into
// a .csv format text file.
template <typename T>
void fprintCudaMatrix(std::ostream& ostream, T* data_d, size_t num_elems_x, size_t num_elems_y, const std::string& name = "")
{
    T* data_h = (T*) malloc( num_elems_x*num_elems_y * sizeof(T) );
    cudaMemcpy( data_h, data_d, num_elems_x*num_elems_y * sizeof(T), cudaMemcpyDeviceToHost);

    if( name.length() != 0 ) {
        ostream << name << std::endl;
    }

    ostream << std::setprecision(8);
    for(int j = 0; j < num_elems_y; j++) {
        for(int i = 0; i < num_elems_x; i++) {
            ostream << *(data_h + j*num_elems_y + i) << ",  ";
        }
        ostream << std::endl;
    }
    ostream << std::endl;

    free( data_h );
}

// vim: fdm=syntax : ft=cuda : tags+=~/.vim/tags/cudacomplete,~/.vim/tags/glcomplete,~/.vim/tags/cula
