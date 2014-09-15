/*
*  pathIntegrator.cu
*  Copyright (c) 2014 Travis Johnson
*  Distributed under the MIT License
*/

#include <fstream>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cula.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

/* An attempt to make the code compile on Apple computers as well */
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/freeglut.h>
#endif
#include <cuda_gl_interop.h>

#include "my_helper.h"

#define OPENGL 0
#define ENERGY 1
#define NUMSTATS 5

static const unsigned int NUM_D = 1024;
static const unsigned int MAT_SIZE = NUM_D*NUM_D;
static const double XMIN = -6; // Shoule be no less than -8
static const double XMAX = 6; // Shoule be no more than 8
static const double DELTAX = (XMAX - XMIN)/NUM_D;

static const double PI = 3.141592653589793;
static const double PERIOD = 2*PI;

static const unsigned int NUM_T = 256; //256 works alright
static const double EPST = PERIOD/NUM_T;
static const unsigned int EPST_PER_STEP = 1;

static const unsigned int WARPSIZE = 32;


__device__ __constant__ double NUM_D_d = NUM_D;
__device__ __constant__ double XMIN_d = XMIN;
__device__ __constant__ double DELTAX_d = DELTAX;
__device__ __constant__ double PI_d = PI;

// Some global variables for openGL
GLuint gl_vbo;
GLuint gl_vboPotential;
GLuint gl_program;
GLint attrib_coord2d;
GLint uniform_color;
GLint uniform_xscale;
struct cudaGraphicsResource* cuda_vbo_resource; 

// Global pointers for host data
double* x_grid_h;
cuDoubleComplex *wavefunction_h;
cuDoubleComplex *propagator_h;

double *statistics;

// Global Pointers for device data
double* x_grid_d;
cuDoubleComplex *wavefunction_d;
cuDoubleComplex *infintesimal_propagator_d;
cuDoubleComplex *propagator_d;

// Handle to the cublas context that we will create
cublasHandle_t cublasHandle;
int windowID;

///////////////////////////////////////////////////////////////////////////////
// Function prototype forward declarations
///////////////////////////////////////////////////////////////////////////////

void allocateMemory();
void cleanUp();

// CUDA kernels
__global__ void genXGrid(double*);
__global__ void genInitialWavefunction(cuDoubleComplex*, const double*);
__global__ void genInfintesimalPropagator(cuDoubleComplex*, const double*);
__host__ __device__ __inline__ double PotentialEnergy(double);
__host__ __device__ __inline__ cuDoubleComplex complexExponential(double);

// Host routines
void genPropagator(cuDoubleComplex*, cuDoubleComplex*, unsigned int);
void applyPropagator(cuDoubleComplex*, cuDoubleComplex*);
void computeStatistics(double*, double*, cuDoubleComplex*);
void printPropagator();

// Now for opengl and glut
void initGL(int argc, char** argv);
GLuint createProgram();
void display(void);
void keyboard(unsigned char, int, int);
__global__ void genPotentialVBO(float2 *vbo, double *x_grid);
__global__ void updateVBO(float2 *vbo_d, double *x_grid_d, cuDoubleComplex *wavefunction_d);

// Useful Error checking code
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void allocateMemory()
{
    // Allocate Memory
    wavefunction_h = (cuDoubleComplex*) malloc (NUM_D * sizeof(cuDoubleComplex));
    propagator_h = (cuDoubleComplex*) malloc (MAT_SIZE * sizeof(cuDoubleComplex));
    x_grid_h = (double*) malloc (NUM_D * sizeof(double));

    cudaMalloc (&x_grid_d, NUM_D * sizeof(double)); 
    cudaMalloc (&wavefunction_d, NUM_D * sizeof(cuDoubleComplex)); 
    cudaMalloc (&infintesimal_propagator_d, MAT_SIZE * sizeof(cuDoubleComplex)); 
    cudaMalloc (&propagator_d, MAT_SIZE * sizeof(cuDoubleComplex)); 

    statistics = (double*) malloc (NUMSTATS * sizeof(double));

    // We will use cublas to handle all the cuda stuff, so we
    // need to initialize a cublas context first.
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "Could not create cublas handle.\n");
    }
}

void cleanUp()
{
    cublasDestroy(cublasHandle);

    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    glDeleteBuffers(1, &gl_vbo);
    glDeleteBuffers(1, &gl_vboPotential);

    free (wavefunction_h);
    free (x_grid_h);
    free (propagator_h);
    free (statistics);
    cudaFree(x_grid_d);
    cudaFree(wavefunction_d);
    cudaFree(infintesimal_propagator_d);

    cudaDeviceReset();
}

// Function to compute the potential energy on the device
__host__ __device__ __inline__ double PotentialEnergy(double x)
{
    double a4 = 1.0;
    double a2 = -2.0;
    double offset = a2*a2/(4*a4);
    double xsquared = x*x;

    return 0.25*a4*xsquared*xsquared + 0.5*a2*xsquared + offset; // for a double well
    //return .5*x*x; // for a SHO
}

///////////////////////////////////////////////////////////////////////////////
// Functions to generate the initial conditions for the problem on the GPU
///////////////////////////////////////////////////////////////////////////////
__global__ void genXGrid(double *grid)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    grid[i] = XMIN_d + i*DELTAX_d;
}

__global__ void genInitialWavefunction(cuDoubleComplex *result, const double *grid)
{
    const double x_start = 1.414214;
    const double alpha = 4;

    const double norm = pow(alpha/PI_d, .25);
    const double factor = alpha/2;
    double dist;

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    dist = grid[i] - x_start;
    result[i] = make_cuDoubleComplex( norm * exp(-factor* dist*dist ), 0 );
}


// Returns the complex solution of e^(i*x) as a cuDoubleComplex
__host__ __device__ __inline__ cuDoubleComplex complexExponential(double x)
{
    return make_cuDoubleComplex( cos(x), sin(x) );
}

__global__ void genInfintesimalPropagator(cuDoubleComplex *result, const double *grid)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    double tmp = sqrt(1/(4*PI_d*EPST));
    cuDoubleComplex Anorm = make_cuDoubleComplex( tmp, -tmp); 
    cuDoubleComplex tmpC;

    tmp = (grid[i]-grid[j]) / EPST; // the velocity
    // Uses the inline function defined above
    /*
     *tmpC = complexExponential( EPST * ( 0.5 * tmp*tmp
     *                                     //- PotentialEnergy( 0.5*(grid[i]+grid[j]) ) )
     *                                     - 0.5 * (PotentialEnergy(grid[i])+ PotentialEnergy(grid[j]))  )
     *                                     //- PotentialEnergy(grid[i])  )
     *                                    );
     */
    tmpC = complexExponential( EPST * 0.5 * tmp*tmp );
    tmpC = cuCmul(tmpC, complexExponential( -EPST * /*0.5 */ PotentialEnergy( grid[i] ) ) );

    *(result + j*NUM_D + i) = cuCmul(Anorm, tmpC);
}

__global__ void genAnalyticPropagator(cuDoubleComplex *result, const double *grid, double time)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    double tmp = sqrt(1/(4*PI_d*sin(time)));
    cuDoubleComplex Anorm = make_cuDoubleComplex( tmp, -tmp); 
    cuDoubleComplex tmpC;

    tmpC = complexExponential( 1/(2*sin(time)) *( ( grid[j]*grid[j] + grid[i]*grid[i] ) * cos(time)
                                         - 2*grid[i]*grid[j]));

    *(result + j*NUM_D + i) = cuCmul(Anorm, tmpC);
}

// Generate a finite time propagator that is the given power of the infintesimal propagator
void genPropagator(cuDoubleComplex *propagator, cuDoubleComplex *infintesimal_prop, unsigned int power)
{
    const cuDoubleComplex alpha = make_cuDoubleComplex( DELTAX, 0);
    const cuDoubleComplex beta = make_cuDoubleComplex( 0, 0);

    // Quick exit if power is 1
    if (power == 1) {
        cudaMemcpy( propagator, infintesimal_prop, MAT_SIZE * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice );
        return;
    }

    // A temporary variable to store intermediate steps
    cuDoubleComplex *temp_d;
    cudaMalloc (&temp_d, MAT_SIZE * sizeof(cuDoubleComplex)); 

    // Set the propagator equal to the infintesimal propagator
    cudaMemcpy( propagator, infintesimal_prop, MAT_SIZE * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice );

    // We want this code to run power-1 times
    for (int i=1; i < power; i++)
    {
    /*
        cublasStatus_t cublasZgemm(cublasHandle_t handle,
                                cublasOperation_t transa, cublasOperation_t transb,
                                int m, int n, int k,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                const cuDoubleComplex *B, int ldb,
                                const cuDoubleComplex *beta,
                                cuDoubleComplex *C, int ldc)
    */
    cublasZgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    NUM_D, NUM_D, NUM_D,
                    &alpha, 
                    infintesimal_prop, NUM_D,
                    propagator, NUM_D,
                    &beta, 
                    temp_d, NUM_D);

    cudaMemcpy( propagator, temp_d, MAT_SIZE * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    }

    cudaFree( temp_d );
}

/*
__global__ void computeTrace( cuDoubleComplex *result_d, const cuDoubleComplex *matrix_d )
{
    const cuDoubleComplex deltaxC = make_cuDoubleComplex( DELTAX_d, 0);
    *result_d = make_cuDoubleComplex(0,0);
    for (unsigned int j = 0; j < NUM_D; j++) {
        *result_d = cuCadd( *result_d, *(matrix_d + j*(NUM_D+1)) );
    }
    // Multiply by deltaxC since this is an integral
    *result_d = cuCmul(*result_d, deltaxC);
}

__global__ void fillTraceVec( cuDoubleComplex *vec_data_d, const cuDoubleComplex *matrix_d )
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    *(vec_data_d+i) = *(matrix_d + i*(NUM_D+1));
}
*/

// Functor for use below with thrust::reduce
struct cuCadd_functor 
{ 
    __host__ __device__ 
        cuDoubleComplex operator()(const cuDoubleComplex& x, const cuDoubleComplex& y) const 
        { 
            return cuCadd( x, y );
        } 
};

// cuCmul functor to be used with thrust::transform
struct cuCmul_functor 
{ 
    __host__ __device__ 
        cuDoubleComplex operator()(const cuDoubleComplex& x, const cuDoubleComplex& y) const 
        { return cuCmul( x, y ); } 
};

// Computes the trace of the propagator at int*epsilson time steps for int from 1 to numSteps
void computeTraceFFT( cuDoubleComplex *traceFFT_d, cuDoubleComplex *inf_prop_d, unsigned int numSteps )
{
    int step = 0;
    const cuDoubleComplex deltaxC = make_cuDoubleComplex( DELTAX, 0);
    const cuDoubleComplex zeroC = make_cuDoubleComplex( 0, 0);

    // Storage for the trace of the propagator as a function of time
    cuDoubleComplex *trace_h;
    trace_h = (cuDoubleComplex*) malloc( numSteps * sizeof(cuDoubleComplex) );
    // On the device as well for cuFFT
    cuDoubleComplex *trace_d;
    cudaMalloc( &trace_d, numSteps * sizeof(cuDoubleComplex) );

    // The input and output matricies from the gemm operation
    // The output will be an epsilon time step later than the input
    cuDoubleComplex *input_prop_d;
    cuDoubleComplex *output_prop_d;
    cudaMalloc( &input_prop_d, MAT_SIZE * sizeof(cuDoubleComplex) );
    cudaMalloc( &output_prop_d, MAT_SIZE * sizeof(cuDoubleComplex) );

    // Set the input propagator equal to the infintesimal propagator
    cudaMemcpy( input_prop_d, inf_prop_d, MAT_SIZE * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice );


    // Fill the trace_d matrix with the time dependence of the trace
    step = 0;
    do {
        // device ptr to the current propagator matrix
        thrust::device_ptr<cuDoubleComplex> prop_ptr( input_prop_d );
        // custom iterator to stride into the array to get only the diagonal elements
        strided_range< thrust::device_vector<cuDoubleComplex>::iterator >
            trace_vec( prop_ptr, prop_ptr + MAT_SIZE, NUM_D + 1 );

        trace_h[step] =  thrust::reduce( trace_vec.begin(), trace_vec.end(), zeroC, cuCadd_functor() );
        trace_h[step] = cuCmul( trace_h[step], deltaxC );

        // Advance the time by multiplying by the infintesimal propagator
        /*  cublasZgemm calling prototype C = alpha A B + beta C
            cublasStatus_t cublasZgemm(cublasHandle_t handle,
                                    cublasOperation_t transa, cublasOperation_t transb,
                                    int m, int n, int k,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A, int lda,
                                    const cuDoubleComplex *B, int ldb,
                                    const cuDoubleComplex *beta,
                                    cuDoubleComplex *C, int ldc)
        */
        cublasZgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        NUM_D, NUM_D, NUM_D,
                        &deltaxC, 
                        inf_prop_d, NUM_D,
                        input_prop_d, NUM_D,
                        &zeroC, 
                        output_prop_d, NUM_D);

        // Swap the pointers for the two prop buffers
        std::swap( input_prop_d, output_prop_d );

        // Increment the step and continue
        step++;
    } while ( step < numSteps );

    // Copy the trace information back to the device for cuFFT
    cudaMemcpy( trace_d, trace_h, numSteps * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice );

    // Now that we have computed the time dependence of the trace, we want to take the FFT
    cufftHandle plan;

    if( cufftPlan1d( &plan, numSteps, CUFFT_Z2Z, 1) != CUFFT_SUCCESS )
        std::cout << "cuFFT: Plan creation failed." << std::endl;

    if( cufftExecZ2Z( plan, trace_d, traceFFT_d, CUFFT_FORWARD) != CUFFT_SUCCESS )
        std::cout << "cuFFT: Transformation failed." << std::endl;

    cudaDeviceSynchronize();
    cufftDestroy( plan );

    free( trace_h );
    cudaFree( input_prop_d );
    cudaFree( output_prop_d );
    cudaFree( trace_d );
}

cuDoubleComplex analyticTrace( double *x_grid_h, double time )
{
    cuDoubleComplex trace = make_cuDoubleComplex(0,0);
    double tmp = sqrt(1/(4*PI*sin(time)));

    cuDoubleComplex Anorm = make_cuDoubleComplex( tmp, -tmp); 
    cuDoubleComplex tmpC;

    for( int i = 0; i < NUM_D; i++ ) {
        tmpC = complexExponential( 1/(2*sin(time)) * ( 2*x_grid_h[i]*x_grid_h[i] * (cos(time) - 1) ) );
        tmpC = cuCmul(Anorm, tmpC);

        trace = cuCadd( trace, tmpC);
    }

    return trace;
}

void computeTraceFFTAnalytic( cuDoubleComplex *traceFFT_d, double *x_grid_h, unsigned int numSteps )
{
    // Storage for the trace of the propagator as a function of time
    cuDoubleComplex *trace_h;
    trace_h = (cuDoubleComplex*) malloc( numSteps * sizeof(cuDoubleComplex) );
    // On the device as well for cuFFT
    cuDoubleComplex *trace_d;
    cudaMalloc( &trace_d, numSteps * sizeof(cuDoubleComplex) );

    for( int i = 0; i < numSteps; i++ ) {
        trace_h[i] = analyticTrace( x_grid_h, EPST*i );
    }

    // Copy the trace information back to the device for cuFFT
    cudaMemcpy( trace_d, trace_h, numSteps * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice );

    // Now that we have computed the time dependence of the trace, we want to take the FFT
    cufftHandle plan;

    if( cufftPlan1d( &plan, numSteps, CUFFT_Z2Z, 1) != CUFFT_SUCCESS )
        std::cout << "cuFFT: Plan creation failed." << std::endl;

    if( cufftExecZ2Z( plan, trace_d, traceFFT_d, CUFFT_FORWARD) != CUFFT_SUCCESS )
        std::cout << "cuFFT: Transformation failed." << std::endl;

    cudaDeviceSynchronize();
    cufftDestroy( plan );

    free( trace_h );
    cudaFree( trace_d );
}

// Computes the trace of the propagator at multiple times using the
// eigenvalue trick to remove the need for multiplying matricies
void computeTraceFFT2( cuDoubleComplex *traceFFT_d, cuDoubleComplex *inf_prop_d, unsigned int numSteps )
{
    cuDoubleComplex deltaxC = make_cuDoubleComplex( DELTAX, 0);
    cuDoubleComplex zeroC = make_cuDoubleComplex( 0, 0);

    // Storage for the trace of the propagator as a function of time
    cuDoubleComplex *trace_h;
    trace_h = (cuDoubleComplex*) malloc( numSteps * sizeof(cuDoubleComplex) );
    // On the device as well for cuFFT
    cuDoubleComplex *trace_d;
    cudaMalloc( &trace_d, numSteps * sizeof(cuDoubleComplex) );

    // Storage for the eigenvalues of the propagator
    cuDoubleComplex *eigenvalues_d;
    cudaMalloc( &eigenvalues_d, NUM_D * sizeof(cuDoubleComplex) );
    // And for the vector of powers of the eigenvalues
    cuDoubleComplex *eigen_powers_d;
    cudaMalloc( &eigen_powers_d, NUM_D * sizeof(cuDoubleComplex) );
    
    // Use cula library to compute the eigenvalues of the propagator
    culaStatus status = culaInitialize();
    if( status != culaNoError ) {
        printf("%s\n", culaGetStatusString(status));
    }

    status = culaDeviceZgeev('N', 'N', NUM_D, (culaDoubleComplex*) inf_prop_d, NUM_D, (culaDoubleComplex*) eigenvalues_d, NULL, 1, NULL, 1);
    if( status != culaNoError ) {
        if( status == culaDataError )
            printf("Data error with code %d, please see LAPACK documentation\n",culaGetErrorInfo());
        else
            printf("%s\n", culaGetStatusString(status));
    }
    // That's all for cula
    culaShutdown();

    // Duplicate the eigenvalues for the first step in the loop below
    cudaMemcpy( eigen_powers_d, eigenvalues_d, NUM_D * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice );

    thrust::device_ptr<cuDoubleComplex> eigen_powers_ptr( eigen_powers_d );
    thrust::device_ptr<cuDoubleComplex> eigenvalues_ptr( eigenvalues_d );

    thrust::constant_iterator<cuDoubleComplex> deltaxC_iter(deltaxC);

    // Sum the eigen_powers vector to get the trace of the matrix
    //  The first element is just the sum of the eigenvalues
    trace_h[0] = thrust::reduce( eigen_powers_ptr, eigen_powers_ptr + NUM_D, zeroC, cuCadd_functor() );
    trace_h[0] = cuCmul( trace_h[0], deltaxC );
    for( int i = 1; i < numSteps; i++)
    {
        // Multiply the current trace vector by the eigenvalues to get (lambda_i)^(i+1)
        thrust::transform( eigen_powers_ptr, eigen_powers_ptr + NUM_D, eigenvalues_ptr, 
                /*output*/ eigen_powers_ptr, cuCmul_functor() );
        // Multiply all elements by deltaxC since the above operation is a simplified
        //  form of the integration
        thrust::transform( eigen_powers_ptr, eigen_powers_ptr + NUM_D, deltaxC_iter, 
                /*output*/ eigen_powers_ptr, cuCmul_functor() );

        // Sum the eigen_powers vector to get the trace of the matrix
        trace_h[i] =  thrust::reduce( eigen_powers_ptr, eigen_powers_ptr + NUM_D, zeroC, cuCadd_functor() );
        // Multiply by deltaX since this was an integration
        trace_h[i] = cuCmul( trace_h[i], deltaxC );
    }

    // Copy the trace information back to the device for cuFFT
    cudaMemcpy( trace_d, trace_h, numSteps * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice );
    fprintCudaVector( std::cout, trace_d, numSteps );
    

    // Now that we have computed the time dependence of the trace, we want to take the FFT
    cufftHandle plan;

    if( cufftPlan1d( &plan, numSteps, CUFFT_Z2Z, 1) != CUFFT_SUCCESS )
        std::cout << "cuFFT: Plan creation failed." << std::endl;

    if( cufftExecZ2Z( plan, trace_d, traceFFT_d, CUFFT_FORWARD) != CUFFT_SUCCESS )
        std::cout << "cuFFT: Transformation failed." << std::endl;

    cudaDeviceSynchronize();
    cufftDestroy( plan );

    free( trace_h );
    cudaFree( trace_d );
    cudaFree( eigenvalues_d );
}

void computeTrace( cuDoubleComplex *result_d, cuDoubleComplex *inf_prop_d, unsigned int numSteps )
{
    if( result_d == NULL ) {
        cudaMalloc( &result_d, numSteps * sizeof(cuDoubleComplex) );
    }

    cuDoubleComplex deltaxC = make_cuDoubleComplex( DELTAX, 0);
    cuDoubleComplex zeroC = make_cuDoubleComplex( 0, 0);

    // Storage for the trace of the propagator as a function of time
    cuDoubleComplex *trace_h;
    trace_h = (cuDoubleComplex*) malloc( numSteps * sizeof(cuDoubleComplex) );

    // Storage for the eigenvalues of the propagator
    cuDoubleComplex *eigenvalues_d;
    cudaMalloc( &eigenvalues_d, NUM_D * sizeof(cuDoubleComplex) );
    // And for the vector of powers of the eigenvalues
    cuDoubleComplex *eigen_powers_d;
    cudaMalloc( &eigen_powers_d, NUM_D * sizeof(cuDoubleComplex) );
    
    // Use cula library to compute the eigenvalues of the propagator
    culaStatus status = culaInitialize();
    if( status != culaNoError ) {
        printf("cula failed to initialize: %s\n", culaGetStatusString(status));
    }

    status = culaDeviceZgeev('N', 'N', NUM_D, (culaDoubleComplex*) inf_prop_d, NUM_D, (culaDoubleComplex*) eigenvalues_d, NULL, 1, NULL, 1);
    if( status != culaNoError ) {
        if( status == culaDataError )
            printf("Data error with code %d, please see LAPACK documentation\n",culaGetErrorInfo());
        else
            printf("%s\n", culaGetStatusString(status));
    }
    // That's all for cula
    culaShutdown();

    // Duplicate the eigenvalues for the first step in the loop below
    cudaMemcpy( eigen_powers_d, eigenvalues_d, NUM_D * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice );

    thrust::device_ptr<cuDoubleComplex> eigen_powers_ptr( eigen_powers_d );
    thrust::device_ptr<cuDoubleComplex> eigenvalues_ptr( eigenvalues_d );

    thrust::constant_iterator<cuDoubleComplex> deltaxC_iter(deltaxC);

    // Sum the eigen_powers vector to get the trace of the matrix
    //  The first element is just the sum of the eigenvalues
    trace_h[0] = thrust::reduce( eigen_powers_ptr, eigen_powers_ptr + NUM_D, zeroC, cuCadd_functor() );
    trace_h[0] = cuCmul( trace_h[0], deltaxC );
    for( int i = 1; i < numSteps; i++)
    {
        // Multiply the current trace vector by the eigenvalues to get (lambda_i)^(i+1)
        thrust::transform( eigen_powers_ptr, eigen_powers_ptr + NUM_D, eigenvalues_ptr, 
                /*output*/ eigen_powers_ptr, cuCmul_functor() );
        // Multiply all elements by deltaxC since the above operation is a simplified
        //  form of the integration
        thrust::transform( eigen_powers_ptr, eigen_powers_ptr + NUM_D, deltaxC_iter, 
                /*output*/ eigen_powers_ptr, cuCmul_functor() );

        // Sum the eigen_powers vector to get the trace of the matrix
        trace_h[i] =  thrust::reduce( eigen_powers_ptr, eigen_powers_ptr + NUM_D, zeroC, cuCadd_functor() );
        // Multiply by deltaX since this was an integration
        trace_h[i] = cuCmul( trace_h[i], deltaxC );
    }

    // Copy the trace information back to the device to be returned by this function
    cudaMemcpy( result_d, trace_h, numSteps * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice );

    free( trace_h );
    cudaFree( eigenvalues_d );
    cudaFree( eigen_powers_d );
}

void computeFFT(cuDoubleComplex *result_d, cuDoubleComplex *data_d, size_t length, int direction)
{
    if( result_d == NULL ) {
        cudaMalloc( &data_d, length * sizeof(cuDoubleComplex) );
    }

    cufftHandle plan;

    if( cufftPlan1d( &plan, length, CUFFT_Z2Z, 1) != CUFFT_SUCCESS ) {
       std::cout << "cuFFT: Plan creation failed." << std::endl;
       return;
    }

    if( cufftExecZ2Z( plan, (cufftDoubleComplex*)data_d, (cufftDoubleComplex*)result_d, direction) != CUFFT_SUCCESS ) {
        std::cout << "cuFFT: Transformation failed." << std::endl;
    }

    cudaDeviceSynchronize();
    cufftDestroy( plan );
}

// Apply the propagator matrix to the wavefunction
//TODO Might be better to have a pointer to a scratch buffer passed into this function
//      as well so that it doesn't allocate and destroy new memory each time.
void applyPropagator(cuDoubleComplex *wavefunction, cuDoubleComplex *propagator)
{
    const cuDoubleComplex deltaxC = make_cuDoubleComplex( DELTAX, 0);
    const cuDoubleComplex zeroC  = make_cuDoubleComplex( 0, 0);
    // A temporary variable to store intermediate steps
    cuDoubleComplex *temp_d;
    cudaMalloc (&temp_d, NUM_D*sizeof(cuDoubleComplex)); 

    /*
        cublasStatus_t cublasZgemv(cublasHandle_t handle, cublasOperation_t trans,
                                int m, int n,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                const cuDoubleComplex *x, int incx,
                                const cuDoubleComplex *beta,
                                cuDoubleComplex *y, int incy)
    */

    cublasZgemv( cublasHandle, CUBLAS_OP_N,
                    NUM_D, NUM_D,
                    &deltaxC,
                    propagator, NUM_D,
                    wavefunction, 1,
                    &zeroC, 
                    temp_d, 1);

    cudaMemcpy( wavefunction, temp_d, NUM_D * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

    cudaFree( temp_d );
}

///////////////////////////////////////////////////////////////////////////////
// OpenGL related functions
///////////////////////////////////////////////////////////////////////////////
void initGL(int argc, char** argv)
{
    GLint window_height = 512;
    GLint window_width = 512;

    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( 512, 512);
    windowID = glutCreateWindow("Wavefunction in a Harmonic Potential");

    std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "GL Version: " << glGetString(GL_VERSION) << std::endl;
    // Register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);

    // Initialize OpenGL extensions with glew if available
#if defined(__APPLE__) || defined(MACOSX)
    // Don't load glew
#else
    glewInit();
#endif

    // Initialize some defaults
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glViewport(0,0, window_width, window_height);

    // Create the vertex buffer objects that we will be using
    glGenBuffers(1, &gl_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, gl_vbo);
    glBufferData(GL_ARRAY_BUFFER, 3*NUM_D*sizeof(float2), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &gl_vboPotential);
    glBindBuffer(GL_ARRAY_BUFFER, gl_vboPotential);
    glBufferData(GL_ARRAY_BUFFER, NUM_D*sizeof(float2), NULL, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Create the program that is used to render the lines to the screen
    gl_program = createProgram();
    attrib_coord2d = glGetAttribLocation( gl_program, "coord2d");
    uniform_color  = glGetUniformLocation( gl_program, "f_color");
    uniform_xscale = glGetUniformLocation( gl_program, "xscale");

    // Default values for uniforms
    glProgramUniform1f(gl_program, uniform_xscale, XMAX);
    glProgramUniform4f(gl_program, uniform_color, 0.0f, 0.0f, 0.0f, 0.0f);

    // Now set up interop between CUDA and OpenGL
    cudaGLSetGLDevice(0);
    // First fill the potential VBO
    float2 *vboPotential_ptr;
    size_t vbo_data_size;
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, gl_vboPotential, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsMapResources( 1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer( (void**)&vboPotential_ptr, &vbo_data_size, cuda_vbo_resource);

    genPotentialVBO <<<NUM_D/WARPSIZE, WARPSIZE>>> (vboPotential_ptr, x_grid_d);
    // We will not need access to this resource again
    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    //cudaGraphicsUnregisterResource(

    // Register the wavefunction vbo that will be used for the rest of the program
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, gl_vbo, cudaGraphicsMapFlagsWriteDiscard);
}

GLuint createProgram()
{
    const GLchar *vertex_shader_src = 
    {
        "#version 120\n"
        "attribute vec2 coord2d;\n"
        "uniform float xscale;\n"
        //"varying vec4 f_color;\n"
        "\n"
        "void main(void) {\n"
        "   gl_Position = vec4(coord2d.x/xscale, coord2d.y - 0.5, 0, 1);\n"
        //"   f_color = vec4(coord2d.xy / 2.0 + 0.5, 1, 1);\n"
        "}"
    };

    const GLchar *fragment_shader_src = 
    {
        "#version 120\n"
        //"varying vec4 f_color;\n"
        "uniform vec4 f_color;\n"
        "\n"
        "void main(void) {\n"
        "   gl_FragColor = f_color;\n"
        "}"
    };

    GLuint program;
    GLuint Vshader, Fshader;
    GLint check_status = GL_FALSE;
    //const GLchar *vertex_sources[] = {vertex_shader_src};

    program = glCreateProgram();

    Vshader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(Vshader, 1, &vertex_shader_src, NULL);
    glCompileShader(Vshader);
    glGetShaderiv(Vshader, GL_COMPILE_STATUS, &check_status);
    if (check_status == GL_FALSE)
    {
        fprintf( stderr, "Vertex shader did not compile.\n");
        return 0;
    }
    glAttachShader(program, Vshader);

    Fshader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(Fshader, 1, &fragment_shader_src, NULL);
    glCompileShader(Fshader);
    glGetShaderiv(Fshader, GL_COMPILE_STATUS, &check_status);
    if (check_status == GL_FALSE)
    {
        fprintf( stderr, "Fragment shader did not compile.\n");
        return 0;
    }
    glAttachShader(program, Fshader);

    glLinkProgram( program);
    glGetProgramiv( program, GL_LINK_STATUS, &check_status);
    if (check_status == GL_FALSE)
    {
        fprintf( stderr, "Program failed to link.\n");
        return 0;
    }

    return program;
}

__global__ void genPotentialVBO(float2 *vbo, double *x_grid)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    vbo[i].x = x_grid[i];
    vbo[i].y = PotentialEnergy( x_grid[i] );
}

__global__ void updateVBO(float2 *vbo, double *x_grid, cuDoubleComplex *wavefunction)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float tmp = cuCabs( wavefunction[i]);

    vbo[i].x = x_grid[i];
    vbo[i].y = tmp*tmp;
}

void display()
{
    float2* vbo_data_d;
    size_t vbo_data_size;

    // Update that data in the VBO
    cudaGraphicsMapResources( 1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer( (void**)&vbo_data_d, &vbo_data_size, cuda_vbo_resource);
    updateVBO <<<NUM_D/WARPSIZE, WARPSIZE>>> (vbo_data_d, x_grid_d, wavefunction_d);
    cudaGraphicsUnmapResources( 1, &cuda_vbo_resource, 0);

    // Set every pixel in the frame buffer to the current clear color.
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_LINE_SMOOTH);
    glUseProgram( gl_program);

    glBindBuffer(GL_ARRAY_BUFFER, gl_vboPotential);
    glEnableVertexAttribArray( attrib_coord2d);
    glVertexAttribPointer( attrib_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glLineWidth( 2.0f);
    glProgramUniform4f(gl_program, uniform_color, 1.0f, 0.0f, 0.0f, 1.0f);
    glDrawArrays( GL_LINE_STRIP, 0, NUM_D);

    glBindBuffer(GL_ARRAY_BUFFER, gl_vbo);
    glEnableVertexAttribArray( attrib_coord2d);
    glVertexAttribPointer( attrib_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glLineWidth( 2.0f);
    glProgramUniform4f(gl_program, uniform_color, 0.0f, 1.0f, 0.0f, 1.0f);
    glDrawArrays( GL_LINE_STRIP, 0, NUM_D);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisable(GL_LINE_SMOOTH);

    glutSwapBuffers();
    // Update the system
    applyPropagator( wavefunction_d, propagator_d);

    glutPostRedisplay();
}

void keyboard( unsigned char key, int x, int y)
{
    switch( key) {
    case 32: // Spacebar
        glutPostRedisplay();
        break;
    case 27: // Escape key
        glutDestroyWindow( windowID );
        cleanUp();
        exit(0);
        break;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Global functions to compute statistics for the wavefunction
///////////////////////////////////////////////////////////////////////////////
__global__ void computeNormVec(double *data, double *x_grid, cuDoubleComplex *wavefunction)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    double tmp = cuCabs( wavefunction[i]);
    tmp = tmp*tmp;
    data[i] = tmp * DELTAX_d;
}

__global__ void computeMeanPositionVec(double *data, double *x_grid, cuDoubleComplex *wavefunction)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    double tmp = cuCabs( wavefunction[i]);
    tmp = tmp*tmp;
    data[i] = tmp * x_grid[i] * DELTAX_d;
}

__global__ void computeMeanPotentialEnergyVec(double *data, double *x_grid, cuDoubleComplex *wavefunction)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    double tmp = cuCabs( wavefunction[i]);
    tmp = tmp*tmp;
    data[i] = tmp * PotentialEnergy(x_grid[i]) * DELTAX_d;
}

__global__ void computeMeanKineticEnergyVec(double *data, double *x_grid, cuDoubleComplex *wavefunction)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    cuDoubleComplex tmpComplex;
    double tmp;

    if ( i == NUM_D-1 )
        tmpComplex = cuCsub( wavefunction[0], wavefunction[i] ); // Periodic boundary condition
    else
        tmpComplex = cuCsub( wavefunction[i+1], wavefunction[i] );

    tmp = cuCabs( tmpComplex);
    tmp = tmp*tmp;

    data[i] = 0.5* tmp / DELTAX_d;
}

/*
void computeStatistics(double *stats, double *x_grid, cuDoubleComplex *wavefunction)
{
    unsigned int block_size_1D = WARPSIZE;
    unsigned int grid_size_1D = ceil(NUM_D/WARPSIZE);
    double *data;

    thrust::device_vector<double> vec(NUM_D);
    data = thrust::raw_pointer_cast( vec.data() );

    computeNormVec <<<grid_size_1D, block_size_1D>>>( data, x_grid, wavefunction );
    stats[0] = thrust::reduce( vec.begin(), vec.end(), (double) 0, thrust::plus<double>() );

    computeMeanPositionVec <<<grid_size_1D, block_size_1D>>>( data, x_grid, wavefunction );
    stats[1] = thrust::reduce( vec.begin(), vec.end(), (double) 0, thrust::plus<double>() );

    computeMeanPotentialEnergyVec <<<grid_size_1D, block_size_1D>>>( data, x_grid, wavefunction );
    stats[2] = thrust::reduce( vec.begin(), vec.end(), (double) 0, thrust::plus<double>() );

    computeMeanKineticEnergyVec <<<grid_size_1D, block_size_1D>>>( data, x_grid, wavefunction );
    stats[3] = thrust::reduce( vec.begin(), vec.end(), (double) 0, thrust::plus<double>() );

    stats[4] = stats[2] + stats[3]; // Total energy
}
*/

struct cuCabsSquared_functor
{
    __host__ __device__
        double operator()(const cuDoubleComplex& x) const 
        {
            double tmp = cuCabs(x);
            return tmp*tmp;
        }
};

struct potentialEnergy_functor 
{ 
    __host__ __device__ 
        double operator()(const double& psi_squared, const double& x_pos) const 
        {return psi_squared * PotentialEnergy(x_pos);} 
};

void computeStatistics(double *stats, double *x_grid, cuDoubleComplex *wavefunction)
{
    unsigned int block_size_1D = WARPSIZE;
    unsigned int grid_size_1D = ceil(NUM_D/WARPSIZE);
    thrust::device_ptr<cuDoubleComplex> wavefunction_ptr( wavefunction );
    // device vector to store psi_squared
    thrust::device_vector<double> psi_squared(NUM_D);

    thrust::device_vector<double> vec(NUM_D);
    double *data = thrust::raw_pointer_cast( vec.data() );

    // Compute the absSquared of the wavefunction and store in the vector psi_squared
    thrust::transform( wavefunction_ptr, wavefunction_ptr + NUM_D, psi_squared.begin(), cuCabsSquared_functor() );

    // The norm of the vector from a simple plus reduction
    stats[0] = DELTAX * thrust::reduce( psi_squared.begin(), psi_squared.end(), (double) 0, thrust::plus<double>() );
    // Compute the average position of the particle by multipling the grid value by psi_squared
    thrust::transform( psi_squared.begin(), psi_squared.end(), thrust::device_ptr<double>(x_grid), 
            /*output*/vec.begin(), thrust::multiplies<double>() );
    stats[1] = DELTAX * thrust::reduce( vec.begin(), vec.end(), (double) 0, thrust::plus<double>() );
    // Compute the average potential using the potentialEnergy functor to compute its value at each grid point
    thrust::transform( psi_squared.begin(), psi_squared.end(), thrust::device_ptr<double>(x_grid), 
            /*output*/vec.begin(), potentialEnergy_functor() );
    stats[2] = DELTAX * thrust::reduce( vec.begin(), vec.end(), (double) 0, thrust::plus<double>() );

    // TODO Figure out how to calculate the kinetic energy with just thrust as well
    //  it is a bit harder to deal with the derivatives though
    computeMeanKineticEnergyVec <<<grid_size_1D, block_size_1D>>>( data, x_grid, wavefunction );
    stats[3] = thrust::reduce( vec.begin(), vec.end(), (double) 0, thrust::plus<double>() );

    stats[4] = stats[2] + stats[3]; // Total energy
}

///////////////////////////////////////////////////////////////////////////////
// Main Function
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    size_t blockEdge = 16;
    dim3 block_size;
    dim3 grid_size;

    unsigned int block_size_1D;
    unsigned int grid_size_1D;

    // Allocate all the memory we will use
    allocateMemory();

    cuDoubleComplex *analytic_propagator_d;
    cudaMalloc( (void**) &analytic_propagator_d, MAT_SIZE * sizeof(cuDoubleComplex) );

    // Set grid and block sizes
    block_size.x = blockEdge ;
    block_size.y = blockEdge;
    block_size.z = 1;

    grid_size.x = NUM_D/blockEdge;
    grid_size.y = NUM_D/blockEdge;
    grid_size.z = 1;

    block_size_1D = WARPSIZE;
    grid_size_1D = NUM_D/WARPSIZE;

    // Populate the initial arrays
    genXGrid <<<grid_size_1D, block_size_1D>>> (x_grid_d);
    genInitialWavefunction <<<grid_size_1D, block_size_1D>>> (wavefunction_d, x_grid_d);

    genInfintesimalPropagator <<<grid_size, block_size>>> (infintesimal_propagator_d, x_grid_d);
    genAnalyticPropagator <<<grid_size, block_size>>> (analytic_propagator_d, x_grid_d, EPST);

    // Compute the finite step propagator that we will use.
    genPropagator (propagator_d, infintesimal_propagator_d, EPST_PER_STEP);

#if ENERGY
    unsigned int numSteps = 256*256;

    cuDoubleComplex *trace_d;
    cudaMalloc( &trace_d, numSteps * sizeof(cuDoubleComplex) );

    cuDoubleComplex *trace_fft_d;
    cudaMalloc( &trace_fft_d, numSteps * sizeof(cuDoubleComplex) );

    cuDoubleComplex *trace_fft_h;
    trace_fft_h = (cuDoubleComplex*) malloc( numSteps * sizeof(cuDoubleComplex) );

    //computeTrace( trace_d, analytic_propagator_d, numSteps);
    computeTrace( trace_d, infintesimal_propagator_d, numSteps);
    computeFFT( trace_fft_d, trace_d, numSteps, CUFFT_INVERSE );

    //cudaMemcpy( x_grid_h, x_grid_d, NUM_D * sizeof(double), cudaMemcpyDeviceToHost );
    //computeTraceFFTAnalytic( trace_fft_d, x_grid_h, numSteps);

    //fprintCudaVector( std::cout, trace_d, numSteps, "Trace" );
    //fprintCudaVector( std::cout, trace_fft_d, numSteps, "Trace FFT" );

    //computeStatistics(statistics, x_grid_d, wavefunction_d);


    cudaMemcpy(trace_fft_h, trace_fft_d, numSteps*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    std::cout << std::setprecision(8);
    for(int i=0; i < numSteps; i++) {
        std::cout << std::setw(16) << ((double)i)*NUM_T/numSteps << std::setw(16) << cuCabs(trace_fft_h[i])/numSteps << std::endl;
    }
#elif OPENGL
    // Now set up the openGL and glut context and everything.
    initGL(argc, argv);
    // Set this option so that execution continues when the window is closed
    glutSetOption( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
    glutMainLoop();

#else
    std::ofstream outFile;
    outFile.open("statistics.out");
    // Or just run everything in our own loop to skip using openGL
    //printPropagator();

    outFile << std::setprecision(8);
    int i = 0;
    do {
        computeStatistics(statistics, x_grid_d, wavefunction_d);

        outFile << std::setw(16) << i*EPST*EPST_PER_STEP;
        for (int j=0; j < NUMSTATS; j++) {
            outFile << std::setw(16) << statistics[j];
        }
        outFile << std::endl;

        applyPropagator( wavefunction_d, propagator_d);
        i++;

    } while ( i*EPST*EPST_PER_STEP <= 8*2*PI);

    outFile << std::endl;

    outFile.close();

    /*
    std::ofstream analytic_file;
    analytic_file.open("analytic.csv");
    fprintCudaMatrix( analytic_file, analytic_propagator_d, NUM_D, NUM_D, "Analytic Propagator");
    analytic_file.close();

    std::ofstream propagator_file;
    propagator_file.open("propagator.csv");
    fprintCudaMatrix( propagator_file, infintesimal_propagator_d, NUM_D, NUM_D, "Infinitesimal Propagator");
    propagator_file.close();
    */
#endif
}

// vim: fdm=syntax : tags+=~/.vim/tags/cudacomplete,~/.vim/tags/glcomplete,~/.vim/tags/cula
