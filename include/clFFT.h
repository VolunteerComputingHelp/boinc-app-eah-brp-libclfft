/***************************************************************************
 *   Copyright (C) 2012 by Oliver Bock,Heinz-Bernd Eggenstein              *
 *   oliver.bock[AT]aei.mpg.de                                             *
 *   heinz-bernd.eggenstein[AT]aei.mpg.de                                  *
 *                                                                         *
 *   This file is part of libclfft (originally for Einstein@Home)          *
 *   Derived from clFFT,  (C) Apple, see notice below.                     *
 *                                                                         *
 *                                                                         *
 *   libclfft  is distributed in the hope that it will be useful,          *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See              *
 *   notice below for more details.                                        *
 *                                                                         *
 ***************************************************************************/
//
// File:       clFFT.h
//
// Version:    <1.0>
//
// Disclaimer: IMPORTANT:  This Apple software is supplied to you by Apple Inc. ("Apple")
//             in consideration of your agreement to the following terms, and your use,
//             installation, modification or redistribution of this Apple software
//             constitutes acceptance of these terms.  If you do not agree with these
//             terms, please do not use, install, modify or redistribute this Apple
//             software.
//
//             In consideration of your agreement to abide by the following terms, and
//             subject to these terms, Apple grants you a personal, non - exclusive
//             license, under Apple's copyrights in this original Apple software ( the
//             "Apple Software" ), to use, reproduce, modify and redistribute the Apple
//             Software, with or without modifications, in source and / or binary forms;
//             provided that if you redistribute the Apple Software in its entirety and
//             without modifications, you must retain this notice and the following text
//             and disclaimers in all such redistributions of the Apple Software. Neither
//             the name, trademarks, service marks or logos of Apple Inc. may be used to
//             endorse or promote products derived from the Apple Software without specific
//             prior written permission from Apple.  Except as expressly stated in this
//             notice, no other rights or licenses, express or implied, are granted by
//             Apple herein, including but not limited to any patent rights that may be
//             infringed by your derivative works or by other works in which the Apple
//             Software may be incorporated.
//
//             The Apple Software is provided by Apple on an "AS IS" basis.  APPLE MAKES NO
//             WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED
//             WARRANTIES OF NON - INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
//             PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND OPERATION
//             ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
//
//             IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL OR
//             CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//             SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//             INTERRUPTION ) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION, MODIFICATION
//             AND / OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED AND WHETHER
//             UNDER THEORY OF CONTRACT, TORT ( INCLUDING NEGLIGENCE ), STRICT LIABILITY OR
//             OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright ( C ) 2008 Apple Inc. All Rights Reserved.
//
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __CLFFT_H
#define __CLFFT_H

#include <stdio.h>
#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// XForm type
typedef enum
{
    clFFT_Forward   =   -1,
    clFFT_Inverse   =    1

}clFFT_Direction;

// XForm dimension
typedef enum
{
    clFFT_1D    = 0,
    clFFT_2D    = 1,
    clFFT_3D    = 3

}clFFT_Dimension;

// XForm Data type
typedef enum
{
    clFFT_SplitComplexFormat       = 0,
    clFFT_InterleavedComplexFormat = 1
}clFFT_DataFormat;

// enum for twiddle factor method selection (essentially different methods to evaluate 
// sin(x) and cos(x) on a grid x_k= 2*pi*k/N for some N, k=0..N-1 )
//
// clFFT_native_trig    : the original method, using native_sin, native_cos
//                      : NOTE: precision is hardware dependent, see OpenCL 1.1 spec
// clFFT_sincosfunc     : alternative method using sincos function (slow in most 
//                      : implementations, accuracy as defined in OpenCL 1.1 spec
// clFFT_BigLUT         : alternative version using Lookup tables stored as part of the 
//                      : plan. The LUTs size grow with O(sqrt(N)) , where N is the 
//                      : total size of the transform (over all dimensions). This should 
//                      : be the most accurate option and have much better performance than 
//                      : with option clFFT_sincosfunc
// clFFT_TaylorLUT      : alternative method using a constant size Look-Up-Table and Taylor
//                      : series approx of sin,cos 
// clFFT_RFU{n}         : reserved for future use, so that clFFT_TwiddleFactorMethod may use
//                      : the lower 3 bits of the flags argument to clFFT_CreatePlanAdv. 
//                      : All options are mutually exclusive. 
     

typedef enum 
{
  clFFT_native_trig       = 0,
  clFFT_sincosfunc        = 1,
  clFFT_BigLUT            = 2,
  clFFT_TaylorLUT         = 3,
  clFFT_RFU4              = 4,
  clFFT_RFU5              = 5,   
  clFFT_RFU6              = 6,  
  clFFT_RFU7              = 7    
} clFFT_TwiddleFactorMethod;

typedef struct
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
}clFFT_Dim3;

typedef struct
{
    float *real;
    float *imag;
} clFFT_SplitComplex;

typedef struct
{
    float real;
    float imag;
}clFFT_Complex;

typedef void* clFFT_Plan;

clFFT_Plan clFFT_CreatePlan( cl_context context, clFFT_Dim3 n, clFFT_Dimension dim, clFFT_DataFormat dataFormat, cl_int *error_code );

/**
 *  Extended plan constructor, allows to specify plan options in the flags parameter.  
 *  Currently only values of the enumeration clFFT_TwiddleFactorMethod are supported to 
 *  choose the method for twiddle factor computations. 
 *
 *  Param:
 *        context   : cl_context to use
 *        n         : transform lengths in (up to) 3 dimensions
 *        dim       : dimension of the transform
 *        dataFormat: see  clFFT_DataFormat type
 *        flags     : plan generation options. Use OR to specify more than one option 
 *                    (currently only the mutually exclusive enumeration values of 
 *                    clFFT_TwiddleFactorMethod are supported) 
 *        error_code: pointer to error code, in case of error
 *
 *  Returns:
 *        freshly allocated clFFT_Plan object holding the plan for the specified transform.
 *        Can be reused for several FFT plan executions. 
 *        The caller is responsible to deallocate this object after use.   
 */

clFFT_Plan clFFT_CreatePlanAdv( cl_context context, clFFT_Dim3 n, clFFT_Dimension dim, clFFT_DataFormat dataFormat, unsigned long flags, cl_int *error_code );

void clFFT_DestroyPlan( clFFT_Plan plan );

cl_int clFFT_ExecuteInterleaved( cl_command_queue queue, clFFT_Plan plan, cl_int batchSize, clFFT_Direction dir,
                                 cl_mem data_in, cl_mem data_out,
                                 cl_int num_events, cl_event *event_list, cl_event *event );

cl_int clFFT_ExecutePlannar( cl_command_queue queue, clFFT_Plan plan, cl_int batchSize, clFFT_Direction dir,
                             cl_mem data_in_real, cl_mem data_in_imag, cl_mem data_out_real, cl_mem data_out_imag,
                             cl_int num_events, cl_event *event_list, cl_event *event );

cl_int clFFT_1DTwistInterleaved(clFFT_Plan Plan, cl_command_queue queue, cl_mem array,
                                size_t numRows, size_t numCols, size_t startRow, size_t rowsToProcess, clFFT_Direction dir);


cl_int clFFT_1DTwistPlannar(clFFT_Plan Plan, cl_command_queue queue, cl_mem array_real, cl_mem array_imag,
                            size_t numRows, size_t numCols, size_t startRow, size_t rowsToProcess, clFFT_Direction dir);

void clFFT_DumpPlan( clFFT_Plan plan, FILE *file);

#ifdef __cplusplus
}
#endif

#endif
