#include <string.h> 
#include <math.h> 
#include <stdio.h> 
#include <stdlib.h> 
#ifdef __APPLE__  
	#include <OpenCL/cl.h> 
#else
	#include <CL/cl.h> 
#endif 
#include <clFFT.h> 
#include <sys/types.h> 
#include <sys/stat.h> 
#include <stdint.h> 
#include <float.h> 
  
#define eps_avg 10.0 
  
#define MAX( _a, _b)    ((_a)>(_b)?(_a) : (_b)) 
  
typedef enum { 
    clFFT_OUT_OF_PLACE, 
    clFFT_IN_PLACE, 
}clFFT_TestType; 
  
typedef struct 
{ 
    double real; 
    double imag; 
}clFFT_ComplexDouble; 
  
typedef struct 
{ 
    double *real; 
    double *imag; 
}clFFT_SplitComplexDouble; 
  
cl_device_id     device_id; 
cl_context       context; 
cl_command_queue queue; 
  
// ADDED
void log_error (char* s, ...) {printf ("ERROR: %s\n", s);}
// ADDED
void log_info (char* s, ...) {printf ("INFO: %s\n", s);}

int runTest(clFFT_Dim3 n, int batchSize, clFFT_Direction dir, clFFT_Dimension dim,  
            clFFT_DataFormat dataFormat, int numIter, clFFT_TestType testType) 
{    
    cl_int err = CL_SUCCESS; 
    int iter; 
    double t; 
     
    uint64_t t0, t1; 
    int mx = log2(n.x); 
    int my = log2(n.y); 
    int mz = log2(n.z); 
  
    int length = n.x * n.y * n.z * batchSize; 
    double gflops = 5e-9 * ((double)mx + (double)my + (double)mz) * (double)n.x * (double)n.y * (double)n.z * (double)batchSize * (double)numIter; 
     
    clFFT_SplitComplex data_i_split = (clFFT_SplitComplex) { NULL, NULL }; 
    clFFT_SplitComplex data_cl_split = (clFFT_SplitComplex) { NULL, NULL }; 
    clFFT_Complex *data_i = NULL; 
    clFFT_Complex *data_cl = NULL; 
    clFFT_SplitComplexDouble data_iref = (clFFT_SplitComplexDouble) { NULL, NULL };  
    clFFT_SplitComplexDouble data_oref = (clFFT_SplitComplexDouble) { NULL, NULL }; 
     
    clFFT_Plan plan = NULL; 
    cl_mem data_in = NULL; 
    cl_mem data_out = NULL; 
    cl_mem data_in_real = NULL; 
    cl_mem data_in_imag = NULL; 
    cl_mem data_out_real = NULL; 
    cl_mem data_out_imag = NULL; 
     
    if(dataFormat == clFFT_SplitComplexFormat) { 
        data_i_split.real     = (float *) malloc(sizeof(float) * length); 
        data_i_split.imag     = (float *) malloc(sizeof(float) * length); 
        data_cl_split.real    = (float *) malloc(sizeof(float) * length); 
        data_cl_split.imag    = (float *) malloc(sizeof(float) * length); 
        if(!data_i_split.real || !data_i_split.imag || !data_cl_split.real || !data_cl_split.imag) 
        { 
            err = -1; 
            log_error((char*)"Out-of-Resources\n"); 
            goto cleanup; 
        } 
    } 
    else { 
        data_i  = (clFFT_Complex *) malloc(sizeof(clFFT_Complex)*length); 
        data_cl = (clFFT_Complex *) malloc(sizeof(clFFT_Complex)*length); 
        if(!data_i || !data_cl) 
        { 
            err = -2; 
            log_error((char*)"Out-of-Resouces\n"); 
            goto cleanup; 
        } 
    } 
     
    data_iref.real   = (double *) malloc(sizeof(double) * length); 
    data_iref.imag   = (double *) malloc(sizeof(double) * length); 
    data_oref.real   = (double *) malloc(sizeof(double) * length); 
    data_oref.imag   = (double *) malloc(sizeof(double) * length);   
    if(!data_iref.real || !data_iref.imag || !data_oref.real || !data_oref.imag) 
    { 
        err = -3; 
        log_error((char*)"Out-of-Resources\n"); 
        goto cleanup; 
    } 
  
    int i; 
    if(dataFormat == clFFT_SplitComplexFormat) { 
        for(i = 0; i < length; i++) 
        { 
            data_i_split.real[i] = 2.0f * (float) rand() / (float) RAND_MAX - 1.0f; 
            data_i_split.imag[i] = 2.0f * (float) rand() / (float) RAND_MAX - 1.0f; 
            data_cl_split.real[i] = 0.0f; 
            data_cl_split.imag[i] = 0.0f;            
            data_iref.real[i] = data_i_split.real[i]; 
            data_iref.imag[i] = data_i_split.imag[i]; 
            data_oref.real[i] = data_iref.real[i]; 
            data_oref.imag[i] = data_iref.imag[i];   
        } 
    } 
    else { 

// ADDED
FILE* f = fopen ("test_waveform.dat", "r");
        for(i = 0; i < length; i++) 
        { 
// ADDED
// ADDED
fscanf (f, "%f\n", &data_i[i].real);
data_i[i].imag = 0;
            data_cl[i].real = 0.0f; 
            data_cl[i].imag = 0.0f;          
            data_iref.real[i] = data_i[i].real; 
            data_iref.imag[i] = data_i[i].imag; 
            data_oref.real[i] = data_iref.real[i]; 
            data_oref.imag[i] = data_iref.imag[i];   
        }        
    } 
     
    plan = clFFT_CreatePlan( context, n, dim, dataFormat, &err ); 
    if(!plan || err)  
    { 
        log_error((char*)"clFFT_CreatePlan failed\n"); 
        goto cleanup; 
    } 
     
    if(dataFormat == clFFT_SplitComplexFormat) 
    { 
        data_in_real = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_i_split.real, &err); 
        if(!data_in_real || err)  
        { 
            log_error((char*)"clCreateBuffer failed\n"); 
            goto cleanup; 
        } 
         
        data_in_imag = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_i_split.imag, &err); 
        if(!data_in_imag || err)  
        { 
            log_error((char*)"clCreateBuffer failed\n"); 
            goto cleanup; 
        } 
         
        if(testType == clFFT_OUT_OF_PLACE) 
        { 
            data_out_real = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_cl_split.real, &err); 
            if(!data_out_real || err)  
            { 
                log_error((char*)"clCreateBuffer failed\n"); 
                goto cleanup; 
            } 
             
            data_out_imag = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float), data_cl_split.imag, &err); 
            if(!data_out_imag || err)  
            { 
                log_error((char*)"clCreateBuffer failed\n"); 
                goto cleanup; 
            }            
        } 
        else 
        { 
            data_out_real = data_in_real; 
            data_out_imag = data_in_imag; 
        } 
    } 
    else 
    { 
        data_in = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float)*2, data_i, &err); 
        if(!data_in)  
        { 
            log_error((char*)"clCreateBuffer failed\n"); 
            goto cleanup; 
        } 
        if(testType == clFFT_OUT_OF_PLACE) 
        { 
            data_out = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length*sizeof(float)*2, data_cl, &err); 
            if(!data_out)  
            { 
                log_error((char*)"clCreateBuffer failed\n"); 
                goto cleanup; 
            }            
        } 
        else 
            data_out = data_in; 
    } 
         
    err = CL_SUCCESS; 
    if(dataFormat == clFFT_SplitComplexFormat) 
    { 
        for(iter = 0; iter < numIter; iter++) 
            err |= clFFT_ExecutePlannar(queue, plan, batchSize, dir, data_in_real, data_in_imag, data_out_real, data_out_imag, 0, NULL, NULL); 
    } 
    else 
    { 
        for(iter = 0; iter < numIter; iter++)  
            err |= clFFT_ExecuteInterleaved(queue, plan, batchSize, dir, data_in, data_out, 0, NULL, NULL); 
    } 
     
    err |= clFinish(queue); 
     
    if(err)  
    { 
        log_error((char*)"clFFT_Execute\n"); 
        goto cleanup;    
    } 

    if(dataFormat == clFFT_SplitComplexFormat) 
    {    
        err |= clEnqueueReadBuffer(queue, data_out_real, CL_TRUE, 0, length*sizeof(float), data_cl_split.real, 0, NULL, NULL); 
        err |= clEnqueueReadBuffer(queue, data_out_imag, CL_TRUE, 0, length*sizeof(float), data_cl_split.imag, 0, NULL, NULL); 
	for (int i = 0;  i < length;  i++)
// ADDED
printf ("%3d  %7.3f %7.3f\n", i, data_cl_split.real[i], data_cl_split.imag[i]);
    } 
    else 
    { 
        err |= clEnqueueReadBuffer(queue, data_out, CL_TRUE, 0, length*sizeof(float)*2, data_cl, 0, NULL, NULL); 

	for (int i = 0;  i < length;  i++)
// ADDED
printf ("%3d  %7.3f %7.3f\n", i, data_cl[i].real, data_cl[i].imag);
    } 
     
    if(err)  
    { 
        log_error((char*)"clEnqueueReadBuffer failed\n"); 
        goto cleanup; 
    }    
  
cleanup: 
    clFFT_DestroyPlan(plan);     
    if(dataFormat == clFFT_SplitComplexFormat)  
    { 
        if(data_i_split.real) 
            free(data_i_split.real); 
        if(data_i_split.imag) 
            free(data_i_split.imag); 
        if(data_cl_split.real) 
            free(data_cl_split.real); 
        if(data_cl_split.imag) 
            free(data_cl_split.imag); 
         
        if(data_in_real) 
            clReleaseMemObject(data_in_real); 
        if(data_in_imag) 
            clReleaseMemObject(data_in_imag); 
        if(data_out_real && testType == clFFT_OUT_OF_PLACE) 
            clReleaseMemObject(data_out_real); 
        if(data_out_imag && clFFT_OUT_OF_PLACE) 
            clReleaseMemObject(data_out_imag); 
    } 
    else  
    { 
        if(data_i) 
            free(data_i); 
        if(data_cl) 
            free(data_cl); 
         
        if(data_in) 
            clReleaseMemObject(data_in); 
        if(data_out && testType == clFFT_OUT_OF_PLACE) 
            clReleaseMemObject(data_out); 
    } 
     
    if(data_iref.real) 
        free(data_iref.real); 
    if(data_iref.imag) 
        free(data_iref.imag);        
    if(data_oref.real) 
        free(data_oref.real); 
    if(data_oref.imag) 
        free(data_oref.imag); 
     
    return err; 
} 
  
bool ifLineCommented(const char *line) { 
    const char *Line = line; 
    while(*Line != '\0') 
        if((*Line == '/') && (*(Line + 1) == '/')) 
            return true; 
        else 
            Line++; 
    return false; 
} 
  
cl_device_type getGlobalDeviceType() 
{ 
    char *force_cpu = getenv( "CL_DEVICE_TYPE" ); 
    if( force_cpu != NULL ) 
    { 
        if( strcmp( force_cpu, "gpu" ) == 0 || strcmp( force_cpu, "CL_DEVICE_TYPE_GPU" ) == 0 ) 
            return CL_DEVICE_TYPE_GPU; 
        else if( strcmp( force_cpu, "cpu" ) == 0 || strcmp( force_cpu, "CL_DEVICE_TYPE_CPU" ) == 0 ) 
            return CL_DEVICE_TYPE_CPU; 
        else if( strcmp( force_cpu, "accelerator" ) == 0 || strcmp( force_cpu, "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 ) 
            return CL_DEVICE_TYPE_ACCELERATOR; 
        else if( strcmp( force_cpu, "CL_DEVICE_TYPE_DEFAULT" ) == 0 ) 
            return CL_DEVICE_TYPE_DEFAULT; 
    } 
    // default 
    return CL_DEVICE_TYPE_GPU; 
} 
  
void  
notify_callback(const char *errinfo, const void *private_info, size_t cb, void *user_data) 
{ 
    log_error((char*) "%s\n", errinfo ); 
} 
  
int 
checkMemRequirements(clFFT_Dim3 n, int batchSize, clFFT_TestType testType, cl_ulong gMemSize) 
{ 
    cl_ulong memReq = (testType == clFFT_OUT_OF_PLACE) ? 3 : 2; 
    memReq *= n.x*n.y*n.z*sizeof(clFFT_Complex)*batchSize; 
    memReq = memReq/1024/1024; 
    if(memReq >= gMemSize) 
        return -1; 
    return 0; 
} 
  
int main (int argc, char * const argv[]) { 
    cl_ulong gMemSize; 
    clFFT_Direction dir = clFFT_Forward; 
    int numIter = 1; 
    clFFT_Dim3 n = { 1024, 1, 1 }; 
    int batchSize = 1; 
    clFFT_DataFormat dataFormat = clFFT_SplitComplexFormat; 
    clFFT_Dimension dim = clFFT_1D; 
    clFFT_TestType testType = clFFT_OUT_OF_PLACE; 
    cl_device_id device_ids[16]; 
     
    FILE *paramFile; 
             
    cl_int err; 

    cl_platform_id cpPlatform;
    err = clGetPlatformIDs(1, &cpPlatform, 0);

   unsigned int num_devices; 
     
    cl_device_type device_type = getGlobalDeviceType();  
    if(device_type != CL_DEVICE_TYPE_GPU)  
    { 
        log_info((char*)"Test only supported on DEVICE_TYPE_GPU\n"); 
        exit(0); 
    } 
     
    err = clGetDeviceIDs(cpPlatform, device_type, sizeof(device_ids), device_ids, &num_devices); 
    if(err)  
    {        
        log_error((char*)"clGetComputeDevice failed\n"); 
        return -1; 
    } 
     
    device_id = NULL; 
     
    unsigned int i; 
    for(i = 0; i < num_devices; i++) 
    { 
        cl_bool available; 
        err = clGetDeviceInfo(device_ids[i], CL_DEVICE_AVAILABLE, sizeof(cl_bool), &available, NULL); 
        if(err) 
        { 
             log_error((char*)"Cannot check device availability of device # %d\n", i); 
        } 
         
        if(available) 
        { 
            device_id = device_ids[i]; 
            break; 
        } 
        else 
        { 
            char name[200]; 
            err = clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, sizeof(name), name, NULL); 
            if(err == CL_SUCCESS) 
            { 
                 log_info((char*)"Device %s not available for compute\n", name); 
            } 
            else 
            { 
                 log_info((char*)"Device # %d not available for compute\n", i); 
            } 
        } 
    } 
     
    if(!device_id) 
    { 
        log_error((char*)"None of the devices available for compute ... aborting test\n"); 
        //test_finish(); 
        return -1; 
    } 
     
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err); 
    if(!context || err)  
    { 
        log_error((char*)"clCreateContext failed\n"); 
        //test_finish(); 
        return -1; 
    } 

    queue = clCreateCommandQueue(context, device_id, 0, &err); 
    if(!queue || err) 
    { 
        log_error((char*)"clCreateCommandQueue() failed.\n"); 
        clReleaseContext(context); 
        //test_finish(); 
        return -1; 
    }   
     
    err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &gMemSize, NULL); 
    if(err) 
    { 
        log_error((char*)"Failed to get global mem size\n"); 
        clReleaseContext(context); 
        clReleaseCommandQueue(queue); 
        //test_finish(); 
        return -2; 
    } 
     
    gMemSize /= (1024*1024); 
             
    char delim[] = " \n"; 
    char tmpStr[100]; 
    char line[200]; 
    char *param, *val;   
    int total_errors = 0; 
    if(argc == 1) { 
        log_error((char*)"Need file name with list of parameters to run the test\n"); 
        //test_finish(); 
        return -1; 
    } 
     
    if(argc == 2) { // arguments are supplied in a file with arguments for a single run are all on the same line 
        paramFile = fopen(argv[1], "r"); 
        if(!paramFile) { 
            log_error((char*)"Cannot open the parameter file\n"); 
            clReleaseContext(context); 
            clReleaseCommandQueue(queue);            
            //test_finish(); 
            return -3; 
        } 
        while(fgets(line, 199, paramFile)) { 
            if(!strcmp(line, "") || !strcmp(line, "\n") || ifLineCommented(line)) 
                continue; 
            param = strtok(line, delim); 
            while(param) { 
                val = strtok(NULL, delim); 
                if(!strcmp(param, "-n")) { 
                    sscanf(val, "%d", &n.x); 
                    val = strtok(NULL, delim); 
                    sscanf(val, "%d", &n.y); 
                    val = strtok(NULL, delim); 
                    sscanf(val, "%d", &n.z);                     
                } 
                else if(!strcmp(param, "-batchsize"))  
                    sscanf(val, "%d", &batchSize); 
                else if(!strcmp(param, "-dir")) { 
                    sscanf(val, "%s", tmpStr); 
                    if(!strcmp(tmpStr, "forward")) 
                        dir = clFFT_Forward; 
                    else if(!strcmp(tmpStr, "inverse")) 
                        dir = clFFT_Inverse; 
                } 
                else if(!strcmp(param, "-dim")) { 
                    sscanf(val, "%s", tmpStr); 
                    if(!strcmp(tmpStr, "1D")) 
                        dim = clFFT_1D; 
                    else if(!strcmp(tmpStr, "2D")) 
                        dim = clFFT_2D;  
                    else if(!strcmp(tmpStr, "3D")) 
                        dim = clFFT_3D;                  
                } 
                else if(!strcmp(param, "-format")) { 
                    sscanf(val, "%s", tmpStr); 
                    if(!strcmp(tmpStr, "plannar")) 
                        dataFormat = clFFT_SplitComplexFormat; 
                    else if(!strcmp(tmpStr, "interleaved")) 
                        dataFormat = clFFT_InterleavedComplexFormat;                     
                } 
                else if(!strcmp(param, "-numiter")) 
                    sscanf(val, "%d", &numIter); 
                else if(!strcmp(param, "-testtype")) { 
                    sscanf(val, "%s", tmpStr); 
                    if(!strcmp(tmpStr, "out-of-place")) 
                        testType = clFFT_OUT_OF_PLACE; 
                    else if(!strcmp(tmpStr, "in-place")) 
                        testType = clFFT_IN_PLACE;                                       
                } 
                param = strtok(NULL, delim); 
            } 
             
            if(checkMemRequirements(n, batchSize, testType, gMemSize)) { 
                log_info((char*)"This test cannot run because memory requirements canot be met by the available device\n"); 
                continue; 
            } 
                 
            err = runTest(n, batchSize, dir, dim, dataFormat, numIter, testType); 
            if (err) 
                total_errors++; 
        } 
    } 
     
    clReleaseContext(context); 
    clReleaseCommandQueue(queue); 
     
    return total_errors;         
} 
