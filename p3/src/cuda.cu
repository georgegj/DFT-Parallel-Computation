#include<iostream>
#include<stdio.h>
#include "input_image.cu"
#include "input_image.cuh"
#include "complex.cu"
#include "complex.cuh"
#include<cmath>
#include<string.h>
#include<time.h>
#include<sys/time.h>
#define THREADS_PER_BLOCK 512

__global__ void DFT(Complex *IM_d, Complex *NI_d, Complex *FI_d, int Height, int Width, Complex *Weight,float m )
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < Width)
	{
	int count = index%Width;
	
	for (int j=0; j<Width; j++)
	{
		NI_d[Width*count +j].real=0;
		NI_d[Width*count +j].imag=0;
		for(int i = 0;i<Width;i++)
		{
			NI_d[Width*count +j] = NI_d[Width*count + j] + Weight[(i*j)%Width]*IM_d[i + Width*count];	
		}
	NI_d[Width*count + j] = NI_d[Width*count + j]*m;
	}
	
	__syncthreads();

	for(int j =0; j<Width; j++)
        {
                FI_d[count + Width*j].real = 0;
                FI_d[count + Width*j].imag = 0;

                for(int i=0; i<Width; i++)
                {
                        FI_d[count + Width*j] = FI_d[count + Width*j] + Weight[(i*j)%Width]*NI_d[i*Width + count];

                }
        FI_d[count + Width*j] = FI_d[count + Width*j]*m;
        }
	

}		

}	


int main(int argc, char*argv[])
{

	struct timeval begin, end;
	gettimeofday (&begin, NULL);
	
	const char *filename,*filename2,*method;
	method = argv[1];
	filename  = argv[2];
	filename2 = argv[3];
	
	InputImage image(filename);
	
	int Width = image.get_width();
	int Height = image.get_height();
	int size = Height * Width * sizeof(Complex);
        int s;
	float m;
	
	Complex *IM,*NI,*FI,*Weight;
	Complex *IM_d, *NI_d, *FI_d, *Weight_d;
	IM = image.get_image_data();
	NI = (Complex *)malloc(size);
	FI = (Complex *)malloc(size);
	Weight = (Complex *)malloc(size);
	
	cudaMalloc((void **)&NI_d, size);
	cudaMalloc((void **)&FI_d, size);
	cudaMalloc((void **)&IM_d, size);
	cudaMalloc((void **)&Weight_d,size);
	cudaMemcpy(IM_d, IM, size, cudaMemcpyHostToDevice);
	cudaMemcpy(NI_d, NI, size, cudaMemcpyHostToDevice);
	cudaMemcpy(FI_d, FI, size, cudaMemcpyHostToDevice);
	
	if(strcmp(method,"forward") == 0)
	{
		s = -1;
		m = 1;
	}
	
	else if(strcmp(method,"reverse")==0)
	{
		s = 1;
		m = (1/(float(Height)));
	}
	
	else
	{
		std::cout<<"Incorrect Transform Specified\n";
		return -1;
	}
	
	for(int i=0; i<Height;i++)
	{
		float a = cos((2*PI*i)/Height);
		float b = s*sin((2*PI*i)/Height);
		Weight[i] = Complex(a,b);
	}

	cudaMemcpy(Weight_d, Weight, size, cudaMemcpyHostToDevice);	
	
	DFT<<<(Height*Width + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(IM_d, NI_d, FI_d, Height,Width,Weight_d,m);
	
	cudaMemcpy(IM, IM_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(NI, NI_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(FI, FI_d, size, cudaMemcpyDeviceToHost);

	gettimeofday (&end, NULL);
	double time = (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
	std::cout << "Execution time using cuda " << time << " s\n";

	if(strcmp(method,"forward")==0)
		image.save_image_data(filename2,FI,Width,Height);
	else if(strcmp(method,"reverse")==0)
		image.save_image_data_real(filename2,FI,Width,Height);

	
	
	cudaFree(NI_d);
	cudaFree(FI_d);
	cudaFree(IM_d);
	cudaFree(Weight_d);
	free(IM);
	free(NI);
	free(FI);
	free(Weight);
	
return 0;
}

