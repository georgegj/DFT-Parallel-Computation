#include<iostream>
#include "input_image.cc"
#include "input_image.h"
#include "complex.cc"
#include "complex.h"
#include <thread>
#include <cmath>
#include <string.h>
#include<time.h>
#include<sys/time.h>

void dftr(int tid,float k, int wid, Complex* W, Complex* img, Complex* nimg)
{
for(int j = 0; j<wid; j++)
{
	nimg[wid*tid+j].real=0;
	nimg[wid*tid+j].imag=0;//Complex(0,0);
	for(int i = 0; i<wid; i++)
	{
		nimg[wid*tid+j] = nimg[wid*tid+j]+ W[(i*j)%(wid)]*img[i+wid*tid];

	}		
	nimg[wid*tid+j]=nimg[wid*tid+j]*k;

}
}

void dftc(int tid,float k, int wid, Complex* W, Complex* img, Complex* nimg)
{
for(int j = 0; j<wid; j++)
{
	nimg[tid+wid*j].real=0;
	nimg[tid+wid*j].imag=0;//Complex(0,0);
	for(int i = 0; i<wid; i++)
	{
		nimg[tid+wid*j] = nimg[tid+wid*j]+ W[(i*j)%(wid)]*img[tid+wid*i];

	}		
	nimg[tid+wid*j]=nimg[tid+wid*j]*k;
}
}

int main(int argc, char*argv[])
{
struct timeval begin,end;
gettimeofday(&begin,NULL);

const char* filename;
const char* filedft;
const char* fileidft;
const char* method;

method = argv[1];
filename = argv[2];
filedft = argv[3];
InputImage image(filename);

int Height = image.get_height();
int Width = image.get_width();
std::thread t[Height];
Complex *Image;
Complex *Weight = new Complex[Height];
Complex *nimage = new Complex[Height*Width];
Complex *fimage = new Complex[Height*Width];

int s,d=0;
float m;

if(strcmp(method,"forward")==0)
{
	s=-1;
	m=1;
}

else if(strcmp(method,"reverse")==0)
{
	s=1;
	m=(float(1/float(Height)));
}

for(int i=0;i<Height;i++)
{
	float a = cos((2*PI*i)/Height);
	float b = s*sin((2*PI*i)/Height);	
	Weight[i] = Complex(a,b);
}
			
Image = image.get_image_data();

for (int i=0;i<Height;i++)
{	
	// send the Image row wise, and a new complex variable as reference to get the data evaluated by each thread.  
	t[i] = std::thread(dftr, i, m, Width, Weight, Image, std::ref(nimage));
}

for(int i=0;i<Height;i++)			
t[i].join();

for(int i=0;i<Width;i++)
{
	// compute column wise after evaluating row wise.
	t[i] = std::thread(dftc,i, m, Height, Weight, nimage, std::ref(fimage));
}

for(int i=0;i<Width;i++)
t[i].join();

gettimeofday(&end,NULL);

double time = (end.tv_sec - begin.tv_sec)+(end.tv_usec - begin.tv_usec)/(1e6);	
std::cout<<"Execution time using threads "<<time<<" s\n";

if(strcmp(method,"forward")==0)
	image.save_image_data(filedft,fimage,Width,Height); 
else if(strcmp(method, "reverse")==0)
	image.save_image_data_real(filedft,fimage,Width,Height);


return 0;
}

