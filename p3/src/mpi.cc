#include <iostream>
#include <fstream>
#include "mpi.h"
#include<stdlib.h>
#include <string>
#include "complex.cc"
#include "input_image.cc"
#include <math.h>
#include <chrono>


using namespace std;

//Inverse FFT
void IFFT_1D(Complex* H, Complex* h, int n)
{
  for ( int j = 0; j < n; ++j )
  {
	for ( int i = 0; i < n; ++i )
	{
		H[j] = H[j] + Complex(cos(2*M_PI * i * j / n), sin(2*M_PI * i * j / n)) * h[i];
	}
	H[j] = H[j]*(1.0/n);
  }
}

//Bit Reverse to use in Cooley Tukey FFT
unsigned int bitReverse(unsigned int x, int log2n) 
{ 
    int n = 0; 
    for (int i = 0; i < log2n; i++) 
    { 
        n <<= 1; 
        n |= (x & 1); 
        x >>= 1; 
    } 
    return n; 
} 
  
//1D Cooley Tukey FFT - Iterative Function
void fft(Complex* H, Complex* h, int n, int log2n) 
{  
  
	//Rearrange the ordering of input numbers for Cooley Tukey FFT
    for (unsigned int i = 0; i < n; ++i) { 
        int rev = bitReverse(i, log2n); 
        H[i] = h[rev]; 
    } 
  
 //Cooley Tukey Core implementation
    for (int s = 1; s <= log2n; ++s) { 
        int m = 1 << s;//2 power s
        int m1 = m >> 1; //m/2
        Complex W(1, 0); 

        Complex Wm = Complex(cos((M_PI/m1)),-sin((M_PI/m1)));  
        for (int j = 0; j < m1; ++j) { 
            for (int k = j; k < n; k += m) {  
                Complex t = W * H[k + m1];  
                Complex u = H[k]; 
                H[k] = u + t;  
                H[k + m1] = u - t;  
            } 
            W = W*Wm; 
        } 
    } 
}

//Transpose Image
void Transpose_Image(Complex* h, int width, int height)
{
  for( int row = 0; row < height; ++row )
    for( int col = 0; col < width; ++col )
      if( col > row)
      {
        Complex temp = h[row * width + col];
        h[row * width + col] = h[col * width + row];
        h[col * width + row] = temp;
      }
}

//2D FFT Function
void Transform_2D(char* filename, int state, char* o_filename)
{
	auto start = std::chrono::system_clock::now();
	
	InputImage image(filename);  
	int Width = image.get_width();
	int Height = image.get_height();
	
	int taskid, ntasks, numtasks;
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	MPI_Comm_size(MPI_COMM_WORLD,&ntasks);
	
	
	int rowspercore = Height/ntasks;
	numtasks = ntasks;
	if(rowspercore==0)
	{ 
		rowspercore = 1;
		numtasks = Height%ntasks;
	}
	
	if(taskid > (numtasks-1)) return;
		
//-------------------------DFT starts-------------------------
	if(state == 0)
	{
		Complex* Image;
		if(taskid == 0)
		{
			Image = image.get_image_data();
		}
	
		Complex* H = new Complex[Width*rowspercore]();
		Complex* h = new Complex[Width*rowspercore]();
	
		//------Row-wise dft------
		if( taskid == 0 )
		{
			for( int i = 1; i < numtasks; ++i)
			{
			int rowstart = i * rowspercore;
			MPI_Send(Image + rowstart * Width, rowspercore * Width * sizeof(Complex), MPI_CHAR, i, rowstart, MPI_COMM_WORLD);//, NULL);
			}
		}

		int rowstart = taskid * rowspercore;
		
		if(taskid==0)
			for(int i=0;i<(rowspercore*Width);i++)
				h[i] = Image[i];
	
		if(taskid>0)
			MPI_Recv(h, rowspercore * Width * sizeof(Complex), MPI_CHAR, 0, rowstart, MPI_COMM_WORLD, NULL);
	
		for ( int i = 0; i < rowspercore; ++i )
		{
			Complex* temp1 = h + i * Width;
			Complex* temp2 = H + i * Width;
			fft(temp2, temp1, Width, log2f(Width));
		}

		if(taskid>0)
			MPI_Send(H, rowspercore * Width * sizeof(Complex), MPI_CHAR, 0, rowstart, MPI_COMM_WORLD);//, NULL);
	
		if( taskid == 0 )
		{
			for(int i=0;i<(rowspercore*Width);i++)
				Image[i] = H[i];
			
			for( int i = 1; i < numtasks; ++i)
			{
				int rowstart = i * rowspercore;
				MPI_Recv(Image + rowstart * Width, rowspercore * Width * sizeof(Complex), MPI_CHAR, i, rowstart, MPI_COMM_WORLD, NULL); 
			}
  
			//image.save_image_data("After_1d_fft.txt", Image, Width, Height); 
			Transpose_Image(Image, Width, Height);
			//image.save_image_data("After_1d_fft_transposed.txt", Image, Width, Height);
		}
	
	
		//------Column wise dft------
		Complex* H1 = new Complex[Width*rowspercore]();
		Complex* h1 = new Complex[Width*rowspercore]();
		
		if( taskid == 0 )
		{
			for( int i = 1; i < numtasks; ++i)
			{
				int rowstart = i * rowspercore;
				MPI_Send(Image + rowstart * Width, rowspercore * Width * sizeof(Complex), MPI_CHAR, i, rowstart, MPI_COMM_WORLD);//, NULL);
			}
		}
	
		if(taskid==0)
			for(int i=0;i<(rowspercore*Width);i++)
				h1[i] = Image[i];
	
		if(taskid>0)	
			MPI_Recv(h1, rowspercore * Width * sizeof(Complex), MPI_CHAR, 0, rowstart, MPI_COMM_WORLD, NULL);
	
		//Calling 1D FFT here
		for ( int i = 0; i < rowspercore; ++i )
		{
			Complex* temp1 = h1 + i * Width;
			Complex* temp2 = H1 + i * Width;
			fft(temp2, temp1, Width, log2f(Width));
		}
	
		if(taskid>0)
			MPI_Send(H1, rowspercore * Width * sizeof(Complex), MPI_CHAR, 0, rowstart, MPI_COMM_WORLD);//, NULL);

		if( taskid == 0 )
		{
			for(int i=0;i<(rowspercore*Width);i++)
				Image[i] = H1[i];
		
			for( int i = 1; i < numtasks; ++i)
			{
				int rowstart = i * rowspercore;
				MPI_Recv(Image + rowstart * Width, rowspercore * Width * sizeof(Complex), MPI_CHAR, i, rowstart, MPI_COMM_WORLD, NULL); 
			}
			Transpose_Image(Image, Width, Height);
				
			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end-start;
			std::cout << "The execution time of the MPI version is "<< elapsed_seconds.count() << " s"<<std::endl;
			
			image.save_image_data(o_filename, Image, Width, Height); 
		}
	
		delete []H;
		delete []h;
		delete []H1;
		delete []h1;	
	
	}
	
//---------------IFFT-----------------------------------------------------
	else if(state == 1)
	{
		//-------Row wise IDFT-------
		Complex* iH = new Complex[Width*rowspercore]();
		Complex* ih = new Complex[Width*rowspercore]();
		Complex* Image2;

		if( taskid == 0 )
		{
			Image2 = image.get_image_data();
			
			for( int i = 1; i < numtasks; ++i)
			{
			int rowstart = i * rowspercore;
			MPI_Send(Image2 + rowstart * Width, rowspercore * Width * sizeof(Complex), MPI_CHAR, i, rowstart, MPI_COMM_WORLD);//, NULL);
			}
		}

		int rowstart = taskid * rowspercore;
		if(taskid==0)
			for(int i=0;i<(rowspercore*Width);i++)
				ih[i] = Image2[i];
			
		if(taskid>0)
			MPI_Recv(ih, rowspercore * Width * sizeof(Complex), MPI_CHAR, 0, rowstart, MPI_COMM_WORLD, NULL);
	
		//Calling 1D IDFT here
		for ( int i = 0; i < rowspercore; ++i )
		{
			Complex* temp1 = ih + i * Width;
			Complex* temp2 = iH + i * Width;
			IFFT_1D(temp2,temp1,Width);
		}

		if(taskid>0)
			MPI_Send(iH, rowspercore * Width * sizeof(Complex), MPI_CHAR, 0, rowstart, MPI_COMM_WORLD);//, NULL);
	
		if( taskid == 0 )
		{
			for(int i=0;i<(rowspercore*Width);i++)
				Image2[i] = iH[i];
			
			for( int i = 1; i < numtasks; ++i)
			{
				int rowstart = i * rowspercore;
				MPI_Recv(Image2 + rowstart * Width, rowspercore * Width * sizeof(Complex), MPI_CHAR, i, rowstart, MPI_COMM_WORLD, NULL); 
			}
  
			//image.save_image_data("After_1d_fft.txt", Image, Width, Height); 
			Transpose_Image(Image2, Width, Height);
			//image.save_image_data("After_1d_fft_transposed.txt", Image, Width, Height);
		}
	
	
		//-------Column wise dft---------
		Complex* iH1 = new Complex[Width*rowspercore]();
		Complex* ih1 = new Complex[Width*rowspercore]();
		if( taskid == 0 )
		{
			for( int i = 1; i < numtasks; ++i)
			{
				int rowstart = i * rowspercore;
				MPI_Send(Image2 + rowstart * Width, rowspercore * Width * sizeof(Complex), MPI_CHAR, i, rowstart, MPI_COMM_WORLD);//, NULL);
			}
		}
	
		if(taskid==0)
			for(int i=0;i<(rowspercore*Width);i++)
				ih1[i] = Image2[i];
	
		if(taskid>0)	
			MPI_Recv(ih1, rowspercore * Width * sizeof(Complex), MPI_CHAR, 0, rowstart, MPI_COMM_WORLD, NULL);
	
		//Calling 1D IDFT here
		for ( int i = 0; i < rowspercore; ++i )
		{
			Complex* temp1 = ih1 + i * Width;
			Complex* temp2 = iH1 + i * Width;
			IFFT_1D(temp2, temp1, Width);
		}

		if(taskid>0)
			MPI_Send(iH1, rowspercore * Width * sizeof(Complex), MPI_CHAR, 0, rowstart, MPI_COMM_WORLD);//, NULL);

		if( taskid == 0 )
		{
			for(int i=0;i<(rowspercore*Width);i++)
				Image2[i] = iH1[i];

			for( int i = 1; i < numtasks; ++i)
			{
				int rowstart = i * rowspercore;
				MPI_Recv(Image2 + rowstart * Width, rowspercore * Width * sizeof(Complex), MPI_CHAR, i, rowstart, MPI_COMM_WORLD, NULL); 
			}
			Transpose_Image(Image2, Width, Height);
			image.save_image_data(o_filename, Image2, Width, Height); 
		
		}
	
		delete []iH;
		delete []ih;
		delete []iH1;
		delete []ih1;
		
	}
	
}

int main(int argc, char** argv)
{
	if(argc!= 4)
	{	cout<<"Incorrect set of arguments"; return -1;}
	
	char i_filename[200], transform_dir[100],o_filename[200];
	strcpy(transform_dir,argv[1]);
	strcpy(i_filename,argv[2]);
	strcpy(o_filename,argv[3]);
	string F = "forward";
	string R = "reverse";

	int taskid, numtasks, state;
	
	if(transform_dir==F)
		state = 0;
	else if(transform_dir==R)
		state = 1;
	else
	{ cout<<"Transform incorrectly specified"; return -1;}


	MPI_Init(&argc,&argv);

	Transform_2D(i_filename,state,o_filename);

	MPI_Finalize();
	
	return 0;
}