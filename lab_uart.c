
#include <stdio.h>
#include <stdlib.h>
#include "platform.h"
#include "xil_printf.h"
#include "xuartps.h"
#include "xtime_l.h"

#include <math.h>

#include <arm_neon.h>

typedef short int DATA;


#define FIXED2FLOAT(a, qf) (((float) (a)) / (1<<qf))
#define FLOAT2FIXED(a, qf) ((short int) round((a) * (1<<qf)))

#define _MAX_ (1 << (sizeof(DATA)*8-1))-1
#define _MIN_ -(_MAX_+1)

DATA image[28*28];  // image

int N_W0 = 64*784;
int N_B0 = 64;
int N_W1 = 32*64;
int N_B1 = 32;
int N_W2 = 16*32;
int N_B2 = 16;
int N_W3 = 10*16;
int N_B3 = 10;




// declare arrays for inputs and for exchanging tensors between layers
DATA gemm0W[64*784];
DATA gemm1W[32*64];
DATA gemm2W[16*32];
DATA gemm3W[10*16];

DATA gemm0B[64];
DATA gemm1B[32];
DATA gemm2B[16];
DATA gemm3B[10];

DATA gemm0R[64]; // Size of N_B0
DATA gemm1R[32]; // Size of N_B1
DATA gemm2R[16];
DATA gemm3R[10];
DATA relu0R[64];
DATA relu1R[32];
DATA relu2R[16];


void FC_forward(DATA* input, DATA* output, int in_s, int out_s, DATA* weights, DATA* bias, int qf) ;
static inline long long int saturate(long long int mac);
static inline void relu_forward(DATA* input, DATA* output, int size);
int resultsProcessing(DATA* results, int size);



DATA readfromUART(){ // reads a sequence of bytes and composes the DATA

	DATA pixel;
	u8 pixel1,pixel2;
	pixel1 = inbyte();
	pixel2 =  inbyte();
	pixel = (pixel2 << 8) + pixel1;
	return pixel;
}

void flatten(DATA * file, int size){
	for (int i=0; i<size; i++){
		file[i] = readfromUART();
	}
}


int main(){

  init_platform();


  XUartPs Uart_1_PS;

  u16 DeviceId_1= XPAR_PS7_UART_1_DEVICE_ID;

  int Status_1;
  XUartPs_Config *Config_1;


  Config_1 = XUartPs_LookupConfig(DeviceId_1);
  if (NULL == Config_1) {
    return XST_FAILURE;
  }

  /*the default configuration is stored in Config and it can be used to initialize the controller */


  Status_1 = XUartPs_CfgInitialize(&Uart_1_PS, Config_1, Config_1->BaseAddress);
  if (Status_1 != XST_SUCCESS) {
    return XST_FAILURE;
  }


  // Set the BAUD rate
  u32 BaudRate = (u32)115200;

  Status_1 = XUartPs_SetBaudRate(&Uart_1_PS, BaudRate);
  if (Status_1 != (s32)XST_SUCCESS) {
    return XST_FAILURE;
  }

  xil_printf ("Started\n");


	int before;
	int after;


  // read weights and bias

  // FILLING THE ARRAYS
  xil_printf("Waiting for the biases of layer 0...\n");
  flatten(gemm0B,N_B0);

  xil_printf("Waiting for the weights of layer 0...\n");
  flatten(gemm0W,N_W0);


  xil_printf("Waiting for the biases of layer 1...\n");
  flatten(gemm1B,N_B1);

  xil_printf("Waiting for the weights of layer 1...\n");
  flatten(gemm1W,N_W1);




  xil_printf("Waiting for the biases of layer 2...\n");
  flatten(gemm2B,N_B2);
  xil_printf("Waiting for the weights of layer 2...\n");

  flatten(gemm2W,N_W2);

  xil_printf("Waiting for the biases of layer 3...\n");
  flatten(gemm3B,N_B3);
  xil_printf("Waiting for the weights of layer 3...\n");
  flatten(gemm3W,N_W3);


  int result;


  // read image

  while (1){



	  xil_printf("Waiting for the image...\n");
	  // add your code reading the image from UART

	  flatten(image, 28*28);

	  xil_printf("Waiting for the process \n");

	  // add your code processing the image  with the DNN according to the onnx file, compose using the DNN functions declared above.

	  xil_printf("Classification in progress... \n");


	  FC_forward(image,gemm0R,28*28,N_B0,gemm0W,gemm0B,8);


	  relu_forward(gemm0R,relu0R,N_B0);


	  FC_forward(relu0R,gemm1R,N_B0,N_B1,gemm1W,gemm1B,8);


	  relu_forward(gemm1R,relu1R,N_B1);


	  FC_forward(relu1R,gemm2R,N_B1,N_B2,gemm2W,gemm2B,8);


	  relu_forward(gemm2R,relu2R,N_B2);


	  FC_forward(relu2R,gemm3R,N_B2,N_B3,gemm3W,gemm3B,8);

	  result = resultsProcessing(gemm3R,N_B3);


	  // send the result in output to the UART
	  xil_printf("Done ! \n");
	  xil_printf("The image most likely represents a : %d \n\n",result);
  }



    cleanup_platform();
    return 0;
}



void FC_forward(DATA* input, DATA* output, int in_s, int out_s, DATA* weights, DATA* bias, int qf) {

	// NOTE return W * x

	int hkern = 0;
	int wkern = 0;

	long long int mac = 0;





	DATA current = 0;

	/* foreach row in kernel */
//	#pragma omp parallel for private (hkern, wkern, mac, current)
	for (hkern = 0; hkern < out_s; hkern++) {

		mac = ((long long int)bias[hkern]) << qf;





		for (wkern = 0; wkern < in_s; wkern++) {
			current = input[wkern];
			mac += current * weights[hkern*in_s + wkern];
		}



			output[hkern] = (DATA)(mac >> qf);

	}


}

static inline long long int saturate(long long int mac)
{

	if(mac > _MAX_) {
		printf("[WARNING] Saturation.mac: %lld -> %llx _MAX_: %d  _MIN_: %d  res: %d\n",  mac, mac, _MAX_, _MIN_, _MAX_);
		return _MAX_;
	}

	if(mac < _MIN_){
		printf( "[WARNING] Saturation.mac: %lld -> %llx _MAX_: %d  _MIN_: %d  res: %d\n",  mac, mac, _MAX_, _MIN_, _MIN_);
		return _MIN_;
	}

	//printf("mac: %lld -> %llx _MAX_: %lld  _MIN_: %lld  res: %lld\n", mac, mac, _MAX_, _MIN_, mac);
    return mac;

}

static inline void relu_forward(DATA* input, DATA* output, int size) {
	int i = 0;

	for(i = 0; i < size; i++) {
		DATA v = input[i];
		v = v > 0 ? v : 0;

		output[i] = v;
	}
}


int resultsProcessing(DATA* results, int size){
 char *labels[10]={"digit 0", "digit 1", "digit 2", "digit 3", "digit 4", "digit 5", "digit 6", "digit 7", "digit 8", "digit 9"};


  int size_wa = 10;
  float* r= (float*) malloc (size_wa*sizeof(float));
  int*  c= (int*)  malloc (size_wa*sizeof(int));
  float* results_float= (float*) malloc (size_wa*sizeof(float));
  float sum=0.0;
  DATA max=0;
  for (int i =0;i<size_wa;i++){
      results_float[i] = FIXED2FLOAT(results[i],8);
    int n;
    if (results[i]>0)
      n=results[i];
    else
      n=-results[i];
    if (n>max){
      max=n;
    }
  }

  for (int i =0;i<size_wa;i++)
    sum+=exp(results_float[i]);

  for (int i =0;i<size_wa;i++){
    r[i]=exp(results_float[i]) / sum;
    c[i]=i;
  }
  for (int i =0;i<size_wa;i++){
    for (int j =i;j<size_wa;j++){
      if (r[j]>r[i]){
        float t= r[j];
        r[j]=r[i];
        r[i]=t;
        int tc= c[j];
        c[j]=c[i];
        c[i]=tc;
      }
    }
  }
  int top0=0;
  float topval=results_float[0];
  for (int i =1;i<size_wa;i++){
    if (results_float[i]>topval){
      top0=i;
      topval=results_float[i];
    }
  }
  xil_printf("\n\n");
  for (int i =0;i<5;i++){
	  xil_printf("            TOP %d: [%d] %s   \n",i, c[i], labels[c[i]]);

  }
  xil_printf("\n\n\n\n\n");
  return top0;


}

