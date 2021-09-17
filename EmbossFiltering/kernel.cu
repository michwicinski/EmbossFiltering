#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bitmap_image.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace std;

void callThreadsAndMeasureTimeOnCPU(bitmap_image& image, bitmap_image& outputImage, double timeTable[]);
void embossFiltering(int whichOneThread, int threadsAmount, bitmap_image& image, bitmap_image& outputImage);
void calculateNewPixelColorValues(bitmap_image& image, unsigned int x, unsigned int y, bitmap_image& outputImage);
void prepareDataForGPUAndMeasureTime(unsigned char* imageRGBValues, unsigned char* outputImageRGBValues, int width, int height, int size, int row, float& timeGPU);
__global__ void embossFilteringOnGPU(unsigned char* imageRGBValues, unsigned char* outputImageRGBValues, int height, int width, int size, int row);
__device__ void calculateNewPixelColorValuesOnGPU(unsigned char* imageRGBValues, int position, int width, int height, unsigned char* outputImageRGBValues);

int main()
{
	bitmap_image image("11846x9945.bmp");
	bitmap_image outputImageCPU(image.width(), image.height());
	bitmap_image outputImageGPU(image.width(), image.height());
	double timeTable[4];
	float timeGPU = 0.0;

	callThreadsAndMeasureTimeOnCPU(image, outputImageCPU, timeTable);
	outputImageCPU.save_image("outputImageCPU.bmp");

	prepareDataForGPUAndMeasureTime(image.data(), outputImageGPU.data(), image.width(), image.height(),
		image.width() * image.height() * image.bytes_per_pixel() * sizeof(unsigned char), image.width() * image.bytes_per_pixel(), timeGPU);
	outputImageGPU.save_image("outputImageGPU.bmp");

	cout << "Czas wykonywania: " << endl;
	cout << "1 watek - " << timeTable[0] << "s" << endl;
	cout << "4 watki - " << timeTable[1] << "s" << endl;
	cout << "8 watkow - " << timeTable[2] << "s" << endl;
	cout << "12 watkow - " << timeTable[3] << "s" << endl;
	cout << "GPU - " << timeGPU << "s";

	return 0;
}

void callThreadsAndMeasureTimeOnCPU(bitmap_image& image, bitmap_image& outputImage, double timeTable[])
{
	int counter = 0;
	vector<thread> threads;
	for (int i = 0; i <= 12; i += 4)
	{
		if (i == 0)
			i++;

		if (!threads.empty())
			threads.clear();

		auto begin = chrono::high_resolution_clock::now();
		for (int j = 0; j < i; j++)
			threads.push_back(thread(&embossFiltering, j, i, ref(image), ref(outputImage)));
		for (auto& t : threads)
			t.join();
		auto end = chrono::high_resolution_clock::now();

		chrono::duration<double> diff = end - begin;

		timeTable[counter] = chrono::duration<double>(diff).count();
		counter++;

		if (i == 1)
			i--;
	}
}

void embossFiltering(int whichOneThread, int threadsAmount, bitmap_image& image, bitmap_image& outputImage)
{
	unsigned int height = (unsigned int)(image.height() / threadsAmount);
	unsigned int width = image.width();

	for (unsigned int y = whichOneThread * height; y < whichOneThread * height + height; y++)
	{
		for (unsigned int x = 0; x < width; x++)
		{
			rgb_t color;

			image.get_pixel(x, y, color);

			calculateNewPixelColorValues(image, x, y, outputImage);
		}
	}
}

//emboss kernel = -2 -1  0
//                -1  1  1
//                 0  1  2
void calculateNewPixelColorValues(bitmap_image& image, unsigned int x, unsigned int y, bitmap_image& outputImage)
{
	int xPosition[] = { -1, 0, 1 };
	int yPosition[] = { -1, 0, 1 };
	int mask[][3] = { {-2, -1, 0}, {-1, 1, 1}, {0, 1, 2} };

	rgb_t color;
	unsigned int xTemp, yTemp;
	int rSum = 0, bSum = 0, gSum = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			xTemp = x + xPosition[i];
			yTemp = y + yPosition[j];

			if (xTemp >= 0 && xTemp <= image.width() - 1 && yTemp >= 0 && yTemp <= image.height() - 1)
			{
				image.get_pixel(xTemp, yTemp, color);

				rSum += color.red * mask[i][j];
				gSum += color.green * mask[i][j];
				bSum += color.blue * mask[i][j];
			}
		}
	}

	if (rSum > 255) rSum = 255;
	if (rSum < 0) rSum = 0;
	if (gSum > 255) gSum = 255;
	if (gSum < 0) gSum = 0;
	if (bSum > 255) bSum = 255;
	if (bSum < 0) bSum = 0;

	outputImage.set_pixel(x, y, rSum, gSum, bSum);
}

void prepareDataForGPUAndMeasureTime(unsigned char* imageRGBValues, unsigned char* outputImageRGBValues, int width, int height, int size, int row, float& timeGPU)
{
	unsigned char* imageRGBValuesGPU, * outputImageRGBValuesGPU;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	cudaMalloc((void**)&imageRGBValuesGPU, size);
	cudaMalloc((void**)&outputImageRGBValuesGPU, size);

	cudaMemcpy(imageRGBValuesGPU, imageRGBValues, size, cudaMemcpyHostToDevice);

	embossFilteringOnGPU<<<(width * height + 255) / 256, 256>>>(imageRGBValuesGPU, outputImageRGBValuesGPU, width, height, size, row);

	cudaMemcpy(outputImageRGBValues, outputImageRGBValuesGPU, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&timeGPU, start, stop);
	timeGPU /= 1000.0;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

__global__ void embossFilteringOnGPU(unsigned char* imageRGBValues, unsigned char* outputImageRGBValues, int width, int height, int size, int row)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	i *= 3;

	if (i > size)
		return;

	int y = i / row;
	int x = (i - y * row) / 3;

	if (y < 0 || y > height)
		return;
	if (x < 0 || x > row)
		return;

	int pixelPosistion = y * row + 3 * x;

	calculateNewPixelColorValuesOnGPU(imageRGBValues, pixelPosistion, width, height, outputImageRGBValues);
}

__device__ void calculateNewPixelColorValuesOnGPU(unsigned char* imageRGBValues, int position, int width, int height, unsigned char* outputImageRGBValues)
{
	int pos[] = { 3 * -width + 3 * -1, 3 * -width, 3 * -width + 3 * 1, 3 * -1, 0, 3 * 1, 3 * width + 3 * -1, 3 * width, 3 * width + 3 * 1 };
	int mask[] = { -2, -1, 0, -1, 1, 1, 0, 1, 2 };

	int rSum = 0, bSum = 0, gSum = 0;
	for (int i = 0; i < 9; i++)
	{
		int positionTemp = position + pos[i];
		int x = positionTemp % (3 * width);
		int xMiddle = position % (3 * width);

		if (positionTemp >= 0 && positionTemp < width * 3 * (height - 1) + 3 * (width - 1) && abs(x - xMiddle) < 3 * 2)
		{
			rSum += imageRGBValues[positionTemp + 2] * mask[i];
			gSum += imageRGBValues[positionTemp + 1] * mask[i];
			bSum += imageRGBValues[positionTemp] * mask[i];
		}
	}

	if (rSum > 255) rSum = 255;
	if (rSum < 0) rSum = 0;
	if (gSum > 255) gSum = 255;
	if (gSum < 0) gSum = 0;
	if (bSum > 255) bSum = 255;
	if (bSum < 0) bSum = 0;

	outputImageRGBValues[position + 2] = rSum;
	outputImageRGBValues[position + 1] = gSum;
	outputImageRGBValues[position] = bSum;
}