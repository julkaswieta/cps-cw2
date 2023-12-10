#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <chrono>

#define MAX_N 10

using namespace std;

void CalculateAllSolutions(bool print);
void CalculateSolutionsCUDA(int N, vector<vector<int>>& solutions);
__device__ bool CheckIfValidSolution(int N, int* rowIndices);
__global__ void GenerateValidCombination(int N, __int64 possibleCombinations, int* solutionsBuffer, int* solutionsCount);
__device__ void GenerateCombination(int N, __int64 currentCombination, int* rowIndices);
void PrintSolutions(int N, vector<vector<int>>& solutions);


int main() {
	// TODO: setup GPU - err check?
	bool PRINT_SOLUTIONS = false;
	CalculateAllSolutions(PRINT_SOLUTIONS);
}

void CalculateAllSolutions(bool print) {
	ofstream data("data.csv");
	for (int N = 4; N <= MAX_N; N++) {
		data << "N " << N << "\n";
		double meanTime = 0;
		int solutionsCount;
		vector<vector<int>> solutions;

		auto startTime = chrono::system_clock::now();
		CalculateSolutionsCUDA(N, solutions);
		auto endTime = chrono::system_clock::now();

		auto total = endTime - startTime;
		auto totalTime = chrono::duration_cast<chrono::microseconds>(total).count();
		data << totalTime << "\n";
		meanTime += totalTime;
		solutionsCount = solutions.size();
		printf("N=%d, solutions=%d, run time=%lld\n", N, solutionsCount, totalTime);

		if (print)
			PrintSolutions(N, solutions);
	}
}

void CalculateSolutionsCUDA(int N, vector<vector<int>>& solutions) {
	__int64 possibleCombinations = powl(N, N); // use powl abnd __int64 to fit the biggest numbers

	// allocate device memory
	int* solutionsBuffer = nullptr;
	int* solutionsCount = 0;
	// since the biggest N supported is 10, the maximum number of solutions is 724
	size_t solutionsSize = 724 * sizeof(int);
	cudaMalloc((void**)&solutionsBuffer, solutionsSize);
	cudaMalloc((void**)&solutionsCount, sizeof(int));

	// copy data to device 
	cudaMemcpy(solutionsBuffer, solutionsCount, sizeof(int), cudaMemcpyHostToDevice);

	// run the kernel
	GenerateValidCombination <<<19531250, 512>>> (N, possibleCombinations, solutionsBuffer, solutionsCount);
	cudaDeviceSynchronize();

	// copy the solutions from device to host
	cudaMemcpy(&solutions, solutionsBuffer, solutionsSize, cudaMemcpyDeviceToHost);

	// clean up resources on device
	cudaFree(solutionsBuffer);
	cudaFree(solutionsCount); // solutionsCount is only used within the device for indexing so no need to copy unnecessarily
}

__global__ void GenerateValidCombination(int N, __int64 possibleCombinations, int* solutionsBuffer, int* solutionsCount) {
	__int64 currentCombination = threadIdx.x + blockIdx.x * blockDim.x; // this is also the conversion base

	// check if the kernel has not gone over the possible combination number 
	// this is possible because the kernel may execute on a bigger number of threads 
	// than there are possible combinations
	if (currentCombination >= possibleCombinations)
		return;

	int rowIndices[MAX_N];
	GenerateCombination(N, currentCombination, &rowIndices[0]);
	
	if (CheckIfValidSolution(N, rowIndices)) {
		int solutionIndex = atomicAdd(solutionsCount, 1); // this returns the old value before addition so it will give the index instead of count
		for (int column = 0; column < N; column++) {
			solutionsBuffer[N * solutionIndex + column] = rowIndices[column];
		}
	}
}

__device__ void GenerateCombination(int N, __int64 currentCombination, int* rowIndices) {
	for (int column = 0; column < N; column++) {
		rowIndices[column] = currentCombination % N;
		currentCombination = currentCombination / N;
	}
}

// device kernel for checking validity of a combination
__device__ bool CheckIfValidSolution(int N, int* rowIndices) {
	// compare each column's row index to every other column's index
	for (int column = 0; column < N; column++)
	{
		for (int otherColumn = column + 1; otherColumn <= N; otherColumn++)
		{
			// check for the same row index
			if (rowIndices[column] == rowIndices[otherColumn])
				return false;

			// check for diagonals
			if (abs(rowIndices[column] - rowIndices[otherColumn])
				== abs(column - otherColumn))
				return false;
		}
	}
	return true;
}

/// <summary>
/// Print all solutions for N to the console
/// Modified from the given one to rpint column-by-column instead of row-by-row
/// </summary>
/// <param name="N"></param>
/// <param name="solutions"></param>
void PrintSolutions(int N, vector<vector<int>>& solutions) {
	std::string text;
	text.resize(N * (N + 1) + 1);
	text.back() = '\n'; // add extra line at the end
	for (const auto& solution : solutions)
	{
		// go through each column
		for (int column = 0; column <= N; column++)
		{
			if (column != N) {
				int rowIndex = solution[column];
				for (int row = 0; row < N; row++)
					text[row * (N + 1) + column] = rowIndex == row ? 'X' : '.';
			}
			// if last column, add the endlines 
			else {
				for (int row = 0; row < N; row++)
					text[row * (N + 1) + column] = '\n';
			}
		}
		std::cout << text << "\n";
	}
}

