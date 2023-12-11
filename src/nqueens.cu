#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <chrono>

#include "gpuErrchk.h"

#define MAX_N 10

using namespace std;

void CalculateAllSolutions(bool print);
void CalculateSolutionsCUDA(int N, vector<vector<int>>* solutions, int* solutionsCount);
__device__ bool CheckIfValidSolution(int N, int* rowIndices);
__global__ void GenerateValidCombination(int N, __int64 possibleCombinations, int* solutionsBuffer, int* solutionsCount);
void PrintSolutions(int N, vector<vector<int>>& solutions);

int main() {

	gpuErrchk(cudaSetDevice(0));
	// TODO: setup GPU - err check?
	bool PRINT_SOLUTIONS = false;
	CalculateAllSolutions(PRINT_SOLUTIONS);
}

void CalculateAllSolutions(bool print) {
	ofstream data("data.csv");
	for (int N = 4; N <= MAX_N; N++) {
		data << "N " << N << "\n";
		double meanTime = 0;
		int solutionsCount = 0;
		vector<vector<int>> solutions = vector<vector<int>>();

		auto startTime = chrono::system_clock::now();
		CalculateSolutionsCUDA(N, &solutions, &solutionsCount);
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

void CalculateSolutionsCUDA(int N, vector<vector<int>>* solutions, int* solutionsCount) {
	__int64 possibleCombinations = powl(N, N); // use powl abnd __int64 to fit the biggest numbers

	// initialise host memory
	*solutionsCount = 0;

	// allocate device memory
	int* solutionsBuffer = nullptr;
	int* countBuffer = nullptr;

	// since the biggest N supported is 10, the maximum number of solutions is 724 and each solution will have up to N ints
	size_t solutionsSize = powl(N, 5) * N * sizeof(int);
	cudaMalloc((void**)&solutionsBuffer, solutionsSize);
	cudaMalloc((void**)&countBuffer, sizeof(int));

	// copy data to device 
	cudaMemcpy(solutionsBuffer, solutionsCount, sizeof(int), cudaMemcpyHostToDevice);

	int blockSize = 1024; // number of threads in a block
	// number of blocks necessary to compute the combinations
	int gridSize = (possibleCombinations / blockSize < 1) ? 1 : possibleCombinations / blockSize;

	// run the kernel
	GenerateValidCombination <<<gridSize, blockSize>>> (N, possibleCombinations, solutionsBuffer, solutionsCount);
	cudaDeviceSynchronize();

	// copy the solutions from device to host
	cudaMemcpy(solutionsCount, countBuffer, sizeof(int), cudaMemcpyDeviceToHost);
	int* h_solutions = (int*)malloc(solutionsSize); // Assuming maximum size is 724
	cudaMemcpy(h_solutions, solutionsBuffer, solutionsSize, cudaMemcpyDeviceToHost);
	
	// clean up resources on device
	cudaFree(solutionsBuffer);
	cudaFree(countBuffer); // solutionsCount is only used within the device for indexing so no need to copy unnecessarily


	for (int i = 0; i < *solutionsCount; i++)
	{
		std::vector<int> solution = std::vector<int>();
		for (int j = 0; j < N; j++)
			solution.push_back(h_solutions[N * i + j]);
		solutions->push_back(solution);
	}

	free(h_solutions);
}

__global__ void GenerateValidCombination(int N, __int64 possibleCombinations, int* solutionsBuffer, int* solutionsCount) {
	__int64 currentCombination = threadIdx.x + blockIdx.x * blockDim.x; // this is also the conversion base

	// check if the kernel has not gone over the possible combination number 
	// this is possible because the kernel may execute on a bigger number of threads 
	// than there are possible combinations
	if (currentCombination >= possibleCombinations)
		return;

	int rowIndices[MAX_N];
	//GenerateCombination(N, currentCombination, &rowIndices[0]);
	for (int column = 0; column < N; column++) {
		rowIndices[column] = currentCombination % N;
		if (!CheckIfValidSolution(column, rowIndices))
			return;
		currentCombination = currentCombination / N;
	}

	int solutionIndex = atomicAdd(solutionsCount, 1); // this returns the old value before addition so it will give the index instead of count
	for (int column = 0; column < N; column++) {
		solutionsBuffer[N * solutionIndex + column] = rowIndices[column];
	}
}

// device kernel for checking validity of a combination
__device__ bool CheckIfValidSolution(int lastPlacedRow, int* rowIndices) {
	int lastPlacedColumn = rowIndices[lastPlacedRow];

	// Check against other queens
	for (int row = 0; row < lastPlacedRow; ++row)
	{
		if (rowIndices[row] == lastPlacedColumn) // same column, fail!
			return false;
		// check the 2 diagonals
		const auto col1 = lastPlacedColumn - (lastPlacedRow - row);
		const auto col2 = lastPlacedColumn + (lastPlacedRow - row);
		if (rowIndices[row] == col1 || rowIndices[row] == col2)
			return false;
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

