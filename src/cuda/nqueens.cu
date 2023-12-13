#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <iostream>
#include <fstream>
#include "nqueens.h"

#define MAX_N 10
#define MAX_SOLUTIONS 724 // for N = 10

using namespace std;

int main() {
    bool PRINT_SOLUTIONS = false;
	CalculateAllSolutions(PRINT_SOLUTIONS);
}

void CalculateAllSolutions(bool print) {
    for (int N = 4; N <= MAX_N; N++) {
        vector<vector<int>> solutions;
        int solutionsCount;

        auto startTime = chrono::system_clock::now();
        CalculateSolutionsCUDA(N, solutions, &solutionsCount);
        auto endTime = chrono::system_clock::now();

        auto total = endTime - startTime;
        auto totalTime = chrono::duration_cast<chrono::microseconds>(total).count();
        solutionsCount = solutions.size();

        if (print)
            PrintSolutions(N, solutions);
        printf("N=%d, solutions=%d, run time=%lld\n", N, solutionsCount, totalTime);

    }
}

// sets up the GPU and executes the kernel
void CalculateSolutionsCUDA(int N, vector<vector<int>>& solutions, int* solutionsCount) {
    __int64 possibleCombinations = powl(N, N); // use powl and __int64 to fit the biggest numbers

    size_t solutionsSize = sizeof(int[MAX_N]) * MAX_SOLUTIONS; // a solutions is an array of size N so calculate the memory for int[MAX_N]

    // initialise host memory
    *solutionsCount = 0;
    int* solutionsRaw = (int*)malloc(solutionsSize);

    // initialise device memory
    int* solutionsBuffer = nullptr;
    int* countBuffer = nullptr;

    // allocate device memory
    cudaMalloc((void**)&solutionsBuffer, solutionsSize);
    cudaMalloc((void**)&countBuffer, sizeof(int));

    // copy the starting number of solutions to device to initialise the in-device counter 
    cudaMemcpy(countBuffer, solutionsCount, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    // if there are less possible combinations than blockSize, have only one block of blockSize
    __int64 gridSize = (possibleCombinations / blockSize < 1) ? 1 : possibleCombinations / blockSize + 1;

    // call the kernel
    GenerateCombinations <<<gridSize, blockSize >>> (N, possibleCombinations, solutionsBuffer, countBuffer);

    // copy the results back to the host and free device memory 
    cudaMemcpy(solutionsRaw, solutionsBuffer, solutionsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(solutionsCount, countBuffer, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(countBuffer);
    cudaFree(solutionsBuffer);

    // process the solutions array to the final solutions vector
    for (int i = 0; i < *solutionsCount; i++) {
        std::vector<int> solution(solutionsRaw + (N * i), solutionsRaw + (N * (i + 1)));
        solutions.push_back(solution);
    }

    free(solutionsRaw); // clean up host
}

// Main CUDA kernel for generating solutions
__global__ void GenerateCombinations(int N, __int64 possibleCombinations, int* solutionsBuffer, int* countBuffer) {
    __int64 currentCombination = threadIdx.x + blockIdx.x * blockDim.x; // this is also the conversion base

    // check if the kernel has not gone over the possible combination number 
    // this is possible because the kernel may execute on a bigger number of threads 
    // than there are possible combinations
    if (currentCombination >= possibleCombinations)
        return;

    int rowIndices[MAX_N];
    if (GenerateValidCombination(N, currentCombination, &rowIndices[0])) {
        int solutionIndex = atomicAdd(countBuffer, 1); // this returns the value of countBuffer before incrementing so will act as the index to solutionsBuffer
        for (int column = 0; column < N; column++) // store combination in solutions buffer
            solutionsBuffer[N * solutionIndex + column] = rowIndices[column];
    }
}

__device__ bool GenerateValidCombination(int N, __int64 currentCombination, int* rowIndices) {
    for (int column = 0; column < N; column++) {
        rowIndices[column] = currentCombination % N;
        currentCombination /= N;

        if (!CheckIfValidSolution(column, rowIndices))
            return false;
    }
    return true;
}

// adapted to reflect column placement instead of row
__device__ bool CheckIfValidSolution(int lastFilledColumn, int* rowIndices)
{
    // Check against other queens
    for (int column = 0; column < lastFilledColumn; ++column)
    {
        // check the rows
        if (rowIndices[column] == rowIndices[lastFilledColumn])
            return false;
        // check the 2 diagonals
        const auto col1 = rowIndices[lastFilledColumn] - (lastFilledColumn - column);
        const auto col2 = rowIndices[lastFilledColumn] + (lastFilledColumn - column);
        if (rowIndices[column] == col1 || rowIndices[column] == col2)
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
        cout << text << "\n";
    }
}