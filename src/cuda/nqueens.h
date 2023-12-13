#pragma once
#include <vector>
#include <cuda_runtime.h>

void CalculateAllSolutions(bool print);
void CalculateSolutionsCUDA(int N, std::vector<std::vector<int>>& solutions, int* solutionsCount);
void PrintSolutions(int N, std::vector<std::vector<int>>& solutions);

// CUDA kernels
__global__ void GenerateCombinations(int N, __int64 possibleCombinations, int* solutionsBuffer, int* countBuffer);
__device__ bool GenerateValidCombination(int N, __int64 currentCombination, int* rowIndices);
__device__ bool CheckIfValidSolution(int lastFilledColumn, int* rowIndices);
