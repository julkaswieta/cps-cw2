#pragma once
#include <vector>

void CalculateSolutionsBruteForce(int N, std::vector<std::vector<int>>& solutions);
void PrintSolutions(int N, std::vector<std::vector<int>>& solutions);
bool CheckIfValidSolution(int lastFilledColumn, int* rowIndices);
void CalculateAllSolutions(bool print);