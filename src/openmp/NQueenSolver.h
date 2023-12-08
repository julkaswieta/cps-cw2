#pragma once
#include <vector>

class NQueenSolver {
private:
	static bool CheckIfValidSolution(int N, int* rowIndices);
	static void PrintSolutions(int N, std::vector<std::vector<int>>& solutions);
	static void CalculateSolutionsBruteForce(int N, std::vector<std::vector<int>>& solutions);

public:
	static void CalculateAllSolutions(bool print);
};