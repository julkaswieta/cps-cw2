#include "NQueenSolver.h"
#include <corecrt_math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <chrono>

using namespace std;

#define TEST_RUNS 10
#define MAX_N 10

// This solutions uses a heuristic that there is only one Queen in a column
void NQueenSolver::CalculateAllSolutions(bool print)
{
	ofstream data("data.csv");
	for (int N = 4; N <= MAX_N; N++) {
		data << "N " << N << "\n";
		double meanTime = 0;
		int solutionsCount;

		for (int run = 0; run < TEST_RUNS; run++) {
			vector<vector<int>> solutions;
			
			auto startTime = chrono::system_clock::now();
			CalculateSolutionsBruteForce(N, solutions);
			auto endTime = chrono::system_clock::now();

			auto total = endTime - startTime;
			auto totalTime = chrono::duration_cast<chrono::microseconds>(total).count();
			data << totalTime << "\n";
			meanTime += totalTime;
			solutionsCount = solutions.size();
			printf("N=%d, run=%d, run time=%lld\n", N, run, totalTime);

			if (run==0 && print)
				PrintSolutions(N, solutions);
			}

		meanTime /= (double)TEST_RUNS;
		printf("N=%d, solutions=%d, mean time=%f\n", N, solutionsCount, meanTime);
	}
}

void NQueenSolver::CalculateSolutionsBruteForce(int N, vector<vector<int>>& solutions) {
	// since there is a single Queen in each row, the number of possibilities are limited to N^N
	__int64 possibleCombinations = powl(N, N); // use powl abnd __int64 to fit the biggest numbers
#pragma omp parallel for shared(solutions)
	for (__int64 combination = 0; combination < possibleCombinations; combination++)
	{
		// this approach uses convertion to N-base number 
		// to get the sequence of row indices for Queens in subsequent columns
		__int64 conversionBase = combination;
		int rowIndices[MAX_N];

		// go column-by-column and generate the index of row 
		// to place the queen using N-base conversion
		for (int column = 0; column < N; column++)
		{
			rowIndices[column] = conversionBase % N; // this is the row index for this column
			conversionBase = conversionBase / N;
		}

		if (CheckIfValidSolution(N, rowIndices))
		{
			vector<int> temp;
			for (int i = 0; i < N; i++)
			{
				temp.push_back(rowIndices[i]);
			}
#pragma omp critical
			solutions.push_back(temp);
		}
	}
}

/// <summary>
/// Check if the combination is a valid solution for N-Queens problem
/// </summary>
/// <param name="N"></param>
/// <param name="rowIndices">combination of row indices to check</param>
/// <returns></returns>
bool NQueenSolver::CheckIfValidSolution(int N, int* rowIndices)
{
	// partially adapted from: https://stackoverflow.com/questions/50379511/less-than-n-loops-for-solving-the-n-queens-with-no-use-of-recursion
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
void NQueenSolver::PrintSolutions(int N, vector<vector<int>>& solutions) {
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
				for(int row = 0; row < N; row++)
					text[row * (N + 1) + column] = '\n';
			}
		}
		std::cout << text << "\n";
	}
}