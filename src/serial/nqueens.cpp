#include <corecrt_math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include "nqueens.h"

using namespace std; 
#define MAX_N 10
#define TEST_RUNS 5 

int main(int argc, char** argv)
{
	bool PRINT_SOLUTIONS = false;
	CalculateAllSolutions(PRINT_SOLUTIONS);
}

// This solutions uses a heuristic that there is only one Queen in a column
void CalculateAllSolutions(bool print)
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

			if (run == 0 && print)
				PrintSolutions(N, solutions);
		}
		meanTime /= (double)TEST_RUNS;
		printf("N=%d, solutions=%d, mean time=%f\n", N, solutionsCount, meanTime);
	}
}

void CalculateSolutionsBruteForce(int N, vector<vector<int>>& solutions) {
	// since there is a single Queen in each row, the number of possibilities are limited to N^N
	__int64 possibleCombinations = powl(N, N); // use powl and __int64 to fit the biggest numbers
	for (__int64 combination = 0; combination < possibleCombinations; combination++)
	{
		bool validSolution = true;
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

			if (!CheckIfValidSolution(column, rowIndices)) {
				validSolution = false;
				break;
			}
		}
		if (validSolution) {
			vector<int> temp;
			for (int i = 0; i < N; i++)
			{
				temp.push_back(rowIndices[i]);
			}
			solutions.push_back(temp);
		}
	}
}

/// <summary>
/// Check if the combination is a valid solution for N-Queens problem
/// </summary>
/// <param name="lastFilledColumn"></param>
/// <param name="rowIndices">combination of row indices to check</param>
/// <returns></returns>
bool CheckIfValidSolution(int lastFilledColumn, int* rowIndices)
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
		std::cout << text << "\n";
	}
}

