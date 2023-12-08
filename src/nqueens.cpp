#include <corecrt_math.h>
#include <string>
#include <iostream>
#include <vector>

using namespace std; 
#define MAX_N 10

/*
*   The below N-Queens chessboard formulation is as follows:
*       We know that at a single row, there can only be 1 Queen
*       The state of the chessboard with regards to the Queens' positions then just needs to store N numbers:
*           for each row, store **the column that the queen is located** (which is an integer between 0 and N-1)
*   The algorithm used here works as follows:
*       Create an empty chessboard
*       Try placing a queen at each column in the first row
*       After each such placement, test the state of the chessboard. If it's still valid, then
*           Try placing a queen at each column in the SECOND row (the first row already stores a queen placement at a column there)
*           After each such placement in the second row, test the state of the chessboard. If it's still valid, then
*               Try placing a queen at each column in the THIRD row (the first and second rows already store a queen placement at columns there)
*               ...
* 
*    This algorithm is recursive: It applies the same logic again and again, while modifying the internal state.
*    GPUs and parallelism DO NOT WORK WELL WITH RECURSION. So, you need to come up with a solution that achieves the same results WITHOUT RECURSION, so that you can then convert it to OpenMP and GPU
*    Feel free to use existing resources (e.g. how to remove recursion), but REFERENCE EVERYTHING YOU USE, but DON'T COPY-PASTE ANY SOLUTION FROM ANY OBSCURE WEBSITES. 
*/

void CalculateSolutionsBruteForce(int N, vector<vector<int>>& solutions);
void PrintSolutions(int N, vector<vector<int>>& solutions);
bool CheckIfValidSolution(int N, int* rowIndices);

// This solutions uses a heuristic that there is only one Queen in a column
void CalculateAllSolutions(int N, bool print)
{
	vector<vector<int>> solutions;
	CalculateSolutionsBruteForce(N, solutions);
	printf("N=%d, solutions=%d\n", N, (int)solutions.size());
	if (print)
		PrintSolutions(N, solutions);
}

void CalculateSolutionsBruteForce(int N, vector<vector<int>>& solutions) {
	// since there is a single Queen in each row, the number of possibilities are limited to N^N
	__int64 possibleCombinations = powl(N, N); // use powl and __int64 to fit the biggest numbers
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
bool CheckIfValidSolution(int N, int* rowIndices)
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

int main(int argc, char** argv)
{
    for (int N = 4; N <= MAX_N; ++N)
        CalculateAllSolutions(N, false);
}