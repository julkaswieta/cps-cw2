#include "NQueenSolver.h"
#include <thread>

using namespace std; 

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

int main(int argc, char** argv)
{
    bool PRINT_SOLUTIONS = false;
    NQueenSolver::CalculateAllSolutions(PRINT_SOLUTIONS);
}