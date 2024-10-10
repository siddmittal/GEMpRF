#pragma once

#include <Eigen/Dense>

class Matrix
{
private:
	int _rows;
	int _cols;
	double* _qArr = nullptr;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _matrix;
	
public:
	Matrix(int rows, int cols, double* inFlatData);
	~Matrix(); // Destructor
	double* qrDecomposition(int nRowsQ, int nColsQ);
	void printMatrix(const char* name);
};

