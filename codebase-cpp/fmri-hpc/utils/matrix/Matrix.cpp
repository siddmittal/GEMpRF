#include "Matrix.h"
#include <cmath>
#include <iostream>


Matrix::Matrix(int rows, int cols, double* inFlatData) : _rows(rows), _cols(cols), _matrix(_rows, _cols)
{
    _matrix = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(inFlatData, _rows, _cols);
    //std::cout << _matrix << "\n\n" << std::endl;
}

Matrix::~Matrix() 
{
    if(_qArr != nullptr)
        delete[] _qArr;
}

double* Matrix::qrDecomposition(int nRowsThinQ, int nColsThinQ)
{
    if (_qArr != nullptr)
        return _qArr;

    // allocate memory
    _qArr = new double[nRowsThinQ * nColsThinQ];
   
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix(_rows, _cols);
    

    auto QR = _matrix.householderQr();  //auto may not be efficient here...
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Q1 = QR.householderQ();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> R1 = QR.matrixQR().triangularView<Eigen::Upper>();

    //std::cout << Q1 << "\n\n" << R1 << "\n\n" << Q1 * R1 << std::endl;

    // Householder algo returns the full size matrix for Q so, we need to thin the Q by multiplying it with a thin identity matrix    
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> identity = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(nRowsThinQ, nColsThinQ);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Q_thinned = Q1 * identity;

    //std::cout << "Q1 * Identity (Q_thinned):\n" << Q_thinned << std::endl;
    
    // Copy data from Q_thinned to _qArr    
    // _qArr = = Q_thinned.data(); // Don't do this!!! data's lifetime is tied to the Q_thinned matrix, which will be go out of scope
    std::copy(Q_thinned.data(), Q_thinned.data() + nRowsThinQ * nColsThinQ, _qArr);

    return _qArr;
}

void Matrix::printMatrix(const char* name)
{
    std::cout << name << ":" << std::endl;
    std::cout << _matrix <<"\n\n" << std::endl;
}


