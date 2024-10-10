#include "Meshgrid.h"

Meshgrid::Meshgrid(const std::vector<double>& x, const std::vector<double>& y)
    : _x(x), _y(y), _numX(x.size()), _numY(y.size()) 
{
    _xMeshGridArr = new double[_numX * _numY];
    _yMeshGridArr = new double[_numX * _numY];
}

Meshgrid::~Meshgrid()
{
    delete[] _xMeshGridArr;
    delete[] _yMeshGridArr;
}

std::vector<std::vector<double>> Meshgrid::X() 
{
    std::vector<std::vector<double>> result(_numY, std::vector<double>(_numX, 0.0));

    for (int i = 0; i < _numY; ++i) 
    {
        for (int j = 0; j < _numX; ++j) 
        {
            result[i][j] = _x[j];
        }
    }

    return result;
}

std::vector<std::vector<double>> Meshgrid::Y() 
{
    std::vector<std::vector<double>> result(_numY, std::vector<double>(_numX, 0.0));

    for (int i = 0; i < _numY; ++i) 
    {
        for (int j = 0; j < _numX; ++j) 
        {
            result[i][j] = _y[i];
        }
    }

    return result;
}


double* Meshgrid::XMeshGridArr()
{
    for (int i = 0; i < _numY; ++i)
    {
        for (int j = 0; j < _numX; ++j)
        {
            _xMeshGridArr[i * _numX + j] = _x[j];
        }
    }

    return _xMeshGridArr;
}

double* Meshgrid::YMeshGridArr()
{
    for (int i = 0; i < _numY; ++i)
    {
        for (int j = 0; j < _numX; ++j)
        {
            _yMeshGridArr[i * _numX + j] = _y[i];
        }
    }

    return _yMeshGridArr;
}