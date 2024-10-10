#include "Meshgrid.h"

Meshgrid::Meshgrid(const std::vector<float>& x, const std::vector<float>& y)
    : _x(x), _y(y), _numX(x.size()), _numY(y.size()) 
{
    _xMeshGridArr = new float[_numX * _numY];
    _yMeshGridArr = new float[_numX * _numY];
}

Meshgrid::~Meshgrid()
{
    delete[] _xMeshGridArr;
    delete[] _yMeshGridArr;
}

std::vector<std::vector<float>> Meshgrid::X() 
{
    std::vector<std::vector<float>> result(_numY, std::vector<float>(_numX, 0.0));

    for (int i = 0; i < _numY; ++i) 
    {
        for (int j = 0; j < _numX; ++j) 
        {
            result[i][j] = _x[j];
        }
    }

    return result;
}

std::vector<std::vector<float>> Meshgrid::Y() 
{
    std::vector<std::vector<float>> result(_numY, std::vector<float>(_numX, 0.0));

    for (int i = 0; i < _numY; ++i) 
    {
        for (int j = 0; j < _numX; ++j) 
        {
            result[i][j] = _y[i];
        }
    }

    return result;
}


float* Meshgrid::XMeshGridArr()
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

float* Meshgrid::YMeshGridArr()
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