#include "Linspace.h"

Linspace::Linspace(double start, double end, int numPoints) : _start(start), _end(end), _numPoints(numPoints)
{    
    _resultArr = new double[_numPoints];

    if (numPoints > 1)
        _step = (_end - _start) / (_numPoints - 1);
    else
        _step = 1;
}

std::vector<double> Linspace::Generate()
{
    std::vector<double> result;
    result.reserve(_numPoints);

    for (int i = 0; i < _numPoints; ++i)
    {
        result.push_back(_start + i * _step);
    }

    return result;
}

double* Linspace::GenerateLinspaceArr()
{
    for (int i = 0; i < _numPoints; ++i)
    {
        _resultArr[i] = static_cast<double>(_start + i * _step);
    }

    return _resultArr;
}

Linspace::~Linspace()
{
    delete[] _resultArr;
}
