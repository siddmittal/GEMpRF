#include "Linspace.h"

Linspace::Linspace(float start, float end, int numPoints) : _start(start), _end(end), _numPoints(numPoints)
{    
    _resultArr = new float[_numPoints];

    if (numPoints > 1)
        _step = (_end - _start) / (_numPoints - 1);
    else
        _step = 1;
}

std::vector<float> Linspace::Generate()
{
    std::vector<float> result;
    result.reserve(_numPoints);

    for (int i = 0; i < _numPoints; ++i)
    {
        result.push_back(_start + i * _step);
    }

    return result;
}

float* Linspace::GenerateLinspaceArr()
{
    for (int i = 0; i < _numPoints; ++i)
    {
        _resultArr[i] = static_cast<float>(_start + i * _step);
    }

    return _resultArr;
}

Linspace::~Linspace()
{
    delete[] _resultArr;
}
