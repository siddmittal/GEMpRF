#pragma once
#include <iostream>
#include <vector>

class Linspace
{
public:
    Linspace(float start, float end, int numPoints);
    std::vector<float> Generate();
    float* GenerateLinspaceArr();
    ~Linspace();

private:
    float _start;
    float _end;
    int _numPoints;
    float _step;
    float* _resultArr;
};