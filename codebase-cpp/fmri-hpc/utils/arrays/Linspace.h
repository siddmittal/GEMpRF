#pragma once
#include <iostream>
#include <vector>

class Linspace
{
public:
    Linspace(double start, double end, int numPoints);
    std::vector<double> Generate();
    double* GenerateLinspaceArr();
    ~Linspace();

private:
    double _start;
    double _end;
    int _numPoints;
    double _step;
    double* _resultArr;
};