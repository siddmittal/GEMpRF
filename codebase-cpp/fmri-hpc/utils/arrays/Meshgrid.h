#pragma once
#include <vector>

class Meshgrid 
{
public:
    Meshgrid(const std::vector<double>& x, const std::vector<double>& y);
    std::vector<std::vector<double>> X();
    std::vector<std::vector<double>> Y();
    double* XMeshGridArr();
    double* YMeshGridArr();
    ~Meshgrid();

private:
    std::vector<double> _x;
    std::vector<double> _y;
    int _numX;
    int _numY;
    double* _xMeshGridArr;
    double* _yMeshGridArr;
};
