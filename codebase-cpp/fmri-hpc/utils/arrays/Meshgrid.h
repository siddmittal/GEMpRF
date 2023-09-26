#pragma once
#include <vector>

class Meshgrid 
{
public:
    Meshgrid(const std::vector<float>& x, const std::vector<float>& y);
    std::vector<std::vector<float>> X();
    std::vector<std::vector<float>> Y();
    float* XMeshGridArr();
    float* YMeshGridArr();
    ~Meshgrid();

private:
    std::vector<float> _x;
    std::vector<float> _y;
    int _numX;
    int _numY;
    float* _xMeshGridArr;
    float* _yMeshGridArr;
};
