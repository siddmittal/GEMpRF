#include "Stimulus.h"
#include <fstream>
#include <iostream>

Stimulus::Stimulus(int rows, int cols, int numFrames)
    : _nRows(rows), _nCols(cols), _nFrames(numFrames) 
{
    // Allocate memory for stimulus data as a 3D array
    _data = new float** [_nFrames];
    for (int frame = 0; frame < _nFrames; ++frame) 
    {
        _data[frame] = new float* [_nRows];
        for (int row = 0; row < _nRows; ++row) 
        {
            _data[frame][row] = new float[_nCols];
        }
    }
}

Stimulus::~Stimulus() 
{
    // Free memory for stimulus data
    for (int frame = 0; frame < _nFrames; ++frame) 
    {
        for (int row = 0; row < _nRows; ++row)
        {
            delete[] _data[frame][row];
        }
        delete[] _data[frame];
    }
    delete[] _data;
}

void Stimulus::LoadStimulusData(const std::string& dataFolder) 
{
    for (int frameNum = 0; frameNum < _nFrames; ++frameNum) 
    {
        std::string frameFilename = dataFolder + "/frame-" + std::to_string(frameNum) + ".txt";
        std::ifstream file(frameFilename);

        if (file.is_open()) 
        {
            for (int row = 0; row < _nRows; ++row) 
            {
                for (int col = 0; col < _nCols; ++col) 
                {
                    if (!(file >> _data[frameNum][row][col])) 
                    {
                        std::cerr << "Error reading frame data from " << frameFilename << std::endl;
                        // Handle error as needed
                        return;
                    }
                }
            }
            file.close();
        }
        else 
        {
            std::cerr << "Error opening file: " << frameFilename << std::endl;
            // Handle error as needed
            return;
        }
    }
}
