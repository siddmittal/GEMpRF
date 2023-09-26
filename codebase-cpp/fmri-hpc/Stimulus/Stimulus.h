#pragma once
#include <string>

class Stimulus {
public:
    Stimulus(int rows, int cols, int numFrames);
    ~Stimulus();
    void LoadStimulusData(const std::string& dataFolder);

    int GetRows() const { return _nRows; }
    int GetCols() const { return _nCols; }
    int GetNumFrames() const { return _nFrames; }
    const float*** GetStimulusData() const { return const_cast<const float***>(_data); }

private:
    int _nRows;
    int _nCols;
    int _nFrames;
    float*** _data;
};
