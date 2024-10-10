#pragma once
#include <string>

class Stimulus {
public:
    Stimulus(int rows, int cols, int numFrames);
    ~Stimulus();
    void LoadStimulusData(const std::string& dataFolder);
    void Stimulus::GetFlatStimulusData(const std::string& dataFolder);

    int GetRows() const { return _nRows; }
    int GetCols() const { return _nCols; }
    int GetNumFrames() const { return _nFrames; }
    const double*** GetStimulusData() const { return const_cast<const double***>(_data); }
    double* GetFlatStimulusData() { return _flatStimData; }

private:
    int _nRows;
    int _nCols;
    int _nFrames;
    double*** _data;
    double* _flatStimData;
};
