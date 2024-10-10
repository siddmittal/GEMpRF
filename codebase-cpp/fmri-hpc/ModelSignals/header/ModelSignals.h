#pragma once

class ModelSignals
{
private:
	double* _flattenedHRFConvolvedStimulusData;
	int _nQuadrilateralSpaceGridRows;
	int _nQuadrilateralSpaceGridCols;
	int _nStimulusRows;
	int _nStimulusCols;
	int _nStimulusFrames;
	double _sigma;
	int _singleFlattenedGaussianCurvelength;
	int _nTotalGaussianCurvesPerStimulusFrame;
	double* computeModelCurves();// could be Gaussian Curves or any other model
	double* applyStimulusMask(double* dev_computed_gaussian_curves);
	double* computeRowwiseSum(double* dev_stimulusMaskAppliedGaussianCurves);
	double* orthonormalizeSignals(double* d_S, double* host_R);
	void freeDeviceMemoryWrapper(double* floatDevicePtrToFree);
	void computeTrends(int nDCT, int nStimulusFrames, double* trends);
	double* _h_result_columns_summation;

public:
	ModelSignals(double* flattenedHRFConvolvedStimulusData
		, int nQuadrilateralSpaceGridRows
		, int nQuadrilateralSpaceGridCols
		, int nStimulusRows
		, int nStimulusCols
		, int nStimulusFrames
		, double sigma
		);
	~ModelSignals();
	bool generateModelSignals();
	double* getResults();
};

