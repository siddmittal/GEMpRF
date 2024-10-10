#include <iostream>
#include "ModelSignals/header/ModelSignals.h"
#include "utils/matrix/Matrix.h"
#include "utils/matrix/Matrix.h"
#include "Eigen/Dense"
#include "Stimulus/Stimulus.h"

const double PI = 3.14159265358979323846;

void AssignDummyStimulusData(double* outResult)
{
	// Stimulus - 4 frames, 5 x 5 size
	double flattenedHRFConvolvedStimulusData[4 * 5 * 5] = {

		// frame - 1
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,


		// frame - 2
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		// frame - 3
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,

		// frame - 4
		0, 0, 1, 0, 0,
		0, 1, 1, 1, 0,
		1, 1, 1, 1, 1,
		0, 1, 1, 1, 0,
		0, 0, 1, 0, 0,

	};

	std::copy(flattenedHRFConvolvedStimulusData, flattenedHRFConvolvedStimulusData + 4 * 5 * 5, outResult);
}

Eigen::MatrixXf makeTrends_1(int nDCT) 
{
	

	int tf = 10;
	int ndct = 2 * nDCT + 1;

	Eigen::MatrixXf trends(tf, std::max(ndct, 1));
	Eigen::VectorXf tc = Eigen::VectorXf::LinSpaced(tf, 0, 2 * PI);

	for (int i = 0; i < tf; i++) 
	{
		for (int j = 0; j < ndct; j++) 
		{
			trends(i, j) = std::cos(tc[i] * (j * 0.5));
		}
	}

	return trends;
}

void makeTrends_work()
{
	const double PI2 = 3.14159265358979323846;

	int nDCT = 1;
	double trends[30];
	const int tf = 10;

	int ndct = 1;

	for (int i = 0; i < tf; i++)
	{
		double tc = 2.0 * PI2 * i / (tf - 1);
		double val = std::cos(tc);
		trends[i * ndct] = val;

	}

	// Print the flattened array in row-major order
	for (int i = 0; i < tf; i++)
	{
		std::cout << trends[i] << " ";
		std::cout << std::endl;
	}

	int i = 0;
}

void makeTrends_3() 
{
	int nDCT = 1;	
	const int tf = 10;
	double trends[tf];

	int ndct = 2 * nDCT + 1;

	for (int i = 0; i < tf; i++) 
	{
		double tc = 2.0 * PI * i / tf;

			double val = std::cos(tc);
			trends[i ] = val;
	}

	// Print the flattened array in row-major order
	for (int i = 0; i < tf; i++) 
	{
			std::cout << trends[i] << " ";

	}

	int i = 0;
}

double* makeTrends(int nDCT, int nStimulusFrames)
{	
	int tf = nStimulusFrames; // timepoints
	const double PI = 3.14159265358979323846;

	int ndct = 2 * nDCT + 1;
	double* trends = new double[nStimulusFrames * ndct];

	for (int i = 0; i < tf; i++) 
	{
		double tc = 2.0 * PI * i / (tf - 1);
		for (int j = 0; j < ndct; j++) 
		{
			double val = std::cos(tc * (j * 0.5));
			trends[i * ndct + j] = val;
		}
	}

	//// Print the flattened array in row-major order
	//for (int i = 0; i < tf; i++) 
	//{
	//	for (int j = 0; j < 2 * nDCT + 1; j++) 
	//	{
	//		std::cout << trends[i * (2 * nDCT + 1) + j] << " ";
	//	}
	//	std::cout << std::endl;
	//}

	return trends;
}

void testTrendsComputation()
{
	const int nStimulusFrames = 4;

	// Trends
	int nDCT = 3;
	double* trends = makeTrends(nDCT, nStimulusFrames);
	Matrix trendsMat(nStimulusFrames, 2 * nDCT + 1, trends);
	trendsMat.printMatrix("trends");

	// QR decomposition on model signal timeseries
	int nRowsThinQ = nStimulusFrames;
	int nColsThinQ = nStimulusFrames;
	double* qArr = trendsMat.qrDecomposition(nRowsThinQ, nColsThinQ); // NOTE: <----Returning Signs flipped

	Matrix qMat(nRowsThinQ, nColsThinQ, qArr); // Q matrix is square matrix
	qMat.printMatrix("Q");
}

int main_dummy()
{
	//double* flattenedHRFConvolvedStimulusData = 0;
	const int nQuadrilateralSpaceGridRows = 2;
	const int nQuadrilateralSpaceGridCols = 3;
	const int nStimulusRows = 5; //3;
	const int nStimulusCols = 5; //3;
	const int nStimulusFrames = 4;
	const double sigma = 2;
	const int nModelSignalsPerStimulusFrame = nQuadrilateralSpaceGridRows * nQuadrilateralSpaceGridCols;

	//double flattenedHRFConvolvedStimulusData[] = {
	//	1, 2, 3, 4, 5,			1, 2, 3, 4, 5,			1, 2, 3, 4, 5,			1, 2, 3, 4, 5,			1, 2, 3, 4, 5,		// frame - 1
	//	11, 12, 13, 14, 15,		11, 12, 13, 14, 15,		11, 12, 13, 14, 15,		11, 12, 13, 14, 15,		11, 12, 13, 14, 15,	// frame - 2
	//};

	//double flattenedHRFConvolvedStimulusData[] = {
	//1, 2, 3,	1, 2 , 3,	1, 2 , 3, 		// frame - 1
	//1, 2, 3,	1, 2 , 3,	1, 2 , 3, 		// frame - 1
	//1, 2, 3,	1, 2 , 3,	1, 2 , 3, 		// frame - 2
	//1, 2, 3,	1, 2 , 3,	1, 2 , 3 		// frame - 2
	//};

	// Dummy stimulus data
	double* flattenedHRFConvolvedStimulusData = new double[nStimulusFrames * nStimulusRows * nStimulusCols]; // Stimulus - 4 frames, 5 x 5 size
	AssignDummyStimulusData(flattenedHRFConvolvedStimulusData);

	// Model signals
	ModelSignals modelSignals(flattenedHRFConvolvedStimulusData
		, nQuadrilateralSpaceGridRows
		, nQuadrilateralSpaceGridCols
		, nStimulusRows
		, nStimulusCols
		, nStimulusFrames
		, sigma
	);

	// model timeseries
	modelSignals.generateModelSignals();
	double* flatModelSignalsArr = modelSignals.getResults();
	Matrix modelSignalsMatrix(nModelSignalsPerStimulusFrame, nStimulusFrames, flatModelSignalsArr);
	//modelSignalsMatrix.printMatrix("Ortho Timeseries Signals");

	// IMPORTANT NOTE: computed timecourses in "flatModelSignalsArr" need to be normalized row-wise. Will write a kernel later!!!

	

	return 0;
}


// Function to write Gaussian curves to text files
#include <fstream>
#include <string>
void writeOrthogonalizedSignalsToFile(const std::string& dataFolder
	, double* result_ortho_signals_data
	, int nModelSpaceGridRows
	, int nModelSpaceGridCols
	, int nStimulusRows
	, int nStimulusCols
	, int nStimulusFrames
)
{
	int nTotalOrthoSignals = nModelSpaceGridRows * nModelSpaceGridCols;
	int nSignalLength = nStimulusFrames;

	for (int signalIdx = 0; signalIdx < nTotalOrthoSignals; signalIdx++)
	{
		std::string filename = dataFolder + "s-" + std::to_string(signalIdx) + ".txt";
		std::ofstream outfile(filename);
		for (int signalDataIdx = 0; signalDataIdx < nSignalLength; signalDataIdx++)
		{			
			double value = result_ortho_signals_data[signalIdx * nSignalLength + signalDataIdx];
			outfile << value << "\n";
		}
		outfile.close();
	}
}

int main()
{
	const int nQuadrilateralSpaceGridRows = 2;
	const int nQuadrilateralSpaceGridCols = 3;
	const int nStimulusRows = 101; //3;
	const int nStimulusCols = 101; //3;
	const int nStimulusFrames = 300;
	const double sigma = 2;
	const int nModelSignalsPerStimulusFrame = nQuadrilateralSpaceGridRows * nQuadrilateralSpaceGridCols;

	Stimulus* stimulus = new Stimulus(nStimulusRows, nStimulusCols, nStimulusFrames);
	stimulus->LoadStimulusData("D:\\code\\sid-git\\fmri\\local-extracted-datasets\\gpu-tests\\stimulus-data-test");
	double *flattenedHRFConvolvedStimulusData = stimulus->GetFlatStimulusData();

	// Model signals
	ModelSignals modelSignals(flattenedHRFConvolvedStimulusData
		, nQuadrilateralSpaceGridRows
		, nQuadrilateralSpaceGridCols
		, nStimulusRows
		, nStimulusCols
		, nStimulusFrames
		, sigma
	);

	// model timeseries
	modelSignals.generateModelSignals();
	double* flatModelSignalsArr = modelSignals.getResults();
	//Matrix modelSignalsMatrix(nModelSignalsPerStimulusFrame, nStimulusFrames, flatModelSignalsArr);
	//modelSignalsMatrix.printMatrix("Ortho Timeseries Signals");

	// IMPORTANT NOTE: computed timecourses in "flatModelSignalsArr" need to be normalized row-wise. Will write a kernel later!!!

	// Verify
	writeOrthogonalizedSignalsToFile("D:\\code\\sid-git\\fmri\\local-extracted-datasets\\gpu-tests\\result-timecourses\\"
		, flatModelSignalsArr
		, nQuadrilateralSpaceGridRows
		, nQuadrilateralSpaceGridCols
		, nStimulusRows
		, nStimulusCols
		, nStimulusFrames
	);

	delete stimulus;
	return 0;
}
