#include "HTMRL.h"

#include <iostream>

using namespace htm;

void HTMRL::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, int reconstructionReceptiveRadius, const std::vector<LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, float minInitCenter, float maxInitCenter, std::mt19937 &generator) {
	struct Uint2 {
		unsigned int _x, _y;
	};

	_addReplaySampleStepCounter = 0;
	
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_layerDescs = layerDescs;
	
	_layers.resize(_layerDescs.size());

	_inputTypes = inputTypes;

	std::uniform_real_distribution<float> weightDist(minInitWeight, maxInitWeight);
	std::uniform_real_distribution<float> actionDist(0.0f, 1.0f);

	_prevMaxQ = 0.0f;
	_prevValue = 0.0f;
	_prevPrevValue = 0.0f;
	_prevQ = 0.0f;
	_prevTDError = 0.0f;

	cl::Kernel initPartOneKernel = cl::Kernel(program.getProgram(), "initializePartOne");
	cl::Kernel initPartTwoKernel = cl::Kernel(program.getProgram(), "initializePartTwo");
	cl::Kernel initPartThreeKernel = cl::Kernel(program.getProgram(), "initializePartThree");

	_input.clear();
	_input.resize(_inputWidth * _inputHeight);

	_output.clear();
	_output.assign(_inputWidth * _inputHeight, 0.0f);

	_prediction.clear();
	_prediction.assign(_inputWidth * _inputHeight, 0.0f);

	_exploratoryOutput.clear();
	_exploratoryOutput.assign(_inputWidth * _inputHeight, 0.0f);

	_prevOutput.clear();
	_prevOutput.assign(_inputWidth * _inputHeight, 0.0f);

	_prevOutputExploratory.clear();
	_prevOutputExploratory.assign(_inputWidth * _inputHeight, 0.0f);

	_prevInput.clear();
	_prevInput.assign(_inputWidth * _inputHeight, 0.0f);

	// Initialize action portions randomly
	for (int i = 0; i < _input.size(); i++)
	if (_inputTypes[i] == _action) {
		float value = actionDist(generator);

		_input[i] = value;

		_exploratoryOutput[i] = value;

		_prevOutput[i] = value;

		_prevOutputExploratory[i] = value;

		_prevInput[i] = value;
	}

	_inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

	_reconstructedPrediction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;
	int prevCellsPerColumn = 1;

	for (int l = 0; l < _layers.size(); l++) {
		initLayer(cs, initPartOneKernel, initPartTwoKernel, initPartThreeKernel, prevWidth, prevHeight, prevCellsPerColumn, _layers[l], _layerDescs[l], l == _layers.size() - 1, minInitWeight, maxInitWeight, minInitCenter, maxInitCenter, minInitWeight, maxInitWeight, generator);

		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
		prevCellsPerColumn = _layerDescs[l]._cellsInColumn;
	}

	_layerColumnActivateKernel = cl::Kernel(program.getProgram(), "layerColumnActivate");
	_layerColumnInhibitBinaryKernel = cl::Kernel(program.getProgram(), "layerColumnInhibitBinary");
	_layerColumnInhibitKernel = cl::Kernel(program.getProgram(), "layerColumnInhibit");
	_layerColumnInhibitProbablisticKernel = cl::Kernel(program.getProgram(), "layerColumnInhibitProbablistic");
	_layerCellActivateKernel = cl::Kernel(program.getProgram(), "layerCellActivate");
	_layerCellWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerCellWeightUpdate");
	_layerCellWeightUpdateLastKernel = cl::Kernel(program.getProgram(), "layerCellWeightUpdateLast");
	_layerCellPredictKernel = cl::Kernel(program.getProgram(), "layerCellPredict");
	_layerCellPredictLastKernel = cl::Kernel(program.getProgram(), "layerCellPredictLast");
	_layerColumnWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerColumnWeightUpdate");
	_layerColumnPredictionKernel = cl::Kernel(program.getProgram(), "layerColumnPrediction");
	_layerColumnQKernel = cl::Kernel(program.getProgram(), "layerColumnQ");
	_layerColumnQLastKernel = cl::Kernel(program.getProgram(), "layerColumnQLast");
	_layerAssignQKernel = cl::Kernel(program.getProgram(), "layerAssignQ");

	_gaussianBlurXKernel = cl::Kernel(program.getProgram(), "gaussianBlurX");
	_gaussianBlurYKernel = cl::Kernel(program.getProgram(), "gaussianBlurY");

	_reconstructInputKernel = cl::Kernel(program.getProgram(), "reconstructInput");
	_inputBiasUpdateKernel = cl::Kernel(program.getProgram(), "inputBiasUpdate");
}

void HTMRL::initLayer(sys::ComputeSystem &cs, cl::Kernel &initPartOneKernel, cl::Kernel &initPartTwoKernel, cl::Kernel &initPartThreeKernel, int inputWidth, int inputHeight, int inputCellsPerColumn, Layer &layer, const LayerDesc &layerDesc, bool isTopmost, float minInitWeight, float maxInitWeight, float minInitCenter, float maxInitCenter, float minInitWidth, float maxInitWidth, std::mt19937 &generator) {
	struct Uint2 {
		unsigned int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};
	
	std::uniform_int_distribution<int> uniformDist(0, 10000);

	int receptiveFieldSize = std::pow(layerDesc._receptiveFieldRadius * 2 + 1, 2) + 1; // + 1 for bias
	int lateralConnectionsSize;

	// If not the last layer, add weights for additional context from next layer
	if (isTopmost)
		lateralConnectionsSize = layerDesc._numSegmentsPerCell * (std::pow(layerDesc._lateralConnectionRadius * 2 + 1, 2) * (layerDesc._cellsInColumn) + 1); // + 1 for bias
	else
		lateralConnectionsSize = layerDesc._numSegmentsPerCell * (std::pow(layerDesc._lateralConnectionRadius * 2 + 1, 2) * (layerDesc._cellsInColumn + 1) + 1); // + 1 for bias

	layer._columnActivations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	
	layer._columnStates = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height);
	
	layer._columnStatesProbabilistic = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._columnStateReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnActivationReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._columnFeedForwardWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, receptiveFieldSize);
	layer._columnFeedForwardWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, receptiveFieldSize);

	layer._cellStates = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellStatesPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._segmentStatesPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn * layerDesc._numSegmentsPerCell);
	layer._segmentStates = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn * layerDesc._numSegmentsPerCell);

	//layer._segmentWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn * layerDesc._numSegmentsPerCell);
	//layer._segmentWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn * layerDesc._numSegmentsPerCell);

	layer._cellQValues = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellQValuesPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._columnQValues = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	
	layer._columnPrevValues = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnPrevValuesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._columnTdErrors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._cellPredictions = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellPredictionsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._cellWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, lateralConnectionsSize);
	layer._cellWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, lateralConnectionsSize);

	layer._columnPredictions = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnPredictionsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	//layer._blurPing = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	//layer._blurPong = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._reconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inputWidth, inputHeight);
	
	layer._inputBiases = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), inputWidth, inputHeight);
	layer._inputBiasesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), inputWidth, inputHeight);

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = inputWidth;
		region[1] = inputHeight;
		region[2] = 1;

		cl_uint4 fillColor;

		fillColor.x = 0;
	
		cs.getQueue().enqueueFillImage(layer._reconstruction, fillColor, origin, region);
	}
	

	Uint2 seed1;
	seed1._x = uniformDist(generator);
	seed1._y = uniformDist(generator);

	initPartOneKernel.setArg(0, layer._columnActivations);
	initPartOneKernel.setArg(1, layer._columnStates);
	initPartOneKernel.setArg(2, layer._columnFeedForwardWeights);
	initPartOneKernel.setArg(3, layer._columnPrevValues);
	initPartOneKernel.setArg(4, layerDesc._cellsInColumn);
	initPartOneKernel.setArg(5, receptiveFieldSize);
	initPartOneKernel.setArg(6, lateralConnectionsSize);
	initPartOneKernel.setArg(7, seed1);
	initPartOneKernel.setArg(8, minInitCenter);
	initPartOneKernel.setArg(9, maxInitCenter);

	cs.getQueue().enqueueNDRangeKernel(initPartOneKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	Uint2 seed2;
	seed2._x = uniformDist(generator);
	seed2._y = uniformDist(generator);

	initPartTwoKernel.setArg(0, layer._cellStates);
	initPartTwoKernel.setArg(1, layer._segmentStates);
	initPartTwoKernel.setArg(2, layer._cellWeights);
	initPartTwoKernel.setArg(3, layer._cellPredictions);
	initPartTwoKernel.setArg(4, layer._cellQValues);
	initPartTwoKernel.setArg(5, layerDesc._cellsInColumn);
	initPartTwoKernel.setArg(6, receptiveFieldSize);
	initPartTwoKernel.setArg(7, lateralConnectionsSize);
	initPartTwoKernel.setArg(8, layerDesc._numSegmentsPerCell);
	initPartTwoKernel.setArg(9, seed2);
	initPartTwoKernel.setArg(10, minInitWeight);
	initPartTwoKernel.setArg(11, maxInitWeight);

	cs.getQueue().enqueueNDRangeKernel(initPartTwoKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	Uint2 seed3;
	seed3._x = uniformDist(generator);
	seed3._y = uniformDist(generator);

	initPartThreeKernel.setArg(0, layer._inputBiases);
	initPartThreeKernel.setArg(1, seed2);
	initPartThreeKernel.setArg(2, minInitWeight);
	initPartThreeKernel.setArg(3, maxInitWeight);

	cs.getQueue().enqueueNDRangeKernel(initPartThreeKernel, cl::NullRange, cl::NDRange(inputWidth, inputHeight));

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = layerDesc._width;
		region[1] = layerDesc._height;
		region[2] = 1;

		cs.getQueue().enqueueCopyImage(layer._columnStates, layer._columnStatesPrev, origin, origin, region);
	}

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = layerDesc._width;
		region[1] = layerDesc._height;
		region[2] = 1;

		cs.getQueue().enqueueCopyImage(layer._columnPredictions, layer._columnPredictionsPrev, origin, origin, region);
	}

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = layerDesc._width;
		region[1] = layerDesc._height;
		region[2] = 1;

		cs.getQueue().enqueueCopyImage(layer._columnPrevValues, layer._columnPrevValuesPrev, origin, origin, region);
	}

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = layerDesc._width;
		region[1] = layerDesc._height;
		region[2] = receptiveFieldSize;

		cs.getQueue().enqueueCopyImage(layer._columnFeedForwardWeights, layer._columnFeedForwardWeightsPrev, origin, origin, region);
	}

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = layerDesc._width;
		region[1] = layerDesc._height;
		region[2] = layerDesc._cellsInColumn;

		cs.getQueue().enqueueCopyImage(layer._cellStates, layer._cellStatesPrev, origin, origin, region);
	}

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = layerDesc._width;
		region[1] = layerDesc._height;
		region[2] = layerDesc._cellsInColumn * layerDesc._numSegmentsPerCell;

		cs.getQueue().enqueueCopyImage(layer._segmentStates, layer._segmentStatesPrev, origin, origin, region);
	}

	/*{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = layerDesc._width;
		region[1] = layerDesc._height;
		region[2] = layerDesc._cellsInColumn * layerDesc._numSegmentsPerCell;

		cs.getQueue().enqueueCopyImage(layer._segmentWeights, layer._segmentWeightsPrev, origin, origin, region);
	}*/

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = layerDesc._width;
		region[1] = layerDesc._height;
		region[2] = layerDesc._cellsInColumn;

		cs.getQueue().enqueueCopyImage(layer._cellQValues, layer._cellQValuesPrev, origin, origin, region);
	}

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = layerDesc._width;
		region[1] = layerDesc._height * layerDesc._cellsInColumn;
		region[2] = lateralConnectionsSize;

		cs.getQueue().enqueueCopyImage(layer._cellWeights, layer._cellWeightsPrev, origin, origin, region);
	}

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = layerDesc._width;
		region[1] = layerDesc._height;
		region[2] = layerDesc._cellsInColumn;

		cs.getQueue().enqueueCopyImage(layer._cellPredictions, layer._cellPredictionsPrev, origin, origin, region);
	}

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = inputWidth;
		region[1] = inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueCopyImage(layer._inputBiases, layer._inputBiasesPrev, origin, origin, region);
	}
}

void HTMRL::stepBegin(sys::ComputeSystem &cs, int addReplaySampleSteps, int maxReplayChainSize) {
	for (int l = 0; l < _layers.size(); l++) {
		std::swap(_layers[l]._columnStates, _layers[l]._columnStatesPrev);	
		std::swap(_layers[l]._columnPredictions, _layers[l]._columnPredictionsPrev);
		std::swap(_layers[l]._columnFeedForwardWeights, _layers[l]._columnFeedForwardWeightsPrev);
		std::swap(_layers[l]._columnPrevValues, _layers[l]._columnPrevValuesPrev);
		std::swap(_layers[l]._cellStates, _layers[l]._cellStatesPrev);
		std::swap(_layers[l]._segmentStates, _layers[l]._segmentStatesPrev);
		//std::swap(_layers[l]._segmentWeights, _layers[l]._segmentWeightsPrev);
		std::swap(_layers[l]._cellQValues, _layers[l]._cellQValuesPrev);
		std::swap(_layers[l]._cellPredictions, _layers[l]._cellPredictionsPrev);
		std::swap(_layers[l]._cellWeights, _layers[l]._cellWeightsPrev);
		std::swap(_layers[l]._inputBiases, _layers[l]._inputBiasesPrev);
	}

	if (_addReplaySampleStepCounter >= addReplaySampleSteps) {
		_addReplaySampleStepCounter = 0;

		if (_inputReplayChain.size() < maxReplayChainSize) {
			cl::size_t<3> origin;
			cl::size_t<3> region;

			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			region[0] = _inputWidth;
			region[1] = _inputHeight;
			region[2] = 1;

			cl::Image2D newSample = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

			cs.getQueue().enqueueCopyImage(_inputImage, newSample, origin, origin, region);

			_inputReplayChain.push_back(newSample);
		}
		else {
			cl::size_t<3> origin;
			cl::size_t<3> region;

			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			region[0] = _inputWidth;
			region[1] = _inputHeight;
			region[2] = 1;

			cl::Image2D temp = _inputReplayChain.back();

			_inputReplayChain.pop_back();

			cs.getQueue().enqueueCopyImage(_inputImage, temp, origin, origin, region);

			_inputReplayChain.push_back(temp);
		}
	}
	
	_addReplaySampleStepCounter++;
}

void HTMRL::spatialPoolLayer(sys::ComputeSystem &cs, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, Layer &layer, const LayerDesc &layerDesc, float columnDecay, std::mt19937 &generator) {
	struct Uint2 {
		unsigned int _x, _y;
	};

	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	std::uniform_int_distribution<int> uniformDist(0, 10000);

	Uint2 seed1;
	seed1._x = uniformDist(generator);
	seed1._y = uniformDist(generator);

	Uint2 seed2;
	seed2._x = uniformDist(generator);
	seed2._y = uniformDist(generator);

	Uint2 seed3;
	seed3._x = uniformDist(generator);
	seed3._y = uniformDist(generator);

	Uint2 seed4;
	seed4._x = uniformDist(generator);
	seed4._y = uniformDist(generator);

	Int2 inputSize;
	inputSize._x = prevLayerWidth;
	inputSize._y = prevLayerHeight;

	Int2 layerSize;
	layerSize._x = layerDesc._width;
	layerSize._y = layerDesc._height;

	Float2 layerSizeMinusOneInv;
	layerSizeMinusOneInv._x = 1.0f / (layerDesc._width - 1);
	layerSizeMinusOneInv._y = 1.0f / (layerDesc._height - 1);

	Float2 layerSizeInv;
	layerSizeInv._x = 1.0f / layerDesc._width;
	layerSizeInv._y = 1.0f / layerDesc._height;

	Int2 inputReceptiveFieldRadius;
	inputReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	inputReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Int2 inputSizeMinusOne;
	inputSizeMinusOne._x = prevLayerWidth - 1;
	inputSizeMinusOne._y = prevLayerHeight - 1;

	// Activation
	_layerColumnActivateKernel.setArg(0, prevLayerOutput);
	_layerColumnActivateKernel.setArg(1, layer._columnFeedForwardWeightsPrev);
	_layerColumnActivateKernel.setArg(2, layer._columnStatesPrev);
	_layerColumnActivateKernel.setArg(3, layer._columnActivations);
	_layerColumnActivateKernel.setArg(4, layerSizeMinusOneInv);
	_layerColumnActivateKernel.setArg(5, inputReceptiveFieldRadius);
	_layerColumnActivateKernel.setArg(6, inputSize);
	_layerColumnActivateKernel.setArg(7, inputSizeMinusOne);
	_layerColumnActivateKernel.setArg(8, seed1);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnActivateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	Int2 layerInhibitionRadius;
	layerInhibitionRadius._x = layerDesc._inhibitionRadius;
	layerInhibitionRadius._y = layerDesc._inhibitionRadius;

	int receptiveFieldSize = std::pow(layerDesc._receptiveFieldRadius * 2 + 1, 2) + 1;

	// Inhibition - not probablistic
	_layerColumnInhibitBinaryKernel.setArg(0, layer._columnActivations);
	_layerColumnInhibitBinaryKernel.setArg(1, layer._columnStatesPrev);
	_layerColumnInhibitBinaryKernel.setArg(2, layer._columnFeedForwardWeightsPrev);
	_layerColumnInhibitBinaryKernel.setArg(3, layer._columnStates);
	_layerColumnInhibitBinaryKernel.setArg(4, layerSize);
	_layerColumnInhibitBinaryKernel.setArg(5, layerSizeInv);
	_layerColumnInhibitBinaryKernel.setArg(6, layerInhibitionRadius);
	_layerColumnInhibitBinaryKernel.setArg(7, receptiveFieldSize);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnInhibitBinaryKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	// Inhibition
	/*_layerColumnInhibitProbablisticKernel.setArg(0, layer._columnActivations);
	_layerColumnInhibitProbablisticKernel.setArg(1, layer._columnStatesPrev);
	_layerColumnInhibitProbablisticKernel.setArg(2, layer._columnFeedForwardWeightsPrev);
	_layerColumnInhibitProbablisticKernel.setArg(3, layer._columnStatesProbabilistic);
	_layerColumnInhibitProbablisticKernel.setArg(4, layerSize);
	_layerColumnInhibitProbablisticKernel.setArg(5, layerSizeInv);
	_layerColumnInhibitProbablisticKernel.setArg(6, layerInhibitionRadius);
	_layerColumnInhibitProbablisticKernel.setArg(7, receptiveFieldSize);
	_layerColumnInhibitProbablisticKernel.setArg(8, seed2);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnInhibitProbablisticKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));*/

	Float2 inputSizeMinusOneInv;
	inputSizeMinusOneInv._x = 1.0f / (prevLayerWidth - 1);
	inputSizeMinusOneInv._y = 1.0f / (prevLayerHeight - 1);

	Int2 reconstructionReceptiveFieldRadii;
	reconstructionReceptiveFieldRadii._x = std::ceil(static_cast<float>(layerDesc._width) / prevLayerWidth * layerDesc._receptiveFieldRadius);
	reconstructionReceptiveFieldRadii._y = std::ceil(static_cast<float>(layerDesc._height) / prevLayerHeight * layerDesc._receptiveFieldRadius);

	Int2 layerSizeMinusOne;
	layerSizeMinusOne._x = layerDesc._width - 1;
	layerSizeMinusOne._y = layerDesc._height - 1;

	// Reconstruct
	_reconstructInputKernel.setArg(0, layer._columnFeedForwardWeightsPrev);
	_reconstructInputKernel.setArg(1, layer._inputBiasesPrev);
	_reconstructInputKernel.setArg(2, layer._columnStates);
	_reconstructInputKernel.setArg(3, layer._reconstruction);
	_reconstructInputKernel.setArg(4, reconstructionReceptiveFieldRadii);
	_reconstructInputKernel.setArg(5, inputReceptiveFieldRadius);
	_reconstructInputKernel.setArg(6, inputSizeMinusOne);
	_reconstructInputKernel.setArg(7, inputSizeMinusOneInv);
	_reconstructInputKernel.setArg(8, layerSize);
	_reconstructInputKernel.setArg(9, layerSizeMinusOne);
	_reconstructInputKernel.setArg(10, layerSizeMinusOneInv);

	cs.getQueue().enqueueNDRangeKernel(_reconstructInputKernel, cl::NullRange, cl::NDRange(prevLayerWidth, prevLayerHeight));

	// Activate from reconstruction to get stateReconstruction

	// Activation
	_layerColumnActivateKernel.setArg(0, layer._reconstruction);
	_layerColumnActivateKernel.setArg(1, layer._columnFeedForwardWeightsPrev);
	_layerColumnActivateKernel.setArg(2, layer._columnStatesPrev);
	_layerColumnActivateKernel.setArg(3, layer._columnActivationReconstruction);
	_layerColumnActivateKernel.setArg(4, layerSizeMinusOneInv);
	_layerColumnActivateKernel.setArg(5, inputReceptiveFieldRadius);
	_layerColumnActivateKernel.setArg(6, inputSize);
	_layerColumnActivateKernel.setArg(7, inputSizeMinusOne);
	_layerColumnActivateKernel.setArg(8, seed3);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnActivateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	// Inhibition
	_layerColumnInhibitKernel.setArg(0, layer._columnActivationReconstruction);
	_layerColumnInhibitKernel.setArg(1, layer._columnStatesPrev);
	_layerColumnInhibitKernel.setArg(2, layer._columnFeedForwardWeightsPrev);
	_layerColumnInhibitKernel.setArg(3, layer._columnStateReconstruction);
	_layerColumnInhibitKernel.setArg(4, layerSize);
	_layerColumnInhibitKernel.setArg(5, layerSizeInv);
	_layerColumnInhibitKernel.setArg(6, layerInhibitionRadius);
	_layerColumnInhibitKernel.setArg(7, receptiveFieldSize);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnInhibitKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::cellActivateLayer(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, float cellStateDecay, std::mt19937 &generator) {
	struct Uint2 {
		unsigned int _x, _y;
	};

	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	std::uniform_int_distribution<int> uniformDist(0, 10000);

	Uint2 seed;
	seed._x = uniformDist(generator);
	seed._y = uniformDist(generator);

	Int2 layerSize;
	layerSize._x = layerDesc._width;
	layerSize._y = layerDesc._height;

	Float2 layerSizeMinusOneInv;
	layerSizeMinusOneInv._x = 1.0f / (layerDesc._width - 1);
	layerSizeMinusOneInv._y = 1.0f / (layerDesc._height - 1);

	Float2 layerSizeInv;
	layerSizeInv._x = 1.0f / layerDesc._width;
	layerSizeInv._y = 1.0f / layerDesc._height;

	Int2 lateralConnectionRadii;
	lateralConnectionRadii._x = layerDesc._lateralConnectionRadius;
	lateralConnectionRadii._y = layerDesc._lateralConnectionRadius;

	// Cell activation
	_layerCellActivateKernel.setArg(0, layer._columnStates);
	_layerCellActivateKernel.setArg(1, layer._cellStatesPrev);
	_layerCellActivateKernel.setArg(2, layer._cellPredictionsPrev);
	_layerCellActivateKernel.setArg(3, layer._cellWeightsPrev);
	_layerCellActivateKernel.setArg(4, layer._columnPredictionsPrev);
	_layerCellActivateKernel.setArg(5, layer._cellStates);
	_layerCellActivateKernel.setArg(6, layerDesc._cellsInColumn);
	_layerCellActivateKernel.setArg(7, lateralConnectionRadii);
	_layerCellActivateKernel.setArg(8, cellStateDecay);
	_layerCellActivateKernel.setArg(9, seed);

	cs.getQueue().enqueueNDRangeKernel(_layerCellActivateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::predictLayer(sys::ComputeSystem &cs, cl::Image2D &nextLayerPrediction, cl::Image2D &nextLayerPredictionPrev, int nextLayerWidth, int nextLayerHeight, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};
	
	Int2 layerSize;
	layerSize._x = layerDesc._width;
	layerSize._y = layerDesc._height;

	Float2 layerSizeMinusOneInv;
	layerSizeMinusOneInv._x = 1.0f / (layerDesc._width - 1);
	layerSizeMinusOneInv._y = 1.0f / (layerDesc._height - 1);

	Int2 lateralConnectionRadii;
	lateralConnectionRadii._x = layerDesc._lateralConnectionRadius;
	lateralConnectionRadii._y = layerDesc._lateralConnectionRadius;

	// Cell prediction
	Int2 nextLayerSize;
	nextLayerSize._x = nextLayerWidth;
	nextLayerSize._y = nextLayerHeight;

	Int2 nextLayerSizeMinusOne;
	nextLayerSizeMinusOne._x = nextLayerWidth - 1;
	nextLayerSizeMinusOne._y = nextLayerHeight - 1;

	_layerCellPredictKernel.setArg(0, layer._cellStates);
	_layerCellPredictKernel.setArg(1, layer._cellStatesPrev);
	_layerCellPredictKernel.setArg(2, layer._cellWeights);
	_layerCellPredictKernel.setArg(3, nextLayerPrediction);
	_layerCellPredictKernel.setArg(4, nextLayerPredictionPrev);
	_layerCellPredictKernel.setArg(5, layer._cellPredictions);
	_layerCellPredictKernel.setArg(6, layer._segmentStates);
	_layerCellPredictKernel.setArg(7, layerDesc._cellsInColumn);
	_layerCellPredictKernel.setArg(8, layerSize);
	_layerCellPredictKernel.setArg(9, lateralConnectionRadii);
	_layerCellPredictKernel.setArg(10, layerDesc._numSegmentsPerCell);
	_layerCellPredictKernel.setArg(11, layerSizeMinusOneInv);
	_layerCellPredictKernel.setArg(12, nextLayerSize);
	_layerCellPredictKernel.setArg(13, nextLayerSizeMinusOne);

	cs.getQueue().enqueueNDRangeKernel(_layerCellPredictKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	// Column prediction
	_layerColumnPredictionKernel.setArg(0, layer._cellPredictions);
	_layerColumnPredictionKernel.setArg(1, layer._cellStates);
	_layerColumnPredictionKernel.setArg(2, layer._columnPredictions);
	_layerColumnPredictionKernel.setArg(3, layerDesc._cellsInColumn);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnPredictionKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::predictLayerLast(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Int2 layerSize;
	layerSize._x = layerDesc._width;
	layerSize._y = layerDesc._height;

	Float2 layerSizeInv;
	layerSizeInv._x = 1.0f / layerDesc._width;
	layerSizeInv._y = 1.0f / layerDesc._height;

	Int2 lateralConnectionRadii;
	lateralConnectionRadii._x = layerDesc._lateralConnectionRadius;
	lateralConnectionRadii._y = layerDesc._lateralConnectionRadius;

	// Cell prediction
	_layerCellPredictLastKernel.setArg(0, layer._cellStates);
	_layerCellPredictLastKernel.setArg(1, layer._cellStatesPrev);
	_layerCellPredictLastKernel.setArg(2, layer._cellWeights);
	_layerCellPredictLastKernel.setArg(3, layer._cellPredictions);
	_layerCellPredictLastKernel.setArg(4, layer._segmentStates);
	_layerCellPredictLastKernel.setArg(5, layerDesc._cellsInColumn);
	_layerCellPredictLastKernel.setArg(6, layerSize);
	_layerCellPredictLastKernel.setArg(7, lateralConnectionRadii);
	_layerCellPredictLastKernel.setArg(8, layerDesc._numSegmentsPerCell);

	cs.getQueue().enqueueNDRangeKernel(_layerCellPredictLastKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	// Column prediction
	_layerColumnPredictionKernel.setArg(0, layer._cellPredictions);
	_layerColumnPredictionKernel.setArg(1, layer._cellStates);
	_layerColumnPredictionKernel.setArg(2, layer._columnPredictions);
	_layerColumnPredictionKernel.setArg(3, layerDesc._cellsInColumn);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnPredictionKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::activate(std::vector<float> &input, sys::ComputeSystem &cs, float reward, float alpha, float gamma, float columnDecay, float cellStateDecay, float columnConnectionAlpha, float columnConnectionBeta, float columnConnectionGamma, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionGamma, float cellConnectionTemperature, float cellWeightEligibilityDecay, int maxReplayChainSize, int numReplaySamples, int addSampleSteps, unsigned long seed) {
	// Create buffer from input
	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueWriteImage(_inputImage, CL_TRUE, origin, region, 0, 0, &input[0]);
	}

	std::mt19937 generator(seed);

	learnSpatialReplay(cs, cellStateDecay, columnConnectionAlpha, columnConnectionBeta, columnConnectionGamma, maxReplayChainSize, numReplaySamples, seed);

	for (int l = 0; l < _layers.size(); l++)
		cellActivateLayer(cs, _layers[l], _layerDescs[l], cellStateDecay, generator);

	for (int l = _layers.size() - 1; l >= 0; l--) {
		if (l == _layers.size() - 1)
			determineLayerColumnQLast(cs, _layers[l], _layerDescs[l]);
		else
			determineLayerColumnQ(cs, _layers[l], _layerDescs[l], _layers[l + 1], _layerDescs[l + 1]);
	}

	float value = retreiveQ(cs);

	float tdError = reward + gamma * value - _prevValue;

	std::cout << "R: " << reward << "Q: " << reward + gamma * value << " T: " << tdError << std::endl;

	_prevValue = value;

	for (int l = _layers.size() - 1; l >= 0; l--)
		assignLayerQ(cs, _layers[l], _layerDescs[l], alpha * tdError);

	learnTemporal(cs, tdError, cellConnectionAlpha * (tdError > 0.0f ? 1.0f : 0.0f), cellConnectionBeta, cellConnectionGamma, cellConnectionTemperature, cellWeightEligibilityDecay, seed + 1);

	for (int l = _layers.size() - 1; l >= 0; l--) {
		if (l == _layers.size() - 1)
			predictLayerLast(cs, _layers[l], _layerDescs[l], generator);
		else
			predictLayer(cs, _layers[l + 1]._columnPredictions, _layers[l + 1]._columnPredictionsPrev, _layerDescs[l + 1]._width, _layerDescs[l + 1]._height, _layers[l], _layerDescs[l], generator);
	}

	/*pPrevLayerOutput = &_inputImage;
	prevLayerWidth = _inputWidth;
	prevLayerHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		learnLayerSpatial(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], columnConnectionAlpha, columnConnectionBeta, columnConnectionGamma, generator);

		pPrevLayerOutput = &_layers[l]._columnStates;
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}*/
}

void HTMRL::determineLayerColumnQ(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc, Layer &nextLayer, LayerDesc &nextLayerDesc) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Int2 layerSize;
	layerSize._x = layerDesc._width;
	layerSize._y = layerDesc._height;

	Int2 nextLayerSize;
	nextLayerSize._x = nextLayerDesc._width;
	nextLayerSize._y = nextLayerDesc._height;

	Int2 nextLayerSizeMinusOne;
	nextLayerSizeMinusOne._x = nextLayerDesc._width - 1;
	nextLayerSizeMinusOne._y = nextLayerDesc._height - 1;

	Float2 layerSizeMinusOneInv;
	layerSizeMinusOneInv._x = 1.0f / (layerDesc._width - 1);
	layerSizeMinusOneInv._y = 1.0f / (layerDesc._height - 1);

	_layerColumnQKernel.setArg(0, layer._cellQValuesPrev);
	_layerColumnQKernel.setArg(1, layer._cellStatesPrev);
	_layerColumnQKernel.setArg(2, layer._cellStates);
	_layerColumnQKernel.setArg(3, layer._columnStates);
	_layerColumnQKernel.setArg(4, nextLayer._columnStates);
	_layerColumnQKernel.setArg(5, nextLayer._columnQValues);
	_layerColumnQKernel.setArg(6, layer._columnQValues);
	_layerColumnQKernel.setArg(7, layerDesc._cellsInColumn);
	_layerColumnQKernel.setArg(8, layerSizeMinusOneInv);
	_layerColumnQKernel.setArg(9, nextLayerSize);
	_layerColumnQKernel.setArg(10, nextLayerSizeMinusOne);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnQKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::determineLayerColumnQLast(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	_layerColumnQLastKernel.setArg(0, layer._cellQValuesPrev);
	_layerColumnQLastKernel.setArg(1, layer._cellStatesPrev);
	_layerColumnQLastKernel.setArg(2, layer._cellStates);
	_layerColumnQLastKernel.setArg(3, layer._columnQValues);
	_layerColumnQLastKernel.setArg(4, layerDesc._cellsInColumn);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnQLastKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

float HTMRL::retreiveQ(sys::ComputeSystem &cs) {
	float total = 0.0f;

	float sum = 0.0f;
	float divisor = 0.0f;

	for (int l = 0; l < _layers.size(); l++) {
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _layerDescs[l]._width;
		region[1] = _layerDescs[l]._height;
		region[2] = 1;

		std::vector<float> layerQ(_layerDescs[l]._width * _layerDescs[l]._height);

		cs.getQueue().enqueueReadImage(_layers[l]._columnQValues, CL_TRUE, origin, region, 0, 0, &layerQ[0]);

		std::vector<float> layerColumns(_layerDescs[l]._width * _layerDescs.front()._height * 2);

		cs.getQueue().enqueueReadImage(_layers[l]._columnStates, CL_TRUE, origin, region, 0, 0, &layerColumns[0]);

		for (int i = 0; i < layerQ.size(); i++) {
			sum += layerQ[i] * _layerDescs[l]._qImportance * layerColumns[i * 2];
			divisor += _layerDescs[l]._qImportance * layerColumns[i * 2];
		}
	}

	if (divisor == 0.0f)
		return 0.0f;

	return sum / divisor;
}

void HTMRL::assignLayerQ(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc, float alpha) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Int2 layerSize;
	layerSize._x = layerDesc._width;
	layerSize._y = layerDesc._height;

	Float2 layerSizeMinusOneInv;
	layerSizeMinusOneInv._x = 1.0f / (layerDesc._width - 1);
	layerSizeMinusOneInv._y = 1.0f / (layerDesc._height - 1);

	_layerAssignQKernel.setArg(0, layer._cellQValuesPrev);
	_layerAssignQKernel.setArg(1, layer._cellStatesPrev);
	_layerAssignQKernel.setArg(2, layer._cellQValues);
	_layerAssignQKernel.setArg(3, layerDesc._cellsInColumn);
	_layerAssignQKernel.setArg(4, alpha);

	cs.getQueue().enqueueNDRangeKernel(_layerAssignQKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::learnLayerSpatial(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float alpha, float beta, float gamma, std::mt19937 &generator) {
	struct Uint2 {
		unsigned int _x, _y;
	};

	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	std::uniform_int_distribution<int> uniformDist(0, 10000);

	Uint2 seed;
	seed._x = uniformDist(generator);
	seed._y = uniformDist(generator);

	Int2 inputSize;
	inputSize._x = prevLayerWidth;
	inputSize._y = prevLayerHeight;

	Int2 layerSize;
	layerSize._x = layerDesc._width;
	layerSize._y = layerDesc._height;

	Float2 layerSizeMinusOneInv;
	layerSizeMinusOneInv._x = 1.0f / (layerDesc._width - 1);
	layerSizeMinusOneInv._y = 1.0f / (layerDesc._height - 1);

	Int2 inputReceptiveFieldRadius;
	inputReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	inputReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	int receptiveFieldSize = std::pow(layerDesc._receptiveFieldRadius * 2 + 1, 2) + 1;

	Int2 influenceRadius;
	influenceRadius._x = layerDesc._columnInfluenceRadius;
	influenceRadius._y = layerDesc._columnInfluenceRadius;

	Int2 inputSizeMinusOne;
	inputSizeMinusOne._x = layerDesc._width - 1;
	inputSizeMinusOne._y = layerDesc._height - 1;

	Int2 inhibitionRadii;
	inhibitionRadii._x = layerDesc._inhibitionRadius;
	inhibitionRadii._y = layerDesc._inhibitionRadius;

	// Column weight update
	_layerColumnWeightUpdateKernel.setArg(0, layer._reconstruction);
	_layerColumnWeightUpdateKernel.setArg(1, layer._columnStateReconstruction);
	_layerColumnWeightUpdateKernel.setArg(2, prevLayerOutput);
	_layerColumnWeightUpdateKernel.setArg(3, layer._columnActivations);
	_layerColumnWeightUpdateKernel.setArg(4, layer._columnStatesProbabilistic);
	_layerColumnWeightUpdateKernel.setArg(5, layer._columnStates);
	_layerColumnWeightUpdateKernel.setArg(6, layer._columnPredictions);
	_layerColumnWeightUpdateKernel.setArg(7, layer._columnFeedForwardWeightsPrev);
	_layerColumnWeightUpdateKernel.setArg(8, layer._columnFeedForwardWeights);
	_layerColumnWeightUpdateKernel.setArg(9, layerSize);
	_layerColumnWeightUpdateKernel.setArg(10, layerSizeMinusOneInv);
	_layerColumnWeightUpdateKernel.setArg(11, inputReceptiveFieldRadius);
	_layerColumnWeightUpdateKernel.setArg(12, inhibitionRadii);
	_layerColumnWeightUpdateKernel.setArg(13, inputSize);
	_layerColumnWeightUpdateKernel.setArg(14, inputSizeMinusOne);
	_layerColumnWeightUpdateKernel.setArg(15, receptiveFieldSize);
	_layerColumnWeightUpdateKernel.setArg(16, alpha);
	_layerColumnWeightUpdateKernel.setArg(17, beta);
	_layerColumnWeightUpdateKernel.setArg(18, gamma);
	_layerColumnWeightUpdateKernel.setArg(19, seed);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	Float2 inputSizeMinusOneInv;
	inputSizeMinusOneInv._x = 1.0f / (prevLayerWidth - 1);
	inputSizeMinusOneInv._y = 1.0f / (prevLayerHeight - 1);

	Int2 reconstructionReceptiveFieldRadii;
	reconstructionReceptiveFieldRadii._x = std::ceil(static_cast<float>(layerDesc._width) / prevLayerWidth * layerDesc._receptiveFieldRadius);
	reconstructionReceptiveFieldRadii._y = std::ceil(static_cast<float>(layerDesc._height) / prevLayerHeight * layerDesc._receptiveFieldRadius);

	Int2 layerSizeMinusOne;
	layerSizeMinusOne._x = layerDesc._width - 1;
	layerSizeMinusOne._y = layerDesc._height - 1;

	// Reconstruct
	_inputBiasUpdateKernel.setArg(0, prevLayerOutput);
	_inputBiasUpdateKernel.setArg(1, layer._reconstruction);
	_inputBiasUpdateKernel.setArg(2, layer._inputBiasesPrev);
	_inputBiasUpdateKernel.setArg(3, layer._inputBiases);
	_inputBiasUpdateKernel.setArg(4, gamma);

	cs.getQueue().enqueueNDRangeKernel(_inputBiasUpdateKernel, cl::NullRange, cl::NDRange(prevLayerWidth, prevLayerHeight));
}

void HTMRL::learnLayerTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float tdError, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionGamma, float cellConnectionTemperature, float cellWeightEligibilityDecay, std::mt19937 &generator) {
	struct Uint2 {
		unsigned int _x, _y;
	};

	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	std::uniform_int_distribution<int> uniformDist(0, 10000);

	Uint2 seed;
	seed._x = uniformDist(generator);
	seed._y = uniformDist(generator);

	Int2 inputSize;
	inputSize._x = prevLayerWidth;
	inputSize._y = prevLayerHeight;

	Int2 layerSize;
	layerSize._x = layerDesc._width;
	layerSize._y = layerDesc._height;

	Float2 inputSizeInv;
	inputSizeInv._x = 1.0f / prevLayerWidth;
	inputSizeInv._y = 1.0f / prevLayerHeight;

	Float2 layerSizeMinusOneInv;
	layerSizeMinusOneInv._x = 1.0f / (layerDesc._width - 1);
	layerSizeMinusOneInv._y = 1.0f / (layerDesc._height - 1);

	Int2 inputReceptiveFieldRadius;
	inputReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	inputReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Int2 layerReceptiveFieldRadius;
	layerReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	layerReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Int2 lateralConnectionRadii;
	lateralConnectionRadii._x = layerDesc._lateralConnectionRadius;
	lateralConnectionRadii._y = layerDesc._lateralConnectionRadius;

	// Lateral weight update
	Int2 nextLayerSize;
	nextLayerSize._x = nextLayerWidth;
	nextLayerSize._y = nextLayerHeight;

	Int2 nextLayerSizeMinusOne;
	nextLayerSizeMinusOne._x = nextLayerWidth - 1;
	nextLayerSizeMinusOne._y = nextLayerHeight - 1;

	_layerCellWeightUpdateKernel.setArg(0, layer._columnStates);
	_layerCellWeightUpdateKernel.setArg(1, layer._columnPredictionsPrev);
	_layerCellWeightUpdateKernel.setArg(2, layer._cellPredictionsPrev);
	_layerCellWeightUpdateKernel.setArg(3, layer._cellStates);
	_layerCellWeightUpdateKernel.setArg(4, layer._cellStatesPrev);
	_layerCellWeightUpdateKernel.setArg(5, nextLayerPrediction);
	_layerCellWeightUpdateKernel.setArg(6, layer._segmentStatesPrev);
	_layerCellWeightUpdateKernel.setArg(7, layer._cellWeightsPrev);
	_layerCellWeightUpdateKernel.setArg(8, layer._cellWeights);
	_layerCellWeightUpdateKernel.setArg(9, layerDesc._cellsInColumn);
	_layerCellWeightUpdateKernel.setArg(10, layerSize);
	_layerCellWeightUpdateKernel.setArg(11, lateralConnectionRadii);
	_layerCellWeightUpdateKernel.setArg(12, layerDesc._numSegmentsPerCell);
	_layerCellWeightUpdateKernel.setArg(13, layerSizeMinusOneInv);
	_layerCellWeightUpdateKernel.setArg(14, nextLayerSize);
	_layerCellWeightUpdateKernel.setArg(15, nextLayerSizeMinusOne);
	_layerCellWeightUpdateKernel.setArg(16, tdError);
	_layerCellWeightUpdateKernel.setArg(17, cellConnectionAlpha);
	_layerCellWeightUpdateKernel.setArg(18, cellConnectionBeta);
	_layerCellWeightUpdateKernel.setArg(19, cellConnectionGamma);
	_layerCellWeightUpdateKernel.setArg(20, cellConnectionTemperature);
	_layerCellWeightUpdateKernel.setArg(21, cellWeightEligibilityDecay);

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::learnLayerTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float tdError, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionGamma, float cellConnectionTemperature, float cellWeightEligibilityDecay, std::mt19937 &generator) {
	struct Uint2 {
		unsigned int _x, _y;
	};

	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	std::uniform_int_distribution<int> uniformDist(0, 10000);

	Uint2 seed;
	seed._x = uniformDist(generator);
	seed._y = uniformDist(generator);

	Int2 inputSize;
	inputSize._x = prevLayerWidth;
	inputSize._y = prevLayerHeight;

	Int2 layerSize;
	layerSize._x = layerDesc._width;
	layerSize._y = layerDesc._height;

	Float2 inputSizeInv;
	inputSizeInv._x = 1.0f / prevLayerWidth;
	inputSizeInv._y = 1.0f / prevLayerHeight;

	Float2 layerSizeInv;
	layerSizeInv._x = 1.0f / layerDesc._width;
	layerSizeInv._y = 1.0f / layerDesc._height;

	Int2 lateralConnectionRadii;
	lateralConnectionRadii._x = layerDesc._lateralConnectionRadius;
	lateralConnectionRadii._y = layerDesc._lateralConnectionRadius;

	// Lateral weight update
	_layerCellWeightUpdateLastKernel.setArg(0, layer._columnStates);
	_layerCellWeightUpdateLastKernel.setArg(1, layer._columnPredictionsPrev);
	_layerCellWeightUpdateLastKernel.setArg(2, layer._cellPredictionsPrev);
	_layerCellWeightUpdateLastKernel.setArg(3, layer._cellStates);
	_layerCellWeightUpdateLastKernel.setArg(4, layer._cellStatesPrev);
	_layerCellWeightUpdateLastKernel.setArg(5, layer._segmentStatesPrev);
	_layerCellWeightUpdateLastKernel.setArg(6, layer._cellWeightsPrev);
	_layerCellWeightUpdateLastKernel.setArg(7, layer._cellWeights);
	_layerCellWeightUpdateLastKernel.setArg(8, layerDesc._cellsInColumn);
	_layerCellWeightUpdateLastKernel.setArg(9, layerSize);
	_layerCellWeightUpdateLastKernel.setArg(10, lateralConnectionRadii);
	_layerCellWeightUpdateLastKernel.setArg(11, layerDesc._numSegmentsPerCell);
	_layerCellWeightUpdateLastKernel.setArg(12, tdError);
	_layerCellWeightUpdateLastKernel.setArg(13, cellConnectionAlpha);
	_layerCellWeightUpdateLastKernel.setArg(14, cellConnectionBeta);
	_layerCellWeightUpdateLastKernel.setArg(15, cellConnectionGamma);
	_layerCellWeightUpdateLastKernel.setArg(16, cellConnectionTemperature);
	_layerCellWeightUpdateLastKernel.setArg(17, cellWeightEligibilityDecay);

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateLastKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::learnSpatialReplay(sys::ComputeSystem &cs, float cellStateDecay, float alpha, float beta, float gamma, int maxReplayChainSize, int numReplaySamples, unsigned long seed) {
	std::mt19937 generator(seed);

	std::uniform_int_distribution<int> sampleDist(0, _inputReplayChain.size());

	for (int i = 0; i < numReplaySamples; i++) {
		int sampleIndex = sampleDist(generator);

		if (sampleIndex == 0) {
			// Replay input
			cl::Image2D* pPrevLayerOutput = &_inputImage;
			int prevLayerWidth = _inputWidth;
			int prevLayerHeight = _inputHeight;

			for (int l = 0; l < _layers.size(); l++) {
				spatialPoolLayer(cs, *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l], _layerDescs[l], 0.0f, generator);
				learnLayerSpatial(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], alpha, beta, gamma, generator);

				pPrevLayerOutput = &_layers[l]._columnStates;
				prevLayerWidth = _layerDescs[l]._width;
				prevLayerHeight = _layerDescs[l]._height;
			}
		}
		else {
			int index = 0;

			cl::Image2D* pPrevLayerOutput;

			for (std::list<cl::Image2D>::iterator it = _inputReplayChain.begin(); it != _inputReplayChain.end(); it++, index++) {
				if (index >= sampleIndex - 1) {
					pPrevLayerOutput = &(*it);
					break;
				}
			}

			// Replay input
			int prevLayerWidth = _inputWidth;
			int prevLayerHeight = _inputHeight;

			for (int l = 0; l < _layers.size(); l++) {
				spatialPoolLayer(cs, *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l], _layerDescs[l], 0.0f, generator);
				learnLayerSpatial(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], alpha, beta, gamma, generator);

				pPrevLayerOutput = &_layers[l]._columnStates;
				prevLayerWidth = _layerDescs[l]._width;
				prevLayerHeight = _layerDescs[l]._height;
			}
		}

		for (int l = 0; l < _layers.size(); l++) {
			std::swap(_layers[l]._columnFeedForwardWeights, _layers[l]._columnFeedForwardWeightsPrev);
			std::swap(_layers[l]._inputBiases, _layers[l]._inputBiasesPrev);
		}
	}

	// Replay input to set state properly
	cl::Image2D* pPrevLayerOutput = &_inputImage;
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		spatialPoolLayer(cs, *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l], _layerDescs[l], 0.0f, generator);
		learnLayerSpatial(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], alpha, beta, gamma, generator);

		pPrevLayerOutput = &_layers[l]._columnStates;
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}
}

void HTMRL::learnTemporal(sys::ComputeSystem &cs, float tdError, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionGamma, float cellConnectionTemperature, float cellWeightEligibilityDecay, unsigned long seed) {
	std::mt19937 generator(seed);

	cl::Image2D* pPrevLayerOutput = &_inputImage;
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		if (l == _layers.size() - 1)
			learnLayerTemporalLast(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], tdError, cellConnectionAlpha, cellConnectionBeta, cellConnectionGamma, cellConnectionTemperature, cellWeightEligibilityDecay, generator);
		else
			learnLayerTemporal(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l + 1]._columnPredictionsPrev, _layerDescs[l + 1]._width, _layerDescs[l + 1]._width, _layerDescs[l], tdError, cellConnectionAlpha, cellConnectionBeta, cellConnectionGamma, cellConnectionTemperature, cellWeightEligibilityDecay, generator);

		pPrevLayerOutput = &_layers[l]._columnStates;
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}
}

void HTMRL::gaussianBlur(sys::ComputeSystem &cs, cl::Image2D &source, cl::Image2D &ping, cl::Image2D &pong, int imageSizeX, int imageSizeY, int passes, float kernelWidth) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Float2 imageSizeInv;
	imageSizeInv._x = 1.0f / imageSizeX;
	imageSizeInv._y = 1.0f / imageSizeY;

	// Blur source to ping
	_gaussianBlurXKernel.setArg(0, source);
	_gaussianBlurXKernel.setArg(1, ping);
	_gaussianBlurXKernel.setArg(2, imageSizeInv);
	_gaussianBlurXKernel.setArg(3, kernelWidth * imageSizeInv._x);

	cs.getQueue().enqueueNDRangeKernel(_gaussianBlurXKernel, cl::NullRange, cl::NDRange(imageSizeX, imageSizeY));

	for (int p = 0; p < passes - 1; p++) {
		_gaussianBlurYKernel.setArg(0, ping);
		_gaussianBlurYKernel.setArg(1, pong);
		_gaussianBlurYKernel.setArg(2, imageSizeInv);
		_gaussianBlurYKernel.setArg(3, kernelWidth * imageSizeInv._y);

		cs.getQueue().enqueueNDRangeKernel(_gaussianBlurYKernel, cl::NullRange, cl::NDRange(imageSizeX, imageSizeY));

		_gaussianBlurXKernel.setArg(0, pong);
		_gaussianBlurXKernel.setArg(1, ping);
		_gaussianBlurXKernel.setArg(2, imageSizeInv);
		_gaussianBlurXKernel.setArg(3, kernelWidth * imageSizeInv._x);

		cs.getQueue().enqueueNDRangeKernel(_gaussianBlurXKernel, cl::NullRange, cl::NDRange(imageSizeX, imageSizeY));
	}

	_gaussianBlurYKernel.setArg(0, ping);
	_gaussianBlurYKernel.setArg(1, pong);
	_gaussianBlurYKernel.setArg(2, imageSizeInv);
	_gaussianBlurYKernel.setArg(3, kernelWidth * imageSizeInv._y);

	cs.getQueue().enqueueNDRangeKernel(_gaussianBlurYKernel, cl::NullRange, cl::NDRange(imageSizeX, imageSizeY));
}

void HTMRL::getReconstructedPrediction(std::vector<float> &prediction, sys::ComputeSystem &cs) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Int2 layerSize;
	layerSize._x = _layerDescs.front()._width;
	layerSize._y = _layerDescs.front()._height;

	Int2 inputSizeMinusOne;
	inputSizeMinusOne._x = _inputWidth - 1;
	inputSizeMinusOne._y = _inputHeight - 1;

	Float2 inputSizeMinusOneInv;
	inputSizeMinusOneInv._x = 1.0f / (_inputWidth - 1);
	inputSizeMinusOneInv._y = 1.0f / (_inputHeight - 1);

	Int2 reconstructionReceptiveFieldRadii;
	reconstructionReceptiveFieldRadii._x = std::ceil(static_cast<float>(_layerDescs.front()._width) / _inputWidth * _layerDescs.front()._receptiveFieldRadius);
	reconstructionReceptiveFieldRadii._y = std::ceil(static_cast<float>(_layerDescs.front()._height) / _inputHeight * _layerDescs.front()._receptiveFieldRadius);

	Int2 sdrReceptiveFieldRadii;
	sdrReceptiveFieldRadii._x = _layerDescs.front()._receptiveFieldRadius;
	sdrReceptiveFieldRadii._y = _layerDescs.front()._receptiveFieldRadius;

	Int2 sdrSizeMinusOne;
	sdrSizeMinusOne._x = _layerDescs.front()._width - 1;
	sdrSizeMinusOne._y = _layerDescs.front()._height - 1;

	Float2 sdrSizeMinusOneInv;
	sdrSizeMinusOneInv._x = 1.0f / (_layerDescs.front()._width - 1);
	sdrSizeMinusOneInv._y = 1.0f / (_layerDescs.front()._height - 1);

	_reconstructInputKernel.setArg(0, _layers.front()._columnFeedForwardWeights);
	_reconstructInputKernel.setArg(1, _layers.front()._inputBiases);
	_reconstructInputKernel.setArg(2, _layers.front()._columnPredictions);
	_reconstructInputKernel.setArg(3, _reconstructedPrediction);
	_reconstructInputKernel.setArg(4, reconstructionReceptiveFieldRadii);
	_reconstructInputKernel.setArg(5, sdrReceptiveFieldRadii);
	_reconstructInputKernel.setArg(6, inputSizeMinusOne);
	_reconstructInputKernel.setArg(7, inputSizeMinusOneInv);
	_reconstructInputKernel.setArg(8, layerSize);
	_reconstructInputKernel.setArg(9, sdrSizeMinusOne);
	_reconstructInputKernel.setArg(10, sdrSizeMinusOneInv);

	cs.getQueue().enqueueNDRangeKernel(_reconstructInputKernel, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));

	if (prediction.size() != _input.size())
		prediction.resize(_input.size());

	// Read prediction
	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueReadImage(_reconstructedPrediction, CL_TRUE, origin, region, 0, 0, &prediction[0]);
	}
}

void HTMRL::step(sys::ComputeSystem &cs, float reward, float reconstructionAlpha, float columnDecay, float cellStateDecay, float columnConnectionAlpha, float columnConnectionBeta, float columnConnectionGamma, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionGamma, float cellConnectionTemperature, float cellWeightEligibilityDecay, float alpha, float gamma, float breakChance, float perturbationStdDev, int maxReplayChainSize, int numReplaySamples, int addReplaySampleSteps, std::mt19937 &generator) {
	std::uniform_int_distribution<int> seedDist(0, 10000);

	unsigned long seed = seedDist(generator);

	stepBegin(cs, addReplaySampleSteps, maxReplayChainSize);

	activate(_input, cs, reward, alpha, gamma, columnDecay, cellStateDecay, columnConnectionAlpha, columnConnectionBeta, columnConnectionGamma, cellConnectionAlpha, cellConnectionBeta, cellConnectionGamma, cellConnectionTemperature, cellWeightEligibilityDecay, maxReplayChainSize, numReplaySamples, addReplaySampleSteps, seed);

	std::vector<float> output;

	getReconstructedPrediction(output, cs);

	// Exploratory action
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, perturbationStdDev);

	for (int i = 0; i < _input.size(); i++)
	if (_inputTypes[i] == _action) {
		if (dist01(generator) < breakChance)
			_input[i] = dist01(generator);
		else
			_input[i] = std::min<float>(1.0f, std::max<float>(0.0f, std::min<float>(1.0f, std::max<float>(0.0f, output[i])) + pertDist(generator)));
	}
	else if (_inputTypes[i] == _unused)
		_input[i] = 0.0f;
}

void HTMRL::exportCellData(sys::ComputeSystem &cs, std::vector<std::shared_ptr<sf::Image>> &images, unsigned long seed) const {
	std::mt19937 generator(seed);
	
	int maxWidth = _inputWidth;
	int maxHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		maxWidth = std::max<int>(maxWidth, _layerDescs[l]._width);
		maxHeight = std::max<int>(maxHeight, _layerDescs[l]._height);
	}
	
	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	{
		std::vector<float> state(_inputWidth * _inputHeight);

		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueReadImage(_layers.front()._reconstruction, CL_TRUE, origin, region, 0, 0, &state[0]);

		sf::Color c;
		c.r = uniformDist(generator) * 255.0f;
		c.g = uniformDist(generator) * 255.0f;
		c.b = uniformDist(generator) * 255.0f;

		// Convert to colors
		std::shared_ptr<sf::Image> image = std::make_shared<sf::Image>();

		image->create(maxWidth, maxHeight, sf::Color::Transparent);

		for (int x = 0; x < _inputWidth; x++)
		for (int y = 0; y < _inputHeight; y++) {
			sf::Color color;

			color = c;

			color.a = std::min<float>(1.0f, std::max<float>(0.0f, state[x + y * _inputWidth])) * (255.0f - 3.0f) + 3;

			image->setPixel(x - _inputWidth / 2 + maxWidth / 2, y - _inputHeight / 2 + maxHeight / 2, color);
		}

		images.push_back(image);
	}

	/*{
		sf::Color c;
		c.r = uniformDist(generator) * 255.0f;
		c.g = uniformDist(generator) * 255.0f;
		c.b = uniformDist(generator) * 255.0f;

		// Convert to colors
		std::shared_ptr<sf::Image> image = std::make_shared<sf::Image>();

		image->create(maxWidth, maxHeight, sf::Color::Transparent);

		for (int x = 0; x < _inputWidth; x++)
		for (int y = 0; y < _inputHeight; y++) {
			sf::Color color;

			color = c;

			color.a = std::min<float>(1.0f, std::max<float>(0.0f, _exploratoryOutput[x + y * _inputWidth])) * (255.0f - 3.0f) + 3;

			image->setPixel(x - _inputWidth / 2 + maxWidth / 2, y - _inputHeight / 2 + maxHeight / 2, color);
		}

		images.push_back(image);
	}*/

	if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
		for (int l = 0; l < _layers.size(); l++) {
			std::vector<float> state(_layerDescs[l]._width * _layerDescs[l]._height * _layerDescs[l]._cellsInColumn * 2);

			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = _layerDescs[l]._cellsInColumn;

			cs.getQueue().enqueueReadImage(_layers[l]._cellPredictions, CL_TRUE, origin, region, 0, 0, &state[0]);

			sf::Color c;
			c.r = uniformDist(generator) * 255.0f;
			c.g = uniformDist(generator) * 255.0f;
			c.b = uniformDist(generator) * 255.0f;

			// Convert to colors
			for (int ci = 0; ci < _layerDescs[l]._cellsInColumn; ci++) {
				std::shared_ptr<sf::Image> image = std::make_shared<sf::Image>();

				image->create(maxWidth, maxHeight, sf::Color::Transparent);

				for (int x = 0; x < _layerDescs[l]._width; x++)
				for (int y = 0; y < _layerDescs[l]._height; y++) {
					sf::Color color;

					color = c;

					color.a = std::min<float>(1.0f, std::max<float>(0.0f, state[2 * (x + y * _layerDescs[l]._width + ci * _layerDescs[l]._width *_layerDescs[l]._height)])) * (255.0f - 3.0f) + 3;

					int wx = x - _layerDescs[l]._width / 2 + maxWidth / 2;
					int wy = y - _layerDescs[l]._height / 2 + maxHeight / 2;

					assert(wx >= 0 && wy >= 0 && wx < maxWidth && wy < maxHeight);

					image->setPixel(wx, wy, color);
				}

				images.push_back(image);
			}
		}
	}
	else {
		for (int l = 0; l < _layers.size(); l++) {
			std::vector<float> state(_layerDescs[l]._width * _layerDescs[l]._height * _layerDescs[l]._cellsInColumn * 2);

			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = _layerDescs[l]._cellsInColumn;

			cs.getQueue().enqueueReadImage(_layers[l]._cellStates, CL_TRUE, origin, region, 0, 0, &state[0]);

			sf::Color c;
			c.r = uniformDist(generator) * 255.0f;
			c.g = uniformDist(generator) * 255.0f;
			c.b = uniformDist(generator) * 255.0f;

			// Convert to colors
			for (int ci = 0; ci < _layerDescs[l]._cellsInColumn; ci++) {
				std::shared_ptr<sf::Image> image = std::make_shared<sf::Image>();

				image->create(maxWidth, maxHeight, sf::Color::Transparent);

				for (int x = 0; x < _layerDescs[l]._width; x++)
				for (int y = 0; y < _layerDescs[l]._height; y++) {
					sf::Color color;

					color = c;

					color.a = std::min<float>(1.0f, std::max<float>(0.0f, std::max<float>(0.0f, state[0 + 2 * (x + y * _layerDescs[l]._width + ci * _layerDescs[l]._width *_layerDescs[l]._height)]))) * (255.0f - 3.0f) + 3;

					//color.g = std::min<float>(1.0f, std::max<float>(0.0f, std::max<float>(0.0f, state[2 + 4 * (x + y * _layerDescs[l]._width + ci * _layerDescs[l]._width *_layerDescs[l]._height)]))) * (255.0f - 3.0f) + 3;

					//color.b = 0;
					//color.a = 0.5f * (color.r + color.g);

					int wx = x - _layerDescs[l]._width / 2 + maxWidth / 2;
					int wy = y - _layerDescs[l]._height / 2 + maxHeight / 2;

					assert(wx >= 0 && wy >= 0 && wx < maxWidth && wy < maxHeight);

					image->setPixel(wx, wy, color);
				}

				images.push_back(image);
			}
		}

		/*for (int l = 0; l < _layers.size(); l++) {
			std::vector<float> state(_layerDescs[l]._width * _layerDescs[l]._height * 2);

			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = 1;

			cs.getQueue().enqueueReadImage(_layers[l]._blurPong, CL_TRUE, origin, region, 0, 0, &state[0]);

			sf::Color c;
			c.r = uniformDist(generator) * 255.0f;
			c.g = uniformDist(generator) * 255.0f;
			c.b = uniformDist(generator) * 255.0f;

			// Convert to colors
			std::shared_ptr<sf::Image> image = std::make_shared<sf::Image>();

			image->create(maxWidth, maxHeight, sf::Color::Transparent);

			for (int x = 0; x < _layerDescs[l]._width; x++)
			for (int y = 0; y < _layerDescs[l]._height; y++) {
				sf::Color color;

				color = c;

				color.a = std::min<float>(1.0f, std::max<float>(0.0f, state[0 + 2 * (x + y * _layerDescs[l]._width)] > 0.0f ? 1.0f : 0.0f)) * (255.0f - 3.0f) + 3;

				int wx = x - _layerDescs[l]._width / 2 + maxWidth / 2;
				int wy = y - _layerDescs[l]._height / 2 + maxHeight / 2;

				assert(wx >= 0 && wy >= 0 && wx < maxWidth && wy < maxHeight);

				image->setPixel(wx, wy, color);
			}

			images.push_back(image);
		}*/
	}
}