#include <htm/HTMRL.h>

#include <iostream>

using namespace htm;

void HTMRL::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, float minInitWidth, float maxInitWidth, float minEncoderInitCenter, float maxEncoderInitCenter, float minEncoderInitWidth, float maxEncoderInitWidth, float minEncoderInitWeight, float maxEncoderInitWeight, std::mt19937 &generator) {
	struct Uint2 {
		unsigned int _x, _y;
	};
	
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_layerDescs = layerDescs;
	
	_layers.resize(_layerDescs.size());

	_inputTypes = inputTypes;

	std::uniform_real_distribution<float> weightDist(minInitWeight, maxInitWeight);
	std::uniform_real_distribution<float> actionDist(-1.0f, 1.0f);

	_outputBias = weightDist(generator);
	_prevMaxQ = 0.0f;
	_prevValue = 0.0f;
	_prevPrevValue = 0.0f;
	_prevQ = 0.0f;
	_prevTDError = 0.0f;

	cl::Kernel initPartOneKernel = cl::Kernel(program.getProgram(), "initializePartOne");
	cl::Kernel initPartTwoKernel = cl::Kernel(program.getProgram(), "initializePartTwo");
	cl::Kernel initPartThreeKernel = cl::Kernel(program.getProgram(), "initializePartThree");

	_input.clear();
	_input.assign(_inputWidth * _inputHeight, 0.0f);

	_output.clear();
	_output.assign(_inputWidth * _inputHeight, 0.0f);

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

	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;
	int prevCellsPerColumn = 1;

	for (int l = 0; l < _layers.size(); l++) {
		initLayer(cs, initPartOneKernel, initPartTwoKernel, initPartThreeKernel, prevWidth, prevHeight, prevCellsPerColumn, _layers[l], _layerDescs[l], l == _layers.size() - 1, minInitWeight, maxInitWeight, minInitWidth, maxInitWidth, generator);

		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
		prevCellsPerColumn = _layerDescs[l]._cellsInColumn;
	}

	_outputWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs.back()._width, _layerDescs.back()._height, _layerDescs.back()._cellsInColumn);
	_outputWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs.back()._width, _layerDescs.back()._height, _layerDescs.back()._cellsInColumn);

	std::uniform_int_distribution<int> uniformDist(0, 10000);

	Uint2 seed1;
	seed1._x = uniformDist(generator);
	seed1._y = uniformDist(generator);

	cl::Kernel initPartFourKernel = cl::Kernel(program.getProgram(), "initializePartFour");

	initPartFourKernel.setArg(0, _outputWeights);
	initPartFourKernel.setArg(1, _layerDescs.back()._cellsInColumn);
	initPartFourKernel.setArg(2, seed1);
	initPartFourKernel.setArg(3, minInitWeight);
	initPartFourKernel.setArg(4, maxInitWeight);

	cs.getQueue().enqueueNDRangeKernel(initPartFourKernel, cl::NullRange, cl::NDRange(_layerDescs.back()._width, _layerDescs.back()._height));

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = _layerDescs.back()._width;
		region[1] = _layerDescs.back()._height;
		region[2] = _layerDescs.back()._cellsInColumn;

		cs.getQueue().enqueueCopyImage(_outputWeights, _outputWeightsPrev, origin, origin, region);
	}

	_partialSums = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs.back()._width, _layerDescs.back()._height);

	_layerColumnActivateKernel = cl::Kernel(program.getProgram(), "layerColumnActivate");
	_layerColumnInhibitKernel = cl::Kernel(program.getProgram(), "layerColumnInhibit");
	_layerColumnDutyCycleUpdateKernel = cl::Kernel(program.getProgram(), "layerColumnDutyCycleUpdate");
	_layerCellActivateKernel = cl::Kernel(program.getProgram(), "layerCellActivate");
	_layerCellWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerCellWeightUpdate");
	_layerCellWeightUpdateLastKernel = cl::Kernel(program.getProgram(), "layerCellWeightUpdateLast");
	_layerCellPredictKernel = cl::Kernel(program.getProgram(), "layerCellPredict");
	_layerCellPredictLastKernel = cl::Kernel(program.getProgram(), "layerCellPredictLast");
	_layerColumnWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerColumnWeightUpdate");
	_layerColumnPredictionKernel = cl::Kernel(program.getProgram(), "layerColumnPrediction");

	_layerNodeActivateKernel = cl::Kernel(program.getProgram(), "layerNodeActivate");
	_layerNodeActivateFirstKernel = cl::Kernel(program.getProgram(), "layerNodeActivateFirst");
	_weighOutputKernel = cl::Kernel(program.getProgram(), "weighOutput");
}

void HTMRL::initLayer(sys::ComputeSystem &cs, cl::Kernel &initPartOneKernel, cl::Kernel &initPartTwoKernel, cl::Kernel &initPartThreeKernel, int inputWidth, int inputHeight, int inputCellsPerColumn, Layer &layer, const LayerDesc &layerDesc, bool isTopmost, float minInitWeight, float maxInitWeight, float minInitWidth, float maxInitWidth, std::mt19937 &generator) {
	struct Uint2 {
		unsigned int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};
	
	std::uniform_int_distribution<int> uniformDist(0, 10000);

	int receptiveFieldSize = std::pow(layerDesc._receptiveFieldRadius * 2 + 1, 2);
	int nodeConnectionsSize = std::pow(layerDesc._nodeFieldRadius * 2 + 1, 2) * inputCellsPerColumn;
	int lateralConnectionsSize;

	// If not the last layer, add weights for additional context from next layer
	if (isTopmost)
		lateralConnectionsSize = std::pow(layerDesc._lateralConnectionRadius * 2 + 1, 2) * (layerDesc._cellsInColumn) + 1; // + 1 for bias
	else
		lateralConnectionsSize = std::pow(layerDesc._lateralConnectionRadius * 2 + 1, 2) * (layerDesc._cellsInColumn + 1) + 1; // + 1 for bias

	layer._columnActivations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height);
	
	layer._columnStates = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._columnWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, receptiveFieldSize + 1); // + 1 for width
	layer._columnWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, receptiveFieldSize + 1); // + 1 for width

	layer._columnDutyCycles = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnDutyCyclesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._cellStates = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellStatesPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._cellPredictions = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellPredictionsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._cellWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, lateralConnectionsSize);
	layer._cellWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, lateralConnectionsSize);

	layer._columnPredictions = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnPredictionsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._nodeOutputs = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._nodeErrors = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._nodeBiases = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._nodeBiasesPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._nodeWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, nodeConnectionsSize);
	layer._nodeWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, nodeConnectionsSize);

	Uint2 seed1;
	seed1._x = uniformDist(generator);
	seed1._y = uniformDist(generator);

	initPartOneKernel.setArg(0, layer._columnActivations);
	initPartOneKernel.setArg(1, layer._columnStates);
	initPartOneKernel.setArg(2, layer._columnWeights);
	initPartOneKernel.setArg(3, layer._columnDutyCycles);
	initPartOneKernel.setArg(4, layerDesc._cellsInColumn);
	initPartOneKernel.setArg(5, receptiveFieldSize);
	initPartOneKernel.setArg(6, lateralConnectionsSize);
	initPartOneKernel.setArg(7, seed1);
	initPartOneKernel.setArg(8, minInitWeight);
	initPartOneKernel.setArg(9, maxInitWeight);
	initPartOneKernel.setArg(10, minInitWidth);
	initPartOneKernel.setArg(11, maxInitWidth);

	cs.getQueue().enqueueNDRangeKernel(initPartOneKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	Uint2 seed2;
	seed2._x = uniformDist(generator);
	seed2._y = uniformDist(generator);

	initPartTwoKernel.setArg(0, layer._cellStates);
	initPartTwoKernel.setArg(1, layer._cellWeights);
	initPartTwoKernel.setArg(2, layer._cellPredictions);
	initPartTwoKernel.setArg(3, layerDesc._cellsInColumn);
	initPartTwoKernel.setArg(4, receptiveFieldSize);
	initPartTwoKernel.setArg(5, lateralConnectionsSize);
	initPartTwoKernel.setArg(6, seed2);
	initPartTwoKernel.setArg(7, minInitWeight);
	initPartTwoKernel.setArg(8, maxInitWeight);

	cs.getQueue().enqueueNDRangeKernel(initPartTwoKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	Uint2 seed3;
	seed3._x = uniformDist(generator);
	seed3._y = uniformDist(generator);

	initPartThreeKernel.setArg(0, layer._nodeOutputs);
	initPartThreeKernel.setArg(1, layer._nodeErrors);
	initPartThreeKernel.setArg(2, layer._nodeBiases);
	initPartThreeKernel.setArg(3, layer._nodeWeights);
	initPartThreeKernel.setArg(4, layerDesc._cellsInColumn);
	initPartThreeKernel.setArg(5, receptiveFieldSize);
	initPartThreeKernel.setArg(6, lateralConnectionsSize);
	initPartThreeKernel.setArg(7, seed3);
	initPartThreeKernel.setArg(8, minInitWeight);
	initPartThreeKernel.setArg(9, maxInitWeight);

	cs.getQueue().enqueueNDRangeKernel(initPartThreeKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

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

		cs.getQueue().enqueueCopyImage(layer._columnDutyCycles, layer._columnDutyCyclesPrev, origin, origin, region);
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

		cs.getQueue().enqueueCopyImage(layer._columnWeights, layer._columnWeightsPrev, origin, origin, region);
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

		region[0] = layerDesc._width;
		region[1] = layerDesc._height;
		region[2] = layerDesc._cellsInColumn;

		cs.getQueue().enqueueCopyImage(layer._nodeBiases, layer._nodeBiasesPrev, origin, origin, region);
	}

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = layerDesc._width;
		region[1] = layerDesc._height * layerDesc._cellsInColumn;
		region[2] = nodeConnectionsSize;

		cs.getQueue().enqueueCopyImage(layer._nodeWeights, layer._nodeWeightsPrev, origin, origin, region);
	}
}

void HTMRL::stepBegin() {
	for (int l = 0; l < _layers.size(); l++) {
		std::swap(_layers[l]._columnStates, _layers[l]._columnStatesPrev);
		std::swap(_layers[l]._columnPredictions, _layers[l]._columnPredictionsPrev);
		std::swap(_layers[l]._columnDutyCycles, _layers[l]._columnDutyCyclesPrev);
		std::swap(_layers[l]._columnWeights, _layers[l]._columnWeightsPrev);
		std::swap(_layers[l]._cellStates, _layers[l]._cellStatesPrev);
		std::swap(_layers[l]._cellPredictions, _layers[l]._cellPredictionsPrev);
		std::swap(_layers[l]._cellWeights, _layers[l]._cellWeightsPrev);
		std::swap(_layers[l]._nodeBiases, _layers[l]._nodeBiasesPrev);
		std::swap(_layers[l]._nodeWeights, _layers[l]._nodeWeightsPrev);
	}

	std::swap(_outputWeights, _outputWeightsPrev);
}

void HTMRL::activateLayer(sys::ComputeSystem &cs, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator) {
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

	Int2 inputReceptiveFieldRadius;
	inputReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	inputReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Float2 inputReceptiveFieldStep;
	inputReceptiveFieldStep._x = inputSizeInv._x;
	inputReceptiveFieldStep._y = inputSizeInv._y;

	// Activation
	_layerColumnActivateKernel.setArg(0, prevLayerOutput);
	_layerColumnActivateKernel.setArg(1, layer._columnWeightsPrev);
	_layerColumnActivateKernel.setArg(2, layer._columnActivations);
	_layerColumnActivateKernel.setArg(3, layerSizeInv);
	_layerColumnActivateKernel.setArg(4, inputReceptiveFieldRadius);
	_layerColumnActivateKernel.setArg(5, inputReceptiveFieldStep);
	_layerColumnActivateKernel.setArg(6, inputSize);
	_layerColumnActivateKernel.setArg(7, seed);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnActivateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	Int2 layerInhibitionRadius;
	layerInhibitionRadius._x = layerDesc._inhibitionRadius;
	layerInhibitionRadius._y = layerDesc._inhibitionRadius;

	Float2 layerInhibitionStep;
	layerInhibitionStep._x = layerSizeInv._x;
	layerInhibitionStep._y = layerSizeInv._y;

	// Inhibition
	_layerColumnInhibitKernel.setArg(0, layer._columnActivations);
	_layerColumnInhibitKernel.setArg(1, layer._columnDutyCyclesPrev);
	_layerColumnInhibitKernel.setArg(2, layer._columnStates);
	_layerColumnInhibitKernel.setArg(3, layerSize);
	_layerColumnInhibitKernel.setArg(4, layerSizeInv);
	_layerColumnInhibitKernel.setArg(5, layerInhibitionRadius);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnInhibitKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	int layerWidth = layerDesc._width;

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

	cs.getQueue().enqueueNDRangeKernel(_layerCellActivateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::predictLayer(sys::ComputeSystem &cs, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator) {
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
	Int2 nextLayerSize;
	nextLayerSize._x = nextLayerWidth;
	nextLayerSize._y = nextLayerHeight;

	_layerCellPredictKernel.setArg(0, layer._columnStates);
	_layerCellPredictKernel.setArg(1, layer._cellStates);
	_layerCellPredictKernel.setArg(2, layer._cellWeightsPrev);
	_layerCellPredictKernel.setArg(3, nextLayerPrediction);
	_layerCellPredictKernel.setArg(4, layer._cellPredictions);
	_layerCellPredictKernel.setArg(5, layerDesc._cellsInColumn);
	_layerCellPredictKernel.setArg(6, layerSize);
	_layerCellPredictKernel.setArg(7, lateralConnectionRadii);
	_layerCellPredictKernel.setArg(8, layerSizeInv);
	_layerCellPredictKernel.setArg(9, nextLayerSize);

	cs.getQueue().enqueueNDRangeKernel(_layerCellPredictKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	// Column prediction
	_layerColumnPredictionKernel.setArg(0, layer._cellPredictions);
	_layerColumnPredictionKernel.setArg(1, layer._cellStates);
	_layerColumnPredictionKernel.setArg(2, layer._columnPredictions);
	_layerColumnPredictionKernel.setArg(3, layerDesc._cellsInColumn);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnPredictionKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
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
	_layerCellPredictLastKernel.setArg(0, layer._columnStates);
	_layerCellPredictLastKernel.setArg(1, layer._cellStates);
	_layerCellPredictLastKernel.setArg(2, layer._cellWeightsPrev);
	_layerCellPredictLastKernel.setArg(3, layer._cellPredictions);
	_layerCellPredictLastKernel.setArg(4, layerDesc._cellsInColumn);
	_layerCellPredictLastKernel.setArg(5, layerSize);
	_layerCellPredictLastKernel.setArg(6, lateralConnectionRadii);

	cs.getQueue().enqueueNDRangeKernel(_layerCellPredictLastKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	// Column prediction
	_layerColumnPredictionKernel.setArg(0, layer._cellPredictions);
	_layerColumnPredictionKernel.setArg(1, layer._cellStates);
	_layerColumnPredictionKernel.setArg(2, layer._columnPredictions);
	_layerColumnPredictionKernel.setArg(3, layerDesc._cellsInColumn);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnPredictionKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::activate(std::vector<float> &input, sys::ComputeSystem &cs, unsigned long seed) {
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

	cl::Image2D* pPrevLayerOutput = &_inputImage;
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		activateLayer(cs, *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l], _layerDescs[l], generator);

		pPrevLayerOutput = &_layers[l]._columnStates;
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}

	for (int l = _layers.size() - 1; l >= 0; l--) {
		if (l == _layers.size() - 1)
			predictLayerLast(cs, _layers[l], _layerDescs[l], generator);
		else
			predictLayer(cs, _layers[l + 1]._columnPredictions, _layerDescs[l + 1]._width, _layerDescs[l + 1]._height, _layers[l], _layerDescs[l], generator);
	}
}

void HTMRL::learnLayerSpatial(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, std::mt19937 &generator) {
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

	Int2 inputReceptiveFieldRadius;
	inputReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	inputReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Int2 layerReceptiveFieldRadius;
	layerReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	layerReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Float2 layerReceptiveFieldStep;
	layerReceptiveFieldStep._x = layerSizeInv._x;
	layerReceptiveFieldStep._y = layerSizeInv._y;

	// Column weight update
	_layerColumnWeightUpdateKernel.setArg(0, prevLayerOutput);
	_layerColumnWeightUpdateKernel.setArg(1, layer._columnActivations);
	_layerColumnWeightUpdateKernel.setArg(2, layer._columnStates);
	_layerColumnWeightUpdateKernel.setArg(3, layer._columnWeightsPrev);
	_layerColumnWeightUpdateKernel.setArg(4, layer._columnDutyCyclesPrev);
	_layerColumnWeightUpdateKernel.setArg(5, layer._columnWeights);
	_layerColumnWeightUpdateKernel.setArg(6, layerSize);
	_layerColumnWeightUpdateKernel.setArg(7, layerSizeInv);
	_layerColumnWeightUpdateKernel.setArg(8, inputReceptiveFieldRadius);
	_layerColumnWeightUpdateKernel.setArg(9, inputSize);
	_layerColumnWeightUpdateKernel.setArg(10, inputSizeInv);
	_layerColumnWeightUpdateKernel.setArg(11, columnConnectionAlpha);
	_layerColumnWeightUpdateKernel.setArg(12, widthAlpha);
	_layerColumnWeightUpdateKernel.setArg(13, seed);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::learnLayerSpatialLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, std::mt19937 &generator) {
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

	Int2 inputReceptiveFieldRadius;
	inputReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	inputReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Int2 layerReceptiveFieldRadius;
	layerReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	layerReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Float2 layerReceptiveFieldStep;
	layerReceptiveFieldStep._x = layerSizeInv._x;
	layerReceptiveFieldStep._y = layerSizeInv._y;

	// Column weight update
	_layerColumnWeightUpdateKernel.setArg(0, prevLayerOutput);
	_layerColumnWeightUpdateKernel.setArg(1, layer._columnActivations);
	_layerColumnWeightUpdateKernel.setArg(2, layer._columnStates);
	_layerColumnWeightUpdateKernel.setArg(3, layer._columnWeightsPrev);
	_layerColumnWeightUpdateKernel.setArg(4, layer._columnDutyCyclesPrev);
	_layerColumnWeightUpdateKernel.setArg(5, layer._columnWeights);
	_layerColumnWeightUpdateKernel.setArg(6, layerSize);
	_layerColumnWeightUpdateKernel.setArg(7, layerSizeInv);
	_layerColumnWeightUpdateKernel.setArg(8, inputReceptiveFieldRadius);
	_layerColumnWeightUpdateKernel.setArg(9, inputSize);
	_layerColumnWeightUpdateKernel.setArg(10, inputSizeInv);
	_layerColumnWeightUpdateKernel.setArg(11, columnConnectionAlpha);
	_layerColumnWeightUpdateKernel.setArg(12, widthAlpha);
	_layerColumnWeightUpdateKernel.setArg(13, seed);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::learnLayerTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float cellConnectionAlpha, float cellWeightEligibilityDecay, std::mt19937 &generator) {
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

	Int2 inputReceptiveFieldRadius;
	inputReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	inputReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Int2 layerReceptiveFieldRadius;
	layerReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	layerReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Float2 layerReceptiveFieldStep;
	layerReceptiveFieldStep._x = layerSizeInv._x;
	layerReceptiveFieldStep._y = layerSizeInv._y;

	Int2 lateralConnectionRadii;
	lateralConnectionRadii._x = layerDesc._lateralConnectionRadius;
	lateralConnectionRadii._y = layerDesc._lateralConnectionRadius;

	// Lateral weight update
	Int2 nextLayerSize;
	nextLayerSize._x = nextLayerWidth;
	nextLayerSize._y = nextLayerHeight;

	_layerCellWeightUpdateKernel.setArg(0, layer._columnStates);
	_layerCellWeightUpdateKernel.setArg(1, layer._cellStatesPrev);
	_layerCellWeightUpdateKernel.setArg(2, nextLayerPrediction);
	_layerCellWeightUpdateKernel.setArg(3, layer._cellWeightsPrev);
	_layerCellWeightUpdateKernel.setArg(4, layer._columnPredictionsPrev);
	_layerCellWeightUpdateKernel.setArg(5, layer._cellWeights);
	_layerCellWeightUpdateKernel.setArg(6, layerDesc._cellsInColumn);
	_layerCellWeightUpdateKernel.setArg(7, layerSize);
	_layerCellWeightUpdateKernel.setArg(8, lateralConnectionRadii);
	_layerCellWeightUpdateKernel.setArg(9, layerSizeInv);
	_layerCellWeightUpdateKernel.setArg(10, nextLayerSize);
	_layerCellWeightUpdateKernel.setArg(11, cellConnectionAlpha);
	_layerCellWeightUpdateKernel.setArg(12, cellWeightEligibilityDecay);

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::learnLayerTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float cellConnectionAlpha, float cellWeightEligibilityDecay, std::mt19937 &generator) {
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

	Int2 inputReceptiveFieldRadius;
	inputReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	inputReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Int2 layerReceptiveFieldRadius;
	layerReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	layerReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Float2 layerReceptiveFieldStep;
	layerReceptiveFieldStep._x = layerSizeInv._x;
	layerReceptiveFieldStep._y = layerSizeInv._y;

	Int2 lateralConnectionRadii;
	lateralConnectionRadii._x = layerDesc._lateralConnectionRadius;
	lateralConnectionRadii._y = layerDesc._lateralConnectionRadius;

	// Lateral weight update
	_layerCellWeightUpdateLastKernel.setArg(0, layer._columnStates);
	_layerCellWeightUpdateLastKernel.setArg(1, layer._cellStatesPrev);
	_layerCellWeightUpdateLastKernel.setArg(2, layer._cellWeightsPrev);
	_layerCellWeightUpdateLastKernel.setArg(3, layer._columnPredictionsPrev);
	_layerCellWeightUpdateLastKernel.setArg(4, layer._cellWeights);
	_layerCellWeightUpdateLastKernel.setArg(5, layerDesc._cellsInColumn);
	_layerCellWeightUpdateLastKernel.setArg(6, layerSize);
	_layerCellWeightUpdateLastKernel.setArg(7, lateralConnectionRadii);
	_layerCellWeightUpdateLastKernel.setArg(8, cellConnectionAlpha);
	_layerCellWeightUpdateLastKernel.setArg(9, cellWeightEligibilityDecay);

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateLastKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::learnLayerSpatialTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellWeightEligibilityDecay, std::mt19937 &generator) {
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

	Int2 inputReceptiveFieldRadius;
	inputReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	inputReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Int2 layerReceptiveFieldRadius;
	layerReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	layerReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Float2 layerReceptiveFieldStep;
	layerReceptiveFieldStep._x = layerSizeInv._x;
	layerReceptiveFieldStep._y = layerSizeInv._y;

	// Column weight update
	_layerColumnWeightUpdateKernel.setArg(0, prevLayerOutput);
	_layerColumnWeightUpdateKernel.setArg(1, layer._columnActivations);
	_layerColumnWeightUpdateKernel.setArg(2, layer._columnStates);
	_layerColumnWeightUpdateKernel.setArg(3, layer._columnWeightsPrev);
	_layerColumnWeightUpdateKernel.setArg(4, layer._columnDutyCyclesPrev);
	_layerColumnWeightUpdateKernel.setArg(5, layer._columnWeights);
	_layerColumnWeightUpdateKernel.setArg(6, layerSize);
	_layerColumnWeightUpdateKernel.setArg(7, layerSizeInv);
	_layerColumnWeightUpdateKernel.setArg(8, inputReceptiveFieldRadius);
	_layerColumnWeightUpdateKernel.setArg(9, inputSize);
	_layerColumnWeightUpdateKernel.setArg(10, inputSizeInv);
	_layerColumnWeightUpdateKernel.setArg(11, columnConnectionAlpha);
	_layerColumnWeightUpdateKernel.setArg(12, widthAlpha);
	_layerColumnWeightUpdateKernel.setArg(13, seed);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	Int2 lateralConnectionRadii;
	lateralConnectionRadii._x = layerDesc._lateralConnectionRadius;
	lateralConnectionRadii._y = layerDesc._lateralConnectionRadius;

	// Lateral weight update
	Int2 nextLayerSize;
	nextLayerSize._x = nextLayerWidth;
	nextLayerSize._y = nextLayerHeight;

	_layerCellWeightUpdateKernel.setArg(0, layer._columnStates);
	_layerCellWeightUpdateKernel.setArg(1, layer._cellStatesPrev);
	_layerCellWeightUpdateKernel.setArg(2, nextLayerPrediction);
	_layerCellWeightUpdateKernel.setArg(3, layer._cellWeightsPrev);
	_layerCellWeightUpdateKernel.setArg(4, layer._columnPredictionsPrev);
	_layerCellWeightUpdateKernel.setArg(5, layer._cellWeights);
	_layerCellWeightUpdateKernel.setArg(6, layerDesc._cellsInColumn);
	_layerCellWeightUpdateKernel.setArg(7, layerSize);
	_layerCellWeightUpdateKernel.setArg(8, lateralConnectionRadii);
	_layerCellWeightUpdateKernel.setArg(9, layerSizeInv);
	_layerCellWeightUpdateKernel.setArg(10, nextLayerSize);
	_layerCellWeightUpdateKernel.setArg(11, cellConnectionAlpha);
	_layerCellWeightUpdateKernel.setArg(12, cellWeightEligibilityDecay);

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::learnLayerSpatialTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellWeightEligibilityDecay, std::mt19937 &generator) {
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

	Int2 inputReceptiveFieldRadius;
	inputReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	inputReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Int2 layerReceptiveFieldRadius;
	layerReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	layerReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Float2 layerReceptiveFieldStep;
	layerReceptiveFieldStep._x = layerSizeInv._x;
	layerReceptiveFieldStep._y = layerSizeInv._y;

	// Column weight update
	_layerColumnWeightUpdateKernel.setArg(0, prevLayerOutput);
	_layerColumnWeightUpdateKernel.setArg(1, layer._columnActivations);
	_layerColumnWeightUpdateKernel.setArg(2, layer._columnStates);
	_layerColumnWeightUpdateKernel.setArg(3, layer._columnWeightsPrev);
	_layerColumnWeightUpdateKernel.setArg(4, layer._columnDutyCyclesPrev);
	_layerColumnWeightUpdateKernel.setArg(5, layer._columnWeights);
	_layerColumnWeightUpdateKernel.setArg(6, layerSize);
	_layerColumnWeightUpdateKernel.setArg(7, layerSizeInv);
	_layerColumnWeightUpdateKernel.setArg(8, inputReceptiveFieldRadius);
	_layerColumnWeightUpdateKernel.setArg(9, inputSize);
	_layerColumnWeightUpdateKernel.setArg(10, inputSizeInv);
	_layerColumnWeightUpdateKernel.setArg(11, columnConnectionAlpha);
	_layerColumnWeightUpdateKernel.setArg(12, widthAlpha);
	_layerColumnWeightUpdateKernel.setArg(13, seed);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	Int2 lateralConnectionRadii;
	lateralConnectionRadii._x = layerDesc._lateralConnectionRadius;
	lateralConnectionRadii._y = layerDesc._lateralConnectionRadius;

	// Lateral weight update
	_layerCellWeightUpdateLastKernel.setArg(0, layer._columnStates);
	_layerCellWeightUpdateLastKernel.setArg(1, layer._cellStatesPrev);
	_layerCellWeightUpdateLastKernel.setArg(2, layer._cellWeightsPrev);
	_layerCellWeightUpdateLastKernel.setArg(3, layer._columnPredictionsPrev);
	_layerCellWeightUpdateLastKernel.setArg(4, layer._cellWeights);
	_layerCellWeightUpdateLastKernel.setArg(5, layerDesc._cellsInColumn);
	_layerCellWeightUpdateLastKernel.setArg(6, layerSize);
	_layerCellWeightUpdateLastKernel.setArg(7, lateralConnectionRadii);
	_layerCellWeightUpdateLastKernel.setArg(8, cellConnectionAlpha);
	_layerCellWeightUpdateLastKernel.setArg(9, cellWeightEligibilityDecay);

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateLastKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::learnSpatial(sys::ComputeSystem &cs, float columnConnectionAlpha, float widthAlpha, unsigned long seed) {
	std::mt19937 generator(seed);

	cl::Image2D* pPrevLayerOutput = &_inputImage;
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		if (l == _layers.size() - 1)
			learnLayerSpatialLast(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], columnConnectionAlpha, widthAlpha, generator);
		else
			learnLayerSpatial(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l + 1]._columnPredictions, _layerDescs[l + 1]._width, _layerDescs[l + 1]._width, _layerDescs[l], columnConnectionAlpha, widthAlpha, generator);

		pPrevLayerOutput = &_layers[l]._columnStates;
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}
}

void HTMRL::learnTemporal(sys::ComputeSystem &cs, float cellConnectionAlpha, float cellWeightEligibilityDecay, unsigned long seed) {
	std::mt19937 generator(seed);

	cl::Image2D* pPrevLayerOutput = &_inputImage;
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		if (l == _layers.size() - 1)
			learnLayerTemporalLast(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], cellConnectionAlpha, cellWeightEligibilityDecay, generator);
		else
			learnLayerTemporal(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l + 1]._columnPredictions, _layerDescs[l + 1]._width, _layerDescs[l + 1]._width, _layerDescs[l], cellConnectionAlpha, cellWeightEligibilityDecay, generator);

		pPrevLayerOutput = &_layers[l]._columnStates;
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}
}

void HTMRL::learnSpatialTemporal(sys::ComputeSystem &cs, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellWeightEligibilityDecay, unsigned long seed) {
	std::mt19937 generator(seed);

	cl::Image2D* pPrevLayerOutput = &_inputImage;
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;
	
	for (int l = 0; l < _layers.size(); l++) {
		if (l == _layers.size() - 1)
			learnLayerSpatialTemporalLast(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], columnConnectionAlpha, widthAlpha,  cellConnectionAlpha, cellWeightEligibilityDecay, generator);
		else
			learnLayerSpatialTemporal(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l + 1]._columnPredictions, _layerDescs[l + 1]._width, _layerDescs[l + 1]._width, _layerDescs[l], columnConnectionAlpha, widthAlpha, cellConnectionAlpha, cellWeightEligibilityDecay, generator);
	
		pPrevLayerOutput = &_layers[l]._columnStates;
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}
}

void HTMRL::dutyCycleLayerUpdate(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, float activationDutyCycleDecay, float stateDutyCycleDecay) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Int2 layerSize;
	layerSize._x = layerDesc._width;
	layerSize._y = layerDesc._height;

	Int2 layerInhibitionRadius;
	layerInhibitionRadius._x = layerDesc._inhibitionRadius;
	layerInhibitionRadius._y = layerDesc._inhibitionRadius;

	// Duty cycles
	_layerColumnDutyCycleUpdateKernel.setArg(0, layer._columnActivations);
	_layerColumnDutyCycleUpdateKernel.setArg(1, layer._columnStates);
	_layerColumnDutyCycleUpdateKernel.setArg(2, layer._columnDutyCyclesPrev);
	_layerColumnDutyCycleUpdateKernel.setArg(3, layer._columnDutyCycles);
	_layerColumnDutyCycleUpdateKernel.setArg(4, layerSize);
	_layerColumnDutyCycleUpdateKernel.setArg(5, layerInhibitionRadius);
	_layerColumnDutyCycleUpdateKernel.setArg(6, activationDutyCycleDecay);
	_layerColumnDutyCycleUpdateKernel.setArg(7, stateDutyCycleDecay);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnDutyCycleUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::dutyCycleUpdate(sys::ComputeSystem &cs, float activationDutyCycleDecay, float stateDutyCycleDecay) {
	for (int l = 0; l < _layers.size(); l++)
		dutyCycleLayerUpdate(cs, _layers[l], _layerDescs[l], activationDutyCycleDecay, stateDutyCycleDecay);
}

void HTMRL::layerNodeActivate(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, const LayerDesc &inputDesc, cl::Image3D &inputImage) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Int2 nodeFieldRadius;
	nodeFieldRadius._x = nodeFieldRadius._y = layerDesc._nodeFieldRadius;

	Float2 layerSizeInv;
	layerSizeInv._x = 1.0f / layerDesc._width;
	layerSizeInv._y = 1.0f / layerDesc._height;

	Int2 inputSize;
	inputSize._x = inputDesc._width;
	inputSize._y = inputDesc._height;

	_layerNodeActivateKernel.setArg(0, inputImage);
	_layerNodeActivateKernel.setArg(1, layer._cellStates);
	_layerNodeActivateKernel.setArg(2, layer._nodeBiases);
	_layerNodeActivateKernel.setArg(3, layer._nodeWeights);
	_layerNodeActivateKernel.setArg(4, layer._nodeOutputs);
	_layerNodeActivateKernel.setArg(5, layerDesc._cellsInColumn);
	_layerNodeActivateKernel.setArg(6, inputDesc._cellsInColumn);
	_layerNodeActivateKernel.setArg(7, nodeFieldRadius);
	_layerNodeActivateKernel.setArg(8, layerSizeInv);
	_layerNodeActivateKernel.setArg(9, inputSize);

	cs.getQueue().enqueueNDRangeKernel(_layerNodeActivateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::layerNodeActivateFirst(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, cl::Image2D &inputImage) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Int2 nodeFieldRadius;
	nodeFieldRadius._x = nodeFieldRadius._y = layerDesc._nodeFieldRadius;

	Float2 layerSizeInv;
	layerSizeInv._x = 1.0f / layerDesc._width;
	layerSizeInv._y = 1.0f / layerDesc._height;

	Int2 inputSize;
	inputSize._x = _inputWidth;
	inputSize._y = _inputHeight;

	_layerNodeActivateFirstKernel.setArg(0, inputImage);
	_layerNodeActivateFirstKernel.setArg(1, layer._cellStates);
	_layerNodeActivateFirstKernel.setArg(2, layer._nodeBiases);
	_layerNodeActivateFirstKernel.setArg(3, layer._nodeWeights);
	_layerNodeActivateFirstKernel.setArg(4, layer._nodeOutputs);
	_layerNodeActivateFirstKernel.setArg(5, layerDesc._cellsInColumn);
	_layerNodeActivateFirstKernel.setArg(6, nodeFieldRadius);
	_layerNodeActivateFirstKernel.setArg(7, layerSizeInv);
	_layerNodeActivateFirstKernel.setArg(8, inputSize);

	cs.getQueue().enqueueNDRangeKernel(_layerNodeActivateFirstKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

float HTMRL::getQ(sys::ComputeSystem &cs) {
	_weighOutputKernel.setArg(0, _layers.back()._nodeOutputs);
	_weighOutputKernel.setArg(1, _outputWeights);
	_weighOutputKernel.setArg(2, _partialSums);
	_weighOutputKernel.setArg(3, _layerDescs.back()._cellsInColumn);

	cs.getQueue().enqueueNDRangeKernel(_weighOutputKernel, cl::NullRange, cl::NDRange(_layerDescs.back()._width, _layerDescs.back()._height));

	// Read partial sums and combine them
	float totalSum = _outputBias;

	std::vector<float> partialSums(_layerDescs.back()._width * _layerDescs.back()._height);

	cl::size_t<3> origin;
	cl::size_t<3> region;

	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;

	region[0] = _layerDescs.back()._width;
	region[1] = _layerDescs.back()._height;
	region[2] = 1;

	cs.getQueue().enqueueReadImage(_partialSums, CL_TRUE, origin, region, 0, 0, &partialSums[0]);

	for (int i = 0; i < partialSums.size(); i++)
		totalSum += partialSums[i];

	return totalSum;
}

void HTMRL::step(sys::ComputeSystem &cs, float reward, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellWeightEligibilityDecay, float cellQWeightEligibilityDecay, float activationDutyCycleDecay, float stateDutyCycleDecay, float reconstructionAlpha, float qBiasAlpha, float annealingStdDev, float annealingIterations, float annealingBreakChance, float annealingDecay, float annealingMomentum, float encoderLocalActivity, float encoderOutputIntensity, float encoderDutyCycleDecay, float encoderBoostThreshold, float encoderBoostIntensity, float encoderCenterAlpha, float encoderWidthAlpha, float encoderWidthScalar, float encoderMinWidth, float encoderReconAlpha, float learnIntensity, float alpha, float gamma, float tauInv, float breakChance, float perturbationStdDev, std::mt19937 &generator) {
	
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

	/*{
		std::vector<float> state(_layerDescs.front()._width * _layerDescs.front()._height);

		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _layerDescs.front()._width;
		region[1] = _layerDescs.front()._height;
		region[2] = 1;

		cs.getQueue().enqueueReadImage(_layers.front()._columnStates, CL_TRUE, origin, region, 0, 0, &state[0]);

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
	}*/

	{
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

			color.a = std::min<float>(1.0f, std::max<float>(0.0f, _output[x + y * _inputWidth])) * (255.0f - 3.0f) + 3;

			image->setPixel(x - _inputWidth / 2 + maxWidth / 2, y - _inputHeight / 2 + maxHeight / 2, color);
		}

		images.push_back(image);
	}
	
	/*for (int l = 0; l < _layers.size(); l++) {
		std::vector<float> state(_layerDescs[l]._width * _layerDescs[l]._height * _layerDescs[l]._cellsInColumn);

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

				color.a = state[ci + x * _layerDescs[l]._cellsInColumn + y * _layerDescs[l]._width * _layerDescs[l]._cellsInColumn] * (255.0f - 3.0f) + 3;

				int wx = x - _layerDescs[l]._width / 2 + maxWidth / 2;
				int wy = y - _layerDescs[l]._height / 2 + maxHeight / 2;

				assert(wx >= 0 && wy >= 0 && wx < maxWidth && wy < maxHeight);

				image->setPixel(wx, wy, color);
			}

			images.push_back(image);
		}
	}*/
}