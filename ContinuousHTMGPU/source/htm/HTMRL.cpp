#include <htm/HTMRL.h>

#include <iostream>

using namespace htm;

void HTMRL::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, float minInitCenter, float maxInitCenter, std::mt19937 &generator) {
	struct Uint2 {
		unsigned int _x, _y;
	};
	
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
	
	_input.clear();
	_input.assign(_inputWidth * _inputHeight, 0.0f);

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
	else if (_inputTypes[i] == _q)
		_qIndices.push_back(i);

	_qEncoder.create(_qIndices.size(), 1, minInitCenter, maxInitCenter, generator);

	_inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;
	int prevCellsPerColumn = 1;

	for (int l = 0; l < _layers.size(); l++) {
		initLayer(cs, initPartOneKernel, initPartTwoKernel, prevWidth, prevHeight, prevCellsPerColumn, _layers[l], _layerDescs[l], l == _layers.size() - 1, minInitWeight, maxInitWeight, minInitCenter, maxInitCenter, minInitWeight, maxInitWeight, generator);

		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
		prevCellsPerColumn = _layerDescs[l]._cellsInColumn;
	}

	_reconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

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
	_layerColumnQKernel = cl::Kernel(program.getProgram(), "layerColumnQ");
	_layerAssignQKernel = cl::Kernel(program.getProgram(), "layerAssignQ");
	_layerTdErrorKernel = cl::Kernel(program.getProgram(), "layerTdError");

	_gaussianBlurXKernel = cl::Kernel(program.getProgram(), "gaussianBlurX");
	_gaussianBlurYKernel = cl::Kernel(program.getProgram(), "gaussianBlurY");

	_reconstructInputKernel = cl::Kernel(program.getProgram(), "reconstructInput");
}

void HTMRL::initLayer(sys::ComputeSystem &cs, cl::Kernel &initPartOneKernel, cl::Kernel &initPartTwoKernel, int inputWidth, int inputHeight, int inputCellsPerColumn, Layer &layer, const LayerDesc &layerDesc, bool isTopmost, float minInitWeight, float maxInitWeight, float minInitCenter, float maxInitCenter, float minInitWidth, float maxInitWidth, std::mt19937 &generator) {
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

	layer._columnActivations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	
	layer._columnStates = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._columnWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, receptiveFieldSize + 1); // + 1 for width
	layer._columnWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, receptiveFieldSize + 1); // + 1 for width

	layer._columnDutyCycles = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnDutyCyclesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._cellStates = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellStatesPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._cellQValues = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellQValuesPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._columnQValues = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	
	layer._columnPrevValues = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnPrevValuesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._columnTdErrors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._cellPredictions = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellPredictionsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._cellWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, lateralConnectionsSize);
	layer._cellWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, lateralConnectionsSize);

	layer._columnPredictions = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnPredictionsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._blurPing = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._blurPong = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	Uint2 seed1;
	seed1._x = uniformDist(generator);
	seed1._y = uniformDist(generator);

	initPartOneKernel.setArg(0, layer._columnActivations);
	initPartOneKernel.setArg(1, layer._columnStates);
	initPartOneKernel.setArg(2, layer._columnWeights);
	initPartOneKernel.setArg(3, layer._columnDutyCycles);
	initPartOneKernel.setArg(4, layer._columnPrevValues);
	initPartOneKernel.setArg(5, layerDesc._cellsInColumn);
	initPartOneKernel.setArg(6, receptiveFieldSize);
	initPartOneKernel.setArg(7, lateralConnectionsSize);
	initPartOneKernel.setArg(8, seed1);
	initPartOneKernel.setArg(9, minInitCenter);
	initPartOneKernel.setArg(10, maxInitCenter);
	initPartOneKernel.setArg(11, minInitWidth);
	initPartOneKernel.setArg(12, maxInitWidth);

	cs.getQueue().enqueueNDRangeKernel(initPartOneKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	Uint2 seed2;
	seed2._x = uniformDist(generator);
	seed2._y = uniformDist(generator);

	initPartTwoKernel.setArg(0, layer._cellStates);
	initPartTwoKernel.setArg(1, layer._cellWeights);
	initPartTwoKernel.setArg(2, layer._cellPredictions);
	initPartTwoKernel.setArg(3, layer._cellQValues);
	initPartTwoKernel.setArg(4, layerDesc._cellsInColumn);
	initPartTwoKernel.setArg(5, receptiveFieldSize);
	initPartTwoKernel.setArg(6, lateralConnectionsSize);
	initPartTwoKernel.setArg(7, seed2);
	initPartTwoKernel.setArg(8, minInitWeight);
	initPartTwoKernel.setArg(9, maxInitWeight);

	cs.getQueue().enqueueNDRangeKernel(initPartTwoKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

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
}

void HTMRL::stepBegin() {
	for (int l = 0; l < _layers.size(); l++) {
		std::swap(_layers[l]._columnStates, _layers[l]._columnStatesPrev);
		std::swap(_layers[l]._columnPredictions, _layers[l]._columnPredictionsPrev);
		std::swap(_layers[l]._columnDutyCycles, _layers[l]._columnDutyCyclesPrev);
		std::swap(_layers[l]._columnWeights, _layers[l]._columnWeightsPrev);
		std::swap(_layers[l]._columnPrevValues, _layers[l]._columnPrevValuesPrev);
		std::swap(_layers[l]._cellStates, _layers[l]._cellStatesPrev);
		std::swap(_layers[l]._cellQValues, _layers[l]._cellQValuesPrev);
		std::swap(_layers[l]._cellPredictions, _layers[l]._cellPredictionsPrev);
		std::swap(_layers[l]._cellWeights, _layers[l]._cellWeightsPrev);
	}
}

void HTMRL::activateLayer(sys::ComputeSystem &cs, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, Layer &layer, const LayerDesc &layerDesc, float cellStateDecay, std::mt19937 &generator) {
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
	_layerColumnActivateKernel.setArg(1, layer._columnWeightsPrev);
	_layerColumnActivateKernel.setArg(2, layer._columnDutyCyclesPrev);
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
	_layerColumnInhibitKernel.setArg(6, layerDesc._noMatchTolerance);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnInhibitKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

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
	_layerCellActivateKernel.setArg(9, seed2);

	cs.getQueue().enqueueNDRangeKernel(_layerCellActivateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
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

	_layerCellPredictKernel.setArg(0, layer._columnStates);
	_layerCellPredictKernel.setArg(1, layer._cellStates);
	_layerCellPredictKernel.setArg(2, layer._cellWeightsPrev);
	_layerCellPredictKernel.setArg(3, nextLayerPrediction);
	_layerCellPredictKernel.setArg(4, layer._cellPredictions);
	_layerCellPredictKernel.setArg(5, layerDesc._cellsInColumn);
	_layerCellPredictKernel.setArg(6, layerSize);
	_layerCellPredictKernel.setArg(7, lateralConnectionRadii);
	_layerCellPredictKernel.setArg(8, layerSizeMinusOneInv);
	_layerCellPredictKernel.setArg(9, nextLayerSize);
	_layerCellPredictKernel.setArg(10, nextLayerSizeMinusOne);

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

void HTMRL::activate(std::vector<float> &input, sys::ComputeSystem &cs, float reward, float alpha, float gamma, float cellStateDecay, float activationDutyCycleDecay, float stateDutyCycleDecay, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, unsigned long seed) {
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
		activateLayer(cs, *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l], _layerDescs[l], cellStateDecay, generator);
		
		// Blur output
		gaussianBlur(cs, _layers[l]._columnStates, _layers[l]._blurPing, _layers[l]._blurPong, _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._numColumnStateBlurPasses, _layerDescs[l]._columnStateBlurKernelWidthMultiplier);

		pPrevLayerOutput = &_layers[l]._blurPong;
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}

	for (int l = _layers.size() - 1; l >= 0; l--) {
		if (l == _layers.size() - 1)
			predictLayerLast(cs, _layers[l], _layerDescs[l], generator);
		else
			predictLayer(cs, _layers[l + 1]._columnPredictions, _layerDescs[l + 1]._width, _layerDescs[l + 1]._height, _layers[l], _layerDescs[l], generator);	
	}

	for (int l = _layers.size() - 1; l >= 0; l--)
		determineLayerColumnQ(cs, _layers[l], _layerDescs[l]);

	for (int l = _layers.size() - 1; l >= 0; l--)
		determineLayerTdError(cs, _layers[l], _layerDescs[l], reward, alpha, gamma);

	for (int l = _layers.size() - 1; l >= 0; l--)
		gaussianBlur(cs, _layers[l]._columnTdErrors, _layers[l]._blurPing, _layers[l]._blurPong, _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._numTdErrorBlurPasses, _layerDescs[l]._tdErrorBlurKernelWidthMultiplier);

	for (int l = _layers.size() - 1; l >= 0; l--)
		assignLayerQ(cs, _layers[l], _layerDescs[l], alpha);

	dutyCycleUpdate(cs, activationDutyCycleDecay, stateDutyCycleDecay);

	learnSpatial(cs, columnConnectionAlpha, widthAlpha, seed);

	learnTemporal(cs, cellConnectionAlpha, cellConnectionBeta, cellConnectionTemperature, cellWeightEligibilityDecay, seed + 1);
}

void HTMRL::determineLayerColumnQ(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	_layerColumnQKernel.setArg(0, layer._cellQValuesPrev);
	_layerColumnQKernel.setArg(1, layer._cellStatesPrev);
	_layerColumnQKernel.setArg(2, layer._cellStates);
	_layerColumnQKernel.setArg(3, layer._columnQValues);
	_layerColumnQKernel.setArg(4, layerDesc._cellsInColumn);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnQKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::determineLayerTdError(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc, float reward, float alpha, float gamma) {
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

	Int2 qConnectionRadii;
	qConnectionRadii._x = layerDesc._columnQRadius;
	qConnectionRadii._y = layerDesc._columnQRadius;

	_layerTdErrorKernel.setArg(0, layer._cellStatesPrev);
	_layerTdErrorKernel.setArg(1, layer._columnStatesPrev);
	_layerTdErrorKernel.setArg(2, layer._columnStates);
	_layerTdErrorKernel.setArg(3, layer._columnDutyCyclesPrev);
	_layerTdErrorKernel.setArg(4, layer._columnQValues);
	_layerTdErrorKernel.setArg(5, layer._columnPrevValuesPrev);
	_layerTdErrorKernel.setArg(6, layer._columnTdErrors);
	_layerTdErrorKernel.setArg(7, layer._columnPrevValues);
	_layerTdErrorKernel.setArg(8, layerDesc._cellsInColumn);
	_layerTdErrorKernel.setArg(9, layerSize);
	_layerTdErrorKernel.setArg(10, qConnectionRadii);
	_layerTdErrorKernel.setArg(11, reward);
	_layerTdErrorKernel.setArg(12, alpha);
	_layerTdErrorKernel.setArg(13, gamma);

	cs.getQueue().enqueueNDRangeKernel(_layerTdErrorKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
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

	_layerAssignQKernel.setArg(0, layer._blurPong); // Blurred td error
	_layerAssignQKernel.setArg(1, layer._cellQValuesPrev);
	_layerAssignQKernel.setArg(2, layer._cellStatesPrev);
	_layerAssignQKernel.setArg(3, layer._cellQValues);
	_layerAssignQKernel.setArg(4, layerDesc._cellsInColumn);

	cs.getQueue().enqueueNDRangeKernel(_layerAssignQKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::learnLayerSpatial(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, std::mt19937 &generator) {
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

	Int2 inputSizeMinusOne;
	inputSizeMinusOne._x = layerDesc._width - 1;
	inputSizeMinusOne._y = layerDesc._height - 1;

	// Column weight update
	_layerColumnWeightUpdateKernel.setArg(0, prevLayerOutput);
	_layerColumnWeightUpdateKernel.setArg(1, layer._columnActivations);
	_layerColumnWeightUpdateKernel.setArg(2, layer._columnStates);
	_layerColumnWeightUpdateKernel.setArg(3, layer._columnStatesPrev);
	_layerColumnWeightUpdateKernel.setArg(4, layer._columnWeightsPrev);
	_layerColumnWeightUpdateKernel.setArg(5, layer._columnDutyCyclesPrev);
	_layerColumnWeightUpdateKernel.setArg(6, layer._columnWeights);
	_layerColumnWeightUpdateKernel.setArg(7, layerSizeMinusOneInv);
	_layerColumnWeightUpdateKernel.setArg(8, inputReceptiveFieldRadius);
	_layerColumnWeightUpdateKernel.setArg(9, inputSize);
	_layerColumnWeightUpdateKernel.setArg(10, inputSizeMinusOne);
	_layerColumnWeightUpdateKernel.setArg(11, columnConnectionAlpha);
	_layerColumnWeightUpdateKernel.setArg(12, widthAlpha);
	_layerColumnWeightUpdateKernel.setArg(13, seed);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::learnLayerTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, std::mt19937 &generator) {
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

	_layerCellWeightUpdateKernel.setArg(0, layer._columnStatesPrev);
	_layerCellWeightUpdateKernel.setArg(1, layer._columnStates);
	_layerCellWeightUpdateKernel.setArg(2, layer._cellPredictionsPrev);
	_layerCellWeightUpdateKernel.setArg(3, layer._cellStates);
	_layerCellWeightUpdateKernel.setArg(4, layer._cellStatesPrev);
	_layerCellWeightUpdateKernel.setArg(5, nextLayerPrediction);
	_layerCellWeightUpdateKernel.setArg(6, layer._cellWeightsPrev);
	_layerCellWeightUpdateKernel.setArg(7, layer._columnPredictionsPrev);
	_layerCellWeightUpdateKernel.setArg(8, layer._blurPong); // Blurred td error
	_layerCellWeightUpdateKernel.setArg(9, layer._cellWeights);
	_layerCellWeightUpdateKernel.setArg(10, layerDesc._cellsInColumn);
	_layerCellWeightUpdateKernel.setArg(11, layerSize);
	_layerCellWeightUpdateKernel.setArg(12, lateralConnectionRadii);
	_layerCellWeightUpdateKernel.setArg(13, layerSizeMinusOneInv);
	_layerCellWeightUpdateKernel.setArg(14, nextLayerSize);
	_layerCellWeightUpdateKernel.setArg(15, nextLayerSizeMinusOne);
	_layerCellWeightUpdateKernel.setArg(16, cellConnectionAlpha);
	_layerCellWeightUpdateKernel.setArg(17, cellConnectionBeta);
	_layerCellWeightUpdateKernel.setArg(18, cellConnectionTemperature);
	_layerCellWeightUpdateKernel.setArg(19, cellWeightEligibilityDecay);

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::learnLayerTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, std::mt19937 &generator) {
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
	_layerCellWeightUpdateLastKernel.setArg(0, layer._columnStatesPrev);
	_layerCellWeightUpdateLastKernel.setArg(1, layer._columnStates);
	_layerCellWeightUpdateLastKernel.setArg(2, layer._cellPredictionsPrev);
	_layerCellWeightUpdateLastKernel.setArg(3, layer._cellStates);
	_layerCellWeightUpdateLastKernel.setArg(4, layer._cellStatesPrev);
	_layerCellWeightUpdateLastKernel.setArg(5, layer._cellWeightsPrev);
	_layerCellWeightUpdateLastKernel.setArg(6, layer._columnPredictionsPrev);
	_layerCellWeightUpdateLastKernel.setArg(7, layer._blurPong); // Blurred td error
	_layerCellWeightUpdateLastKernel.setArg(8, layer._cellWeights);
	_layerCellWeightUpdateLastKernel.setArg(9, layerDesc._cellsInColumn);
	_layerCellWeightUpdateLastKernel.setArg(10, layerSize);
	_layerCellWeightUpdateLastKernel.setArg(11, lateralConnectionRadii);
	_layerCellWeightUpdateLastKernel.setArg(12, cellConnectionAlpha);
	_layerCellWeightUpdateLastKernel.setArg(13, cellConnectionBeta);
	_layerCellWeightUpdateLastKernel.setArg(14, cellConnectionTemperature);
	_layerCellWeightUpdateLastKernel.setArg(15, cellWeightEligibilityDecay);

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateLastKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::learnSpatial(sys::ComputeSystem &cs, float columnConnectionAlpha, float widthAlpha, unsigned long seed) {
	std::mt19937 generator(seed);

	cl::Image2D* pPrevLayerOutput = &_inputImage;
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		learnLayerSpatial(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], columnConnectionAlpha, widthAlpha, generator);

		pPrevLayerOutput = &_layers[l]._columnStates;
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}
}

void HTMRL::learnTemporal(sys::ComputeSystem &cs, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, unsigned long seed) {
	std::mt19937 generator(seed);

	cl::Image2D* pPrevLayerOutput = &_inputImage;
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		if (l == _layers.size() - 1)
			learnLayerTemporalLast(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], cellConnectionAlpha, cellConnectionBeta, cellConnectionTemperature, cellWeightEligibilityDecay, generator);
		else
			learnLayerTemporal(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l + 1]._columnPredictionsPrev, _layerDescs[l + 1]._width, _layerDescs[l + 1]._width, _layerDescs[l], cellConnectionAlpha, cellConnectionBeta, cellConnectionTemperature, cellWeightEligibilityDecay, generator);

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

	Int2 layerDutyCycleRadius;
	layerDutyCycleRadius._x = layerDesc._dutyCycleRadius;
	layerDutyCycleRadius._y = layerDesc._dutyCycleRadius;

	// Duty cycles
	_layerColumnDutyCycleUpdateKernel.setArg(0, layer._columnActivations);
	_layerColumnDutyCycleUpdateKernel.setArg(1, layer._columnStates);
	_layerColumnDutyCycleUpdateKernel.setArg(2, layer._columnDutyCyclesPrev);
	_layerColumnDutyCycleUpdateKernel.setArg(3, layer._columnDutyCycles);
	_layerColumnDutyCycleUpdateKernel.setArg(4, layerSize);
	_layerColumnDutyCycleUpdateKernel.setArg(5, layerDutyCycleRadius);
	_layerColumnDutyCycleUpdateKernel.setArg(6, activationDutyCycleDecay);
	_layerColumnDutyCycleUpdateKernel.setArg(7, stateDutyCycleDecay);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnDutyCycleUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::dutyCycleUpdate(sys::ComputeSystem &cs, float activationDutyCycleDecay, float stateDutyCycleDecay) {
	for (int l = 0; l < _layers.size(); l++)
		dutyCycleLayerUpdate(cs, _layers[l], _layerDescs[l], activationDutyCycleDecay, stateDutyCycleDecay);
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

void HTMRL::getReconstruction(std::vector<float> &prediction, sys::ComputeSystem &cs) {
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

	_reconstructInputKernel.setArg(0, _layers.front()._columnWeightsPrev);
	_reconstructInputKernel.setArg(1, _layers.front()._columnStates);
	_reconstructInputKernel.setArg(2, _reconstruction);
	_reconstructInputKernel.setArg(3, reconstructionReceptiveFieldRadii);
	_reconstructInputKernel.setArg(4, sdrReceptiveFieldRadii);
	_reconstructInputKernel.setArg(5, inputSizeMinusOne);
	_reconstructInputKernel.setArg(6, inputSizeMinusOneInv);
	_reconstructInputKernel.setArg(7, layerSize);
	_reconstructInputKernel.setArg(8, sdrSizeMinusOne);
	_reconstructInputKernel.setArg(9, sdrSizeMinusOneInv);

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

		cs.getQueue().enqueueReadImage(_reconstruction, CL_TRUE, origin, region, 0, 0, &prediction[0]);
	}
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

	_reconstructInputKernel.setArg(0, _layers.front()._columnWeightsPrev);
	_reconstructInputKernel.setArg(1, _layers.front()._columnPredictions);
	_reconstructInputKernel.setArg(2, _reconstruction);
	_reconstructInputKernel.setArg(3, reconstructionReceptiveFieldRadii);
	_reconstructInputKernel.setArg(4, sdrReceptiveFieldRadii);
	_reconstructInputKernel.setArg(5, inputSizeMinusOne);
	_reconstructInputKernel.setArg(6, inputSizeMinusOneInv);
	_reconstructInputKernel.setArg(7, layerSize);
	_reconstructInputKernel.setArg(8, sdrSizeMinusOne);
	_reconstructInputKernel.setArg(9, sdrSizeMinusOneInv);

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

		cs.getQueue().enqueueReadImage(_reconstruction, CL_TRUE, origin, region, 0, 0, &prediction[0]);
	}
}

void HTMRL::getReconstructedPrevPrediction(std::vector<float> &prediction, sys::ComputeSystem &cs) {
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

	_reconstructInputKernel.setArg(0, _layers.front()._columnWeightsPrev);
	_reconstructInputKernel.setArg(1, _layers.front()._columnPredictionsPrev);
	_reconstructInputKernel.setArg(2, _reconstruction);
	_reconstructInputKernel.setArg(3, reconstructionReceptiveFieldRadii);
	_reconstructInputKernel.setArg(4, sdrReceptiveFieldRadii);
	_reconstructInputKernel.setArg(5, inputSizeMinusOne);
	_reconstructInputKernel.setArg(6, inputSizeMinusOneInv);
	_reconstructInputKernel.setArg(7, layerSize);
	_reconstructInputKernel.setArg(8, sdrSizeMinusOne);
	_reconstructInputKernel.setArg(9, sdrSizeMinusOneInv);

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

		cs.getQueue().enqueueReadImage(_reconstruction, CL_TRUE, origin, region, 0, 0, &prediction[0]);
	}
}

float HTMRL::reconstructQFromInput(const std::vector<float> &input) {
	std::vector<float> recon;
	_qEncoder.decode(input, recon);

	return recon[0];
}

void HTMRL::assignInputsFromQ(std::vector<float> &input, float q, float encoderLocalActivity, float encoderOutputIntensity, float encoderDutyCycleDecay) {
	std::vector<float> sdr;
	_qEncoder.encode(std::vector<float>(1, q), sdr, encoderLocalActivity, encoderOutputIntensity, encoderDutyCycleDecay);

	for (int i = 0; i < _qIndices.size(); i++)
		input[_qIndices[i]] = sdr[i];
}

void HTMRL::learnQReconstruction(float q, float encoderCenterAlpha, float encoderMaxDutyCycleForLearn, float encoderNoMatchIntensity) {
	_qEncoder.learn(std::vector<float>(1, q), encoderCenterAlpha, encoderMaxDutyCycleForLearn, encoderNoMatchIntensity);
}

void HTMRL::step(sys::ComputeSystem &cs, float reward, float cellStateDecay, float columnConnectionAlpha, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, float activationDutyCycleDecay, float stateDutyCycleDecay, float reconstructionAlpha, float alpha, float gamma, float tauInv, float breakChance, float perturbationStdDev, float maxTdError, std::mt19937 &generator) {
	std::uniform_int_distribution<int> seedDist(0, 10000);

	unsigned long seed = seedDist(generator);

	stepBegin();

	activate(_input, cs, reward, alpha, gamma, cellStateDecay, activationDutyCycleDecay, stateDutyCycleDecay, columnConnectionAlpha, 0.0f, cellConnectionAlpha, cellConnectionBeta, cellConnectionTemperature, cellWeightEligibilityDecay, seed);

	getReconstructedPrediction(_input, cs);

	// Exploratory action
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, perturbationStdDev);

	for (int i = 0; i < _input.size(); i++)
	if (_inputTypes[i] == _action) {
		if (dist01(generator) < breakChance)
			_input[i] = dist01(generator);
		else
			_input[i] = std::min<float>(1.0f, std::max<float>(0.0f, std::min<float>(1.0f, std::max<float>(0.0f, _input[i])) + pertDist(generator)));
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

		cs.getQueue().enqueueReadImage(_reconstruction, CL_TRUE, origin, region, 0, 0, &state[0]);

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
			std::vector<float> state(_layerDescs[l]._width * _layerDescs[l]._height * _layerDescs[l]._cellsInColumn);

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

					color.a = std::min<float>(1.0f, std::max<float>(0.0f, state[(x + y * _layerDescs[l]._width + ci * _layerDescs[l]._width *_layerDescs[l]._height)])) * (255.0f - 3.0f) + 3;

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
		/*for (int l = 0; l < _layers.size(); l++) {
			std::vector<float> state(_layerDescs[l]._width * _layerDescs[l]._height * _layerDescs[l]._cellsInColumn * 2);

			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = _layerDescs[l]._cellsInColumn;

			cs.getQueue().enqueueReadImage(_layers[l]._cellQValues, CL_TRUE, origin, region, 0, 0, &state[0]);

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

					color.a = std::min<float>(1.0f, std::max<float>(0.0f, std::max<float>(0.0f, state[1 + 2 * (x + y * _layerDescs[l]._width + ci * _layerDescs[l]._width *_layerDescs[l]._height)]))) * (255.0f - 3.0f) + 3;

					int wx = x - _layerDescs[l]._width / 2 + maxWidth / 2;
					int wy = y - _layerDescs[l]._height / 2 + maxHeight / 2;

					assert(wx >= 0 && wy >= 0 && wx < maxWidth && wy < maxHeight);

					image->setPixel(wx, wy, color);
				}

				images.push_back(image);
			}
		}*/

		for (int l = 0; l < _layers.size(); l++) {
			std::vector<float> state(_layerDescs[l]._width * _layerDescs[l]._height);

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

				color.a = std::min<float>(1.0f, std::max<float>(0.0f, state[(x + y * _layerDescs[l]._width)])) * (255.0f - 3.0f) + 3;

				int wx = x - _layerDescs[l]._width / 2 + maxWidth / 2;
				int wy = y - _layerDescs[l]._height / 2 + maxHeight / 2;

				assert(wx >= 0 && wy >= 0 && wx < maxWidth && wy < maxHeight);

				image->setPixel(wx, wy, color);
			}

			images.push_back(image);
		}
	}
}