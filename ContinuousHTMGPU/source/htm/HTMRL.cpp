#include <htm/HTMRL.h>

#include <iostream>

using namespace htm;

void HTMRL::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<bool> &actionMask, float minInitWeight, float maxInitWeight, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_layerDescs = layerDescs;
	
	_layers.resize(_layerDescs.size());

	_actionMask = actionMask;

	std::uniform_real_distribution<float> weightDist(minInitWeight, maxInitWeight);
	std::uniform_real_distribution<float> actionDist(0.0f, 1.0f);

	_qBias = weightDist(generator);
	_prevMaxQ = 0.0f;
	_prevValue = 0.0f;

	cl::Kernel initPartOneKernel = cl::Kernel(program.getProgram(), "initializePartOne");
	cl::Kernel initPartTwoKernel = cl::Kernel(program.getProgram(), "initializePartTwo");

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
	if (_actionMask[i]) {
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

	for (int l = 0; l < _layers.size(); l++) {
		initLayer(cs, initPartOneKernel, initPartTwoKernel, prevWidth, prevHeight, _layers[l], _layerDescs[l], l == _layers.size() - 1, minInitWeight, maxInitWeight, generator);

		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
	}

	_reconstructionReceptiveRadius = std::ceil(static_cast<float>(_layerDescs.front()._width) / static_cast<float>(_inputWidth) * static_cast<float>(_layerDescs.front()._receptiveFieldRadius));

	_reconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

	_layerColumnActivateKernel = cl::Kernel(program.getProgram(), "layerColumnActivate");
	_layerColumnInhibitKernel = cl::Kernel(program.getProgram(), "layerColumnInhibit");
	_layerExploreKernel = cl::Kernel(program.getProgram(), "layerExplore");
	_layerColumnDutyCycleUpdateKernel = cl::Kernel(program.getProgram(), "layerColumnDutyCycleUpdate");
	_layerCellActivateKernel = cl::Kernel(program.getProgram(), "layerCellActivate");
	_layerCellWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerCellWeightUpdate");
	_layerCellWeightUpdateLastKernel = cl::Kernel(program.getProgram(), "layerCellWeightUpdateLast");
	_layerCellPredictKernel = cl::Kernel(program.getProgram(), "layerCellPredict");
	_layerCellPredictLastKernel = cl::Kernel(program.getProgram(), "layerCellPredictLast");
	_layerColumnWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerColumnWeightUpdate");
	_layerColumnPredictionKernel = cl::Kernel(program.getProgram(), "layerColumnPrediction");
	_layerRetrieveQKernel = cl::Kernel(program.getProgram(), "layerRetrieveQ");
	_layerUpdateQWeightsKernel = cl::Kernel(program.getProgram(), "layerUpdateQWeights");
	_reconstructInputKernel = cl::Kernel(program.getProgram(), "reconstructInput");
}

void HTMRL::initLayer(sys::ComputeSystem &cs, cl::Kernel &initPartOneKernel, cl::Kernel &initPartTwoKernel, int inputWidth, int inputHeight, Layer &layer, const LayerDesc &layerDesc, bool isTopmost, float minInitWeight, float maxInitWeight, std::mt19937 &generator) {
	struct Uint2 {
		unsigned int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};
	
	std::uniform_int_distribution<int> uniformDist(0, 10000);

	int receptiveFieldSize = std::pow(layerDesc._receptiveFieldRadius * 2 + 1, 2);
	int lateralConnectionsSize;

	// If not the last layer, add weights for additional context from next layer
	if (isTopmost)
		lateralConnectionsSize = std::pow(layerDesc._lateralConnectionRadius * 2 + 1, 2) * (layerDesc._cellsInColumn + 1) + 1; // + 1 for bias
	else
		lateralConnectionsSize = std::pow(layerDesc._lateralConnectionRadius * 2 + 1, 2) * layerDesc._cellsInColumn + 1; // + 1 for bias

	layer._columnActivations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	
	layer._columnStates = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._columnWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, receptiveFieldSize);
	layer._columnWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, receptiveFieldSize);

	layer._columnDutyCycles = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnDutyCyclesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._cellStates = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellStatesPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._cellPredictions = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellPredictionsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._cellWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, lateralConnectionsSize);
	layer._cellWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, lateralConnectionsSize);

	layer._columnPredictions = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnPredictionsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._cellQWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellQWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._partialQSums = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

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

	cs.getQueue().enqueueNDRangeKernel(initPartOneKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	Uint2 seed2;
	seed2._x = uniformDist(generator);
	seed2._y = uniformDist(generator);

	initPartTwoKernel.setArg(0, layer._cellStates);
	initPartTwoKernel.setArg(1, layer._cellWeights);
	initPartTwoKernel.setArg(2, layer._cellPredictions);
	initPartTwoKernel.setArg(3, layer._cellQWeights);
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

		cs.getQueue().enqueueCopyImage(layer._cellQWeights, layer._cellQWeightsPrev, origin, origin, region);
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
		std::swap(_layers[l]._cellQWeights, _layers[l]._cellQWeightsPrev);
	}
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
	inputReceptiveFieldStep._x = inputSizeInv._x * layerDesc._receptiveFieldRadius;
	inputReceptiveFieldStep._y = inputSizeInv._y * layerDesc._receptiveFieldRadius;

	// Activation
	_layerColumnActivateKernel.setArg(0, prevLayerOutput);
	_layerColumnActivateKernel.setArg(1, layer._columnWeightsPrev);
	_layerColumnActivateKernel.setArg(2, layer._columnDutyCyclesPrev);
	_layerColumnActivateKernel.setArg(3, layer._columnActivations);
	_layerColumnActivateKernel.setArg(4, layerSizeInv);
	_layerColumnActivateKernel.setArg(5, inputReceptiveFieldRadius);
	_layerColumnActivateKernel.setArg(6, inputReceptiveFieldStep);
	_layerColumnActivateKernel.setArg(7, inputSize);
	_layerColumnActivateKernel.setArg(8, seed);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnActivateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	Int2 layerInhibitionRadius;
	layerInhibitionRadius._x = layerDesc._inhibitionRadius;
	layerInhibitionRadius._y = layerDesc._inhibitionRadius;

	Float2 layerInhibitionStep;
	layerInhibitionStep._x = layerSizeInv._x * layerDesc._inhibitionRadius;
	layerInhibitionStep._y = layerSizeInv._y * layerDesc._inhibitionRadius;

	// Inhibition
	_layerColumnInhibitKernel.setArg(0, layer._columnActivations);
	_layerColumnInhibitKernel.setArg(1, layer._columnStates);
	_layerColumnInhibitKernel.setArg(2, layerSize);
	_layerColumnInhibitKernel.setArg(3, layerSizeInv);
	_layerColumnInhibitKernel.setArg(4, layerInhibitionRadius);
	_layerColumnInhibitKernel.setArg(5, seed);

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

float HTMRL::retreiveLayerQ(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc) {
	float totalSum = 0.0f;

	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	_layerRetrieveQKernel.setArg(0, layer._columnStates);
	_layerRetrieveQKernel.setArg(1, layer._columnDutyCycles);
	_layerRetrieveQKernel.setArg(2, layer._cellStates);
	_layerRetrieveQKernel.setArg(3, layer._cellQWeightsPrev);
	_layerRetrieveQKernel.setArg(4, layer._partialQSums);
	_layerRetrieveQKernel.setArg(5, layerDesc._cellsInColumn);

	cs.getQueue().enqueueNDRangeKernel(_layerRetrieveQKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();

	// Retrieve result
	std::vector<float> result(layerDesc._width * layerDesc._height * 2);

	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = layerDesc._width;
		region[1] = layerDesc._height;
		region[2] = 1;

		cs.getQueue().enqueueReadImage(layer._partialQSums, CL_TRUE, origin, region, 0, 0, &result[0]);
	}

	for (int i = 0; i < result.size(); i++)
		totalSum += result[i] * layerDesc._qInfluenceMultiplier;

	return totalSum;
}

float HTMRL::retrieveQ(sys::ComputeSystem &cs) {
	float totalSum = _qBias;

	for (int l = 0; l < _layers.size(); l++)
		totalSum += retreiveLayerQ(cs, _layers[l], _layerDescs[l]);

	return totalSum;
}

void HTMRL::learnLayerSpatial(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float cellConnectionAlpha, std::mt19937 &generator) {
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
	layerReceptiveFieldStep._x = layerSizeInv._x * layerDesc._receptiveFieldRadius;
	layerReceptiveFieldStep._y = layerSizeInv._y * layerDesc._receptiveFieldRadius;

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
	_layerColumnWeightUpdateKernel.setArg(12, seed);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::learnLayerSpatialLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float cellConnectionAlpha, std::mt19937 &generator) {
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
	layerReceptiveFieldStep._x = layerSizeInv._x * layerDesc._receptiveFieldRadius;
	layerReceptiveFieldStep._y = layerSizeInv._y * layerDesc._receptiveFieldRadius;

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
	_layerColumnWeightUpdateKernel.setArg(12, seed);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::learnLayerSpatialTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float cellConnectionAlpha, std::mt19937 &generator) {
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
	layerReceptiveFieldStep._x = layerSizeInv._x * layerDesc._receptiveFieldRadius;
	layerReceptiveFieldStep._y = layerSizeInv._y * layerDesc._receptiveFieldRadius;

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
	_layerColumnWeightUpdateKernel.setArg(12, seed);

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

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::learnLayerSpatialTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float cellConnectionAlpha, std::mt19937 &generator) {
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
	layerReceptiveFieldStep._x = layerSizeInv._x * layerDesc._receptiveFieldRadius;
	layerReceptiveFieldStep._y = layerSizeInv._y * layerDesc._receptiveFieldRadius;

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
	_layerColumnWeightUpdateKernel.setArg(12, seed);

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

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateLastKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::learnSpatial(sys::ComputeSystem &cs, float columnConnectionAlpha, float cellConnectionAlpha, unsigned long seed) {
	std::mt19937 generator(seed);

	cl::Image2D* pPrevLayerOutput = &_inputImage;
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		if (l == _layers.size() - 1)
			learnLayerSpatialLast(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], columnConnectionAlpha, cellConnectionAlpha, generator);
		else
			learnLayerSpatial(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l + 1]._columnPredictions, _layerDescs[l + 1]._width, _layerDescs[l + 1]._width, _layerDescs[l], columnConnectionAlpha, cellConnectionAlpha, generator);

		pPrevLayerOutput = &_layers[l]._columnStates;
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}
}

void HTMRL::learnSpatialTemporal(sys::ComputeSystem &cs, float columnConnectionAlpha, float cellConnectionAlpha, unsigned long seed) {
	std::mt19937 generator(seed);

	cl::Image2D* pPrevLayerOutput = &_inputImage;
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;
	
	for (int l = 0; l < _layers.size(); l++) {
		if (l == _layers.size() - 1)
			learnLayerSpatialTemporalLast(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], columnConnectionAlpha, cellConnectionAlpha, generator);
		else
			learnLayerSpatialTemporal(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l + 1]._columnPredictions, _layerDescs[l + 1]._width, _layerDescs[l + 1]._width, _layerDescs[l], columnConnectionAlpha, cellConnectionAlpha, generator);
	
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

void HTMRL::updateLayerQWeights(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, float tdError, float cellQWeightEligibilityDecay) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	_layerUpdateQWeightsKernel.setArg(0, layer._columnStates);
	_layerUpdateQWeightsKernel.setArg(1, layer._columnDutyCycles);
	_layerUpdateQWeightsKernel.setArg(2, layer._cellStates);
	_layerUpdateQWeightsKernel.setArg(3, layer._cellQWeightsPrev);
	_layerUpdateQWeightsKernel.setArg(4, layer._cellQWeights);
	_layerUpdateQWeightsKernel.setArg(5, layerDesc._cellsInColumn);
	_layerUpdateQWeightsKernel.setArg(6, cellQWeightEligibilityDecay);
	_layerUpdateQWeightsKernel.setArg(7, tdError * layerDesc._qInfluenceMultiplier);

	cs.getQueue().enqueueNDRangeKernel(_layerUpdateQWeightsKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::updateQWeights(sys::ComputeSystem &cs, float tdError, float cellQWeightEligibilityDecay, float qBiasAlpha) {
	_qBias += qBiasAlpha * tdError;

	for (int l = _layers.size() - 1; l >= 0; l--)
		updateLayerQWeights(cs, _layers[l], _layerDescs[l], tdError, cellQWeightEligibilityDecay);
}

void HTMRL::getReconstructedPrediction(std::vector<float> &prediction, sys::ComputeSystem &cs) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Float2 inputSizeInv;
	inputSizeInv._x = 1.0f / _inputWidth;
	inputSizeInv._y = 1.0f / _inputHeight;

	Int2 layerSize;
	layerSize._x = _layerDescs.front()._width;
	layerSize._y = _layerDescs.front()._height;

	Float2 layerSizeInv;
	layerSizeInv._x = 1.0f / _layerDescs.front()._width;
	layerSizeInv._y = 1.0f / _layerDescs.front()._height;

	Int2 reconstructionReceptiveFieldRadii;
	reconstructionReceptiveFieldRadii._x = _reconstructionReceptiveRadius;
	reconstructionReceptiveFieldRadii._y = _reconstructionReceptiveRadius;

	Int2 sdrReceptiveFieldRadii;
	sdrReceptiveFieldRadii._x = _layerDescs.front()._receptiveFieldRadius;
	sdrReceptiveFieldRadii._y = _layerDescs.front()._receptiveFieldRadius;

	Float2 inputOverSdr;
	inputOverSdr._x = static_cast<float>(_inputWidth) / _layerDescs.front()._width;
	inputOverSdr._y = static_cast<float>(_inputHeight) / _layerDescs.front()._height;

	_reconstructInputKernel.setArg(0, _inputImage);
	_reconstructInputKernel.setArg(1, _layers.front()._columnPredictions);
	_reconstructInputKernel.setArg(2, _layers.front()._columnWeightsPrev);
	_reconstructInputKernel.setArg(3, _reconstruction);
	_reconstructInputKernel.setArg(4, reconstructionReceptiveFieldRadii);
	_reconstructInputKernel.setArg(5, sdrReceptiveFieldRadii);
	_reconstructInputKernel.setArg(6, inputSizeInv);
	_reconstructInputKernel.setArg(7, layerSize);
	_reconstructInputKernel.setArg(8, layerSizeInv);
	_reconstructInputKernel.setArg(9, inputOverSdr);

	cs.getQueue().enqueueNDRangeKernel(_reconstructInputKernel, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));

	cs.getQueue().flush();

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

	// Divide by maximum
	float maximum = 0.0f;

	for (int i = 0; i < prediction.size(); i++)
		maximum = std::max<float>(maximum, prediction[i]);

	if (maximum == 0.0f)
		return;

	float maxInv = 1.0f / maximum;

	for (int i = 0; i < prediction.size(); i++)
		prediction[i] *= maxInv;
}

void HTMRL::dutyCycleUpdate(sys::ComputeSystem &cs, float activationDutyCycleDecay, float stateDutyCycleDecay) {
	for (int l = 0; l < _layers.size(); l++)
		dutyCycleLayerUpdate(cs, _layers[l], _layerDescs[l], activationDutyCycleDecay, stateDutyCycleDecay);

	cs.getQueue().flush();
}

void HTMRL::step(sys::ComputeSystem &cs, float reward, float columnConnectionAlpha, float cellConnectionAlpha, float cellQWeightEligibilityDecay, float activationDutyCycleDecay, float stateDutyCycleDecay, float qBiasAlpha, int annealingIterations, float annealingStdDev, float annealingBreakChance, float annealingDecay, float annealingMomentum, float alpha, float gamma, float tauInv, float breakChance, float perturbationStdDev, std::mt19937 &generator) {
	stepBegin();
	
	std::vector<float> maxQInput = _input;

	// Complete input
	for (int j = 0; j < _input.size(); j++)
	if (_actionMask[j]) {
		_input[j] = _prevOutputExploratory[j];

		maxQInput[j] = _prevOutput[j];
	}

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
	std::uniform_int_distribution<int> seedDist(0, 10000);

	int seed = seedDist(generator);
	int exploreSeed = seedDist(generator);
	int learnSeed = seedDist(generator);

	activate(_input, cs, seed);

	float exploratoryQ = retrieveQ(cs);

	dutyCycleUpdate(cs, activationDutyCycleDecay, stateDutyCycleDecay);

	// Get prediction for next action
	getReconstructedPrediction(_output, cs);

	std::normal_distribution<float> perturbationDist(0.0f, perturbationStdDev);

	for (int i = 0; i < _output.size(); i++)
	if (_actionMask[i]) {
		if (perturbationDist(generator) < breakChance)
			_exploratoryOutput[i] = uniformDist(generator);
		else
			_exploratoryOutput[i] = std::min<float>(1.0f, std::max<float>(0.0f, _output[i] + perturbationDist(generator)));
	}
	else
		_exploratoryOutput[i] = _output[i];

	float newQ = reward + gamma * exploratoryQ;

	//float suboptimality = std::max<float>(0.0f, (_prevMaxQ - newQ) * tauInv);

	//float adv = newQ - suboptimality;
	float adv = newQ;

	float tdError = alpha * (adv - _prevValue);

	std::cout << tdError << " " << exploratoryQ << std::endl;

	//_prevMaxQ = maxQ;
	_prevValue = exploratoryQ;

	//if (tdError < 0.0f)
	//	activate(maxQInput, cs, seed);

	//activate(_input, cs, seed);

	updateQWeights(cs, tdError, cellQWeightEligibilityDecay, qBiasAlpha);

	//learnSpatialTemporal(cs, columnConnectionAlpha, cellConnectionAlpha, true, false, learnSeed);

	if (reward > 0.0f)
		learnSpatialTemporal(cs, columnConnectionAlpha, cellConnectionAlpha, learnSeed);
	else
		learnSpatial(cs, columnConnectionAlpha, cellConnectionAlpha, learnSeed);

	//activate(_input, cs, seed);

	//activate(_input, cs, seed);

	_prevOutput = _output;
	_prevOutputExploratory = _exploratoryOutput;
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

			color.a = std::min<float>(1.0f, std::max<float>(0.0f, _prevOutput[x + y * _inputWidth])) * (255.0f - 3.0f) + 3;

			image->setPixel(x - _inputWidth / 2 + maxWidth / 2, y - _inputHeight / 2 + maxHeight / 2, color);
		}

		images.push_back(image);
	}
	
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

		cs.getQueue().enqueueReadImage(_layers[l]._columnStates, CL_TRUE, origin, region, 0, 0, &state[0]);

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

			color.a = state[x + y * _layerDescs[l]._width] * (255.0f - 3.0f) + 3;

			image->setPixel(x - _layerDescs[l]._width / 2 + maxWidth / 2, y - _layerDescs[l]._height / 2 + maxHeight / 2, color);
		}

		images.push_back(image);
	}
}