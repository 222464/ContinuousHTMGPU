#include <htm/HTMRL.h>

#include <iostream>

using namespace htm;

void HTMRL::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, float minInitWidth, float maxInitWidth, float minEncoderInitCenter, float maxEncoderInitCenter, float minEncoderInitWidth, float maxEncoderInitWidth, float minEncoderInitWeight, float maxEncoderInitWeight, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_layerDescs = layerDescs;
	
	_layers.resize(_layerDescs.size());

	_inputTypes = inputTypes;

	_qInputIndices.clear();

	for (int i = 0; i < _inputTypes.size(); i++)
	if (_inputTypes[i] == _q)
		_qInputIndices.push_back(i);

	_qEncoder.create(_qInputIndices.size(), 1, minEncoderInitCenter, maxEncoderInitCenter, minEncoderInitWidth, maxEncoderInitWidth, minEncoderInitWeight, maxEncoderInitWeight, generator);

	std::uniform_real_distribution<float> weightDist(minInitWeight, maxInitWeight);
	std::uniform_real_distribution<float> actionDist(0.0f, 1.0f);

	_qBias = weightDist(generator);
	_prevMaxQ = 0.0f;
	_prevValue = 0.0f;
	_prevPrevValue = 0.0f;
	_prevQ = 0.0f;

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

	for (int l = 0; l < _layers.size(); l++) {
		initLayer(cs, initPartOneKernel, initPartTwoKernel, prevWidth, prevHeight, _layers[l], _layerDescs[l], l == _layers.size() - 1, minInitWeight, maxInitWeight, minInitWidth, maxInitWidth, generator);

		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
	}

	_reconstructionReceptiveRadius = std::ceil(static_cast<float>(_layerDescs.front()._width) / static_cast<float>(_inputWidth) * static_cast<float>(_layerDescs.front()._receptiveFieldRadius));

	int reconstructionNumWeights = std::pow(_reconstructionReceptiveRadius * 2 + 1, 2) + 1; // + 1 for bias

	_reconstructionWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight, reconstructionNumWeights);
	_reconstructionWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight, reconstructionNumWeights);

	_reconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

	cl::Kernel reconstructionInit = cl::Kernel(program.getProgram(), "reconstructionInit");

	struct Uint2 {
		unsigned int _x, _y;
	};

	std::uniform_int_distribution<int> uniformDist(0, 10000);

	Uint2 seed;
	seed._x = uniformDist(generator);
	seed._y = uniformDist(generator);

	reconstructionInit.setArg(0, _reconstructionWeights);
	reconstructionInit.setArg(1, reconstructionNumWeights);
	reconstructionInit.setArg(2, seed);
	reconstructionInit.setArg(3, minInitWeight);
	reconstructionInit.setArg(4, maxInitWeight);

	cs.getQueue().enqueueNDRangeKernel(reconstructionInit, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));

	_reconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = reconstructionNumWeights;

		cs.getQueue().enqueueCopyImage(_reconstructionWeights, _reconstructionWeightsPrev, origin, origin, region);
	}

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

	_reconstructInputKernel = cl::Kernel(program.getProgram(), "reconstructInput");
	_learnReconstructionKernel = cl::Kernel(program.getProgram(), "learnReconstruction");
}

void HTMRL::initLayer(sys::ComputeSystem &cs, cl::Kernel &initPartOneKernel, cl::Kernel &initPartTwoKernel, int inputWidth, int inputHeight, Layer &layer, const LayerDesc &layerDesc, bool isTopmost, float minInitWeight, float maxInitWeight, float minInitWidth, float maxInitWidth, std::mt19937 &generator) {
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
	layer._cellStatesPrevPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._cellPredictions = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellPredictionsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._cellWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, lateralConnectionsSize);
	layer._cellWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, lateralConnectionsSize);

	layer._columnPredictions = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnPredictionsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

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
		cs.getQueue().enqueueCopyImage(layer._cellStates, layer._cellStatesPrevPrev, origin, origin, region);
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
		std::swap(_layers[l]._cellStates, _layers[l]._cellStatesPrev);
		std::swap(_layers[l]._cellPredictions, _layers[l]._cellPredictionsPrev);
		std::swap(_layers[l]._cellWeights, _layers[l]._cellWeightsPrev);
	}

	std::swap(_reconstructionWeights, _reconstructionWeightsPrev);
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

void HTMRL::getReconstruction(std::vector<float> &prediction, sys::ComputeSystem &cs) {
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

	_reconstructInputKernel.setArg(0, _reconstructionWeightsPrev);
	_reconstructInputKernel.setArg(1, _layers.front()._columnStates);
	_reconstructInputKernel.setArg(2, _reconstruction);
	_reconstructInputKernel.setArg(3, reconstructionReceptiveFieldRadii);
	_reconstructInputKernel.setArg(4, inputSizeInv);
	_reconstructInputKernel.setArg(5, layerSize);
	_reconstructInputKernel.setArg(6, layerSizeInv);
	_reconstructInputKernel.setArg(7, inputOverSdr);

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

	_reconstructInputKernel.setArg(0, _reconstructionWeightsPrev);
	_reconstructInputKernel.setArg(1, _layers.front()._columnPredictions);
	_reconstructInputKernel.setArg(2, _reconstruction);
	_reconstructInputKernel.setArg(3, reconstructionReceptiveFieldRadii);
	_reconstructInputKernel.setArg(4, inputSizeInv);
	_reconstructInputKernel.setArg(5, layerSize);
	_reconstructInputKernel.setArg(6, layerSizeInv);
	_reconstructInputKernel.setArg(7, inputOverSdr);

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
}

void HTMRL::learnReconstruction(sys::ComputeSystem &cs, float reconstructionAlpha) {
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

	_learnReconstructionKernel.setArg(0, _inputImage);
	_learnReconstructionKernel.setArg(1, _reconstruction);
	_learnReconstructionKernel.setArg(2, _reconstructionWeightsPrev);
	_learnReconstructionKernel.setArg(3, _layers.front()._columnStates);
	_learnReconstructionKernel.setArg(4, _reconstructionWeights);
	_learnReconstructionKernel.setArg(5, reconstructionReceptiveFieldRadii);
	_learnReconstructionKernel.setArg(6, inputSizeInv);
	_learnReconstructionKernel.setArg(7, layerSize);
	_learnReconstructionKernel.setArg(8, layerSizeInv);
	_learnReconstructionKernel.setArg(9, inputOverSdr);
	_learnReconstructionKernel.setArg(10, reconstructionAlpha);

	cs.getQueue().enqueueNDRangeKernel(_learnReconstructionKernel, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));

	cs.getQueue().flush();
}

void HTMRL::dutyCycleUpdate(sys::ComputeSystem &cs, float activationDutyCycleDecay, float stateDutyCycleDecay) {
	for (int l = 0; l < _layers.size(); l++)
		dutyCycleLayerUpdate(cs, _layers[l], _layerDescs[l], activationDutyCycleDecay, stateDutyCycleDecay);

	cs.getQueue().flush();
}

float HTMRL::retrieveQ(const std::vector<float> &sdr) {
	std::vector<float> encoderInput(_qInputIndices.size());

	for (int i = 0; i < _qInputIndices.size(); i++)
		encoderInput[i] = sdr[_qInputIndices[i]];

	std::vector<float> result;

	_qEncoder.decode(encoderInput, result);

	return result[0];
}

void HTMRL::setQ(std::vector<float> &sdr, float q, float localActivity, float outputIntensity, float dutyCycleDecay, float boostThreshold, float boostIntensity) {
	std::vector<float> result;

	_qEncoder.encode(std::vector<float>(1, q), result, localActivity, outputIntensity, dutyCycleDecay, boostThreshold, boostIntensity);

	for (int i = 0; i < _qInputIndices.size(); i++)
		sdr[_qInputIndices[i]] = result[i];
}

void HTMRL::learnQEncoder(float q, float prevQ, float centerAlpha, float widthAlpha, float widthScalar, float minWidth, float reconAlpha) {
	_qEncoder.learn(std::vector<float>(1, q), std::vector<float>(1, prevQ), centerAlpha, widthAlpha, widthScalar, minWidth, reconAlpha);
}

void HTMRL::step(sys::ComputeSystem &cs, float reward, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellWeightEligibilityDecay, float cellQWeightEligibilityDecay, float activationDutyCycleDecay, float stateDutyCycleDecay, float reconstructionAlpha, float qBiasAlpha, float encoderLocalActivity, float encoderOutputIntensity, float encoderDutyCycleDecay, float encoderBoostThreshold, float encoderBoostIntensity, float encoderCenterAlpha, float encoderWidthAlpha, float encoderWidthScalar, float encoderMinWidth, float encoderReconAlpha, float alpha, float gamma, float tauInv, float breakChance, float perturbationStdDev, std::mt19937 &generator) {
	stepBegin();
	
	// Complete input
	for (int j = 0; j < _input.size(); j++)
	if (_inputTypes[j] != _state)
		_input[j] = _prevOutputExploratory[j];

	setQ(_input, _prevQ, encoderLocalActivity, encoderOutputIntensity, encoderDutyCycleDecay, encoderBoostThreshold, encoderBoostIntensity);

	float reconstruction = retrieveQ(_input);

	learnQEncoder(_prevQ, reconstruction, encoderCenterAlpha, encoderWidthAlpha, encoderWidthScalar, encoderMinWidth, encoderReconAlpha);

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
	std::uniform_int_distribution<int> seedDist(0, 10000);

	int seed = seedDist(generator);
	int exploreSeed = seedDist(generator);
	int learnSeed = seedDist(generator);

	activate(_input, cs, seed);

	dutyCycleUpdate(cs, activationDutyCycleDecay, stateDutyCycleDecay);

	std::vector<float> stateRecon;

	getReconstruction(stateRecon, cs);

	learnReconstruction(cs, reconstructionAlpha);

	// Get prediction for next action
	getReconstructedPrediction(_output, cs);

	float exploratoryQ = retrieveQ(_output);

	float newQ = reward + gamma * exploratoryQ;

	float tdError = alpha * (newQ - _prevValue);

	std::cout << tdError << " " << exploratoryQ << std::endl;

	float q = _prevValue + tdError;

	std::normal_distribution<float> perturbationDist(0.0f, perturbationStdDev);

	for (int i = 0; i < _output.size(); i++)
	if (_inputTypes[i] == _action) {
		if (uniformDist(generator) < breakChance)
			_exploratoryOutput[i] = uniformDist(generator);
		else
			_exploratoryOutput[i] = std::min<float>(1.0f, std::max<float>(0.0f, _output[i] + perturbationDist(generator)));
	}
	else
		_exploratoryOutput[i] = _output[i];

	learnSpatialTemporal(cs, columnConnectionAlpha, widthAlpha, tdError > 0.0f ? cellConnectionAlpha : 0.0f, cellWeightEligibilityDecay, learnSeed);

	_prevPrevValue = _prevValue;
	_prevValue = exploratoryQ;
	_prevOutput = _output;
	_prevOutputExploratory = _exploratoryOutput;
	_prevQ = q;
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

				image->setPixel(x - _layerDescs[l]._width / 2 + maxWidth / 2, y - _layerDescs[l]._height / 2 + maxHeight / 2, color);
			}

			images.push_back(image);
		}
	}*/
}