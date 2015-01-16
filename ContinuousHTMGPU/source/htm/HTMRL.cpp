#include <htm/HTMRL.h>

#include <iostream>

using namespace htm;

void HTMRL::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, float minInitCenter, float maxInitCenter, float minInitWidth, float maxInitWidth, std::mt19937 &generator) {
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
		initLayer(cs, initPartOneKernel, initPartTwoKernel, initPartThreeKernel, prevWidth, prevHeight, prevCellsPerColumn, _layers[l], _layerDescs[l], l == _layers.size() - 1, minInitWeight, maxInitWeight, minInitCenter, maxInitCenter, minInitWidth, maxInitWidth, generator);

		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
		prevCellsPerColumn = _layerDescs[l]._cellsInColumn;
	}

	_outputWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs.back()._width, _layerDescs.back()._height, _layerDescs.back()._cellsInColumn);
	_outputWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs.back()._width, _layerDescs.back()._height, _layerDescs.back()._cellsInColumn);

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
	
	_inputErrors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

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

	_layerNodeActivateKernel = cl::Kernel(program.getProgram(), "layerNodeActivate");
	_layerNodeActivateFirstKernel = cl::Kernel(program.getProgram(), "layerNodeActivateFirst");
	_weighOutputKernel = cl::Kernel(program.getProgram(), "weighOutput");
	_layerNodeBackpropagateKernel = cl::Kernel(program.getProgram(), "layerNodeBackpropagate");
	_layerNodeBackpropagateLastKernel = cl::Kernel(program.getProgram(), "layerNodeBackpropagateLast");
	_layerNodeBackpropagateToInputKernel = cl::Kernel(program.getProgram(), "layerNodeBackpropagateToInput");
	_layerNodeWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerNodeWeightUpdate");
	_layerNodeWeightUpdateFirstKernel = cl::Kernel(program.getProgram(), "layerNodeWeightUpdateFirst");
	_layerNodeWeightUpdateLastKernel = cl::Kernel(program.getProgram(), "layerNodeWeightUpdateLast");

	_gaussianBlurXKernel = cl::Kernel(program.getProgram(), "gaussianBlurX");
	_gaussianBlurYKernel = cl::Kernel(program.getProgram(), "gaussianBlurY");

	_reconstructInputKernel = cl::Kernel(program.getProgram(), "reconstructInput");
}

void HTMRL::initLayer(sys::ComputeSystem &cs, cl::Kernel &initPartOneKernel, cl::Kernel &initPartTwoKernel, cl::Kernel &initPartThreeKernel, int inputWidth, int inputHeight, int inputCellsPerColumn, Layer &layer, const LayerDesc &layerDesc, bool isTopmost, float minInitWeight, float maxInitWeight, float minInitCenter, float maxInitCenter, float minInitWidth, float maxInitWidth, std::mt19937 &generator) {
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
	
	layer._columnStates = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._columnWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, receptiveFieldSize + 1); // + 1 for width
	layer._columnWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, receptiveFieldSize + 1); // + 1 for width

	layer._columnDutyCycles = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnDutyCyclesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._cellStates = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellStatesPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._cellPredictions = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._cellPredictionsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._cellWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, lateralConnectionsSize);
	layer._cellWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, lateralConnectionsSize);

	layer._columnPredictions = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._columnPredictionsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

	layer._nodeOutputs = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._nodeErrors = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._nodeBiases = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);
	layer._nodeBiasesPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height, layerDesc._cellsInColumn);

	layer._nodeWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, nodeConnectionsSize);
	layer._nodeWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), layerDesc._width, layerDesc._height * layerDesc._cellsInColumn, nodeConnectionsSize);

	layer._blurredLayerOutputsPing = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
	layer._blurredLayerOutputsPong = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

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
	initPartOneKernel.setArg(8, minInitCenter);
	initPartOneKernel.setArg(9, maxInitCenter);
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
	initPartThreeKernel.setArg(5, nodeConnectionsSize);
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
	_layerColumnInhibitKernel.setArg(6, layerDesc._noMatchIntensity);

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
	_layerCellActivateKernel.setArg(8, seed2);

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

void HTMRL::activate(std::vector<float> &input, sys::ComputeSystem &cs, bool predict, unsigned long seed) {
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
		
		// Blur output
		gaussianBlur(cs, _layers[l]._columnStates, _layers[l]._blurredLayerOutputsPing, _layers[l]._blurredLayerOutputsPong, _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._numBlurPasses, _layerDescs[l]._blurKernelWidthMuliplier);

		pPrevLayerOutput = &_layers[l]._blurredLayerOutputsPong;
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}

	if (predict)
	for (int l = _layers.size() - 1; l >= 0; l--) {
		if (l == _layers.size() - 1)
			predictLayerLast(cs, _layers[l], _layerDescs[l], generator);
		else
			predictLayer(cs, _layers[l + 1]._columnPredictions, _layerDescs[l + 1]._width, _layerDescs[l + 1]._height, _layers[l], _layerDescs[l], generator);
	}
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

	cs.getQueue().flush();
}

void HTMRL::learnLayerTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, float traceFactorA, float traceFactorB, std::mt19937 &generator) {
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

	Float2 traceFactors;
	traceFactors._x = traceFactorA;
	traceFactors._y = traceFactorB;

	_layerCellWeightUpdateKernel.setArg(0, layer._columnStates);
	_layerCellWeightUpdateKernel.setArg(1, layer._cellPredictionsPrev);
	_layerCellWeightUpdateKernel.setArg(2, layer._cellStates);
	_layerCellWeightUpdateKernel.setArg(3, layer._cellStatesPrev);
	_layerCellWeightUpdateKernel.setArg(4, nextLayerPrediction);
	_layerCellWeightUpdateKernel.setArg(5, layer._cellWeightsPrev);
	_layerCellWeightUpdateKernel.setArg(6, layer._columnPredictionsPrev);
	_layerCellWeightUpdateKernel.setArg(7, layer._cellWeights);
	_layerCellWeightUpdateKernel.setArg(8, layerDesc._cellsInColumn);
	_layerCellWeightUpdateKernel.setArg(9, layerSize);
	_layerCellWeightUpdateKernel.setArg(10, lateralConnectionRadii);
	_layerCellWeightUpdateKernel.setArg(11, layerSizeMinusOneInv);
	_layerCellWeightUpdateKernel.setArg(12, nextLayerSize);
	_layerCellWeightUpdateKernel.setArg(13, nextLayerSizeMinusOne);
	_layerCellWeightUpdateKernel.setArg(14, cellConnectionAlpha);
	_layerCellWeightUpdateKernel.setArg(15, cellConnectionBeta);
	_layerCellWeightUpdateKernel.setArg(16, cellConnectionTemperature);
	_layerCellWeightUpdateKernel.setArg(17, cellWeightEligibilityDecay);
	_layerCellWeightUpdateKernel.setArg(18, traceFactors);

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::learnLayerTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, float traceFactorA, float traceFactorB, std::mt19937 &generator) {
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

	Float2 traceFactors;
	traceFactors._x = traceFactorA;
	traceFactors._y = traceFactorB;

	// Lateral weight update
	_layerCellWeightUpdateLastKernel.setArg(0, layer._columnStates);
	_layerCellWeightUpdateLastKernel.setArg(1, layer._cellPredictionsPrev);
	_layerCellWeightUpdateLastKernel.setArg(2, layer._cellStates);
	_layerCellWeightUpdateLastKernel.setArg(3, layer._cellStatesPrev);
	_layerCellWeightUpdateLastKernel.setArg(4, layer._cellWeightsPrev);
	_layerCellWeightUpdateLastKernel.setArg(5, layer._columnPredictionsPrev);
	_layerCellWeightUpdateLastKernel.setArg(6, layer._cellWeights);
	_layerCellWeightUpdateLastKernel.setArg(7, layerDesc._cellsInColumn);
	_layerCellWeightUpdateLastKernel.setArg(8, layerSize);
	_layerCellWeightUpdateLastKernel.setArg(9, lateralConnectionRadii);
	_layerCellWeightUpdateLastKernel.setArg(10, cellConnectionAlpha);
	_layerCellWeightUpdateLastKernel.setArg(11, cellConnectionBeta);
	_layerCellWeightUpdateLastKernel.setArg(12, cellConnectionTemperature);
	_layerCellWeightUpdateLastKernel.setArg(13, cellWeightEligibilityDecay);
	_layerCellWeightUpdateLastKernel.setArg(14, traceFactors);

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateLastKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::learnLayerSpatialTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, float traceFactorA, float traceFactorB, std::mt19937 &generator) {
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

	Int2 layerReceptiveFieldRadius;
	layerReceptiveFieldRadius._x = layerDesc._receptiveFieldRadius;
	layerReceptiveFieldRadius._y = layerDesc._receptiveFieldRadius;

	Int2 inputSizeMinusOne;
	inputSizeMinusOne._x = prevLayerWidth - 1;
	inputSizeMinusOne._y = prevLayerHeight - 1;

	// Column weight update
	_layerColumnWeightUpdateKernel.setArg(0, prevLayerOutput);
	_layerColumnWeightUpdateKernel.setArg(1, layer._columnActivations);
	_layerColumnWeightUpdateKernel.setArg(2, layer._columnStates);
	_layerColumnWeightUpdateKernel.setArg(3, layer._columnWeightsPrev);
	_layerColumnWeightUpdateKernel.setArg(4, layer._columnDutyCyclesPrev);
	_layerColumnWeightUpdateKernel.setArg(5, layer._columnWeights);
	_layerColumnWeightUpdateKernel.setArg(6, layerSizeMinusOneInv);
	_layerColumnWeightUpdateKernel.setArg(7, inputReceptiveFieldRadius);
	_layerColumnWeightUpdateKernel.setArg(8, inputSize);
	_layerColumnWeightUpdateKernel.setArg(9, inputSizeMinusOne);
	_layerColumnWeightUpdateKernel.setArg(10, columnConnectionAlpha);
	_layerColumnWeightUpdateKernel.setArg(11, widthAlpha);
	_layerColumnWeightUpdateKernel.setArg(12, seed);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

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

	Float2 traceFactors;
	traceFactors._x = traceFactorA;
	traceFactors._y = traceFactorB;

	_layerCellWeightUpdateKernel.setArg(0, layer._columnStates);
	_layerCellWeightUpdateKernel.setArg(1, layer._cellPredictionsPrev);
	_layerCellWeightUpdateKernel.setArg(2, layer._cellStates);
	_layerCellWeightUpdateKernel.setArg(3, layer._cellStatesPrev);
	_layerCellWeightUpdateKernel.setArg(4, nextLayerPrediction);
	_layerCellWeightUpdateKernel.setArg(5, layer._cellWeightsPrev);
	_layerCellWeightUpdateKernel.setArg(6, layer._columnPredictionsPrev);
	_layerCellWeightUpdateKernel.setArg(7, layer._cellWeights);
	_layerCellWeightUpdateKernel.setArg(8, layerDesc._cellsInColumn);
	_layerCellWeightUpdateKernel.setArg(9, layerSize);
	_layerCellWeightUpdateKernel.setArg(10, lateralConnectionRadii);
	_layerCellWeightUpdateKernel.setArg(11, layerSizeMinusOneInv);
	_layerCellWeightUpdateKernel.setArg(12, nextLayerSize);
	_layerCellWeightUpdateKernel.setArg(13, nextLayerSizeMinusOne);
	_layerCellWeightUpdateKernel.setArg(14, cellConnectionAlpha);
	_layerCellWeightUpdateKernel.setArg(15, cellConnectionBeta);
	_layerCellWeightUpdateKernel.setArg(16, cellConnectionTemperature);
	_layerCellWeightUpdateKernel.setArg(17, cellWeightEligibilityDecay);
	_layerCellWeightUpdateKernel.setArg(18, traceFactors);

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
}

void HTMRL::learnLayerSpatialTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, float traceFactorA, float traceFactorB, std::mt19937 &generator) {
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

	Float2 layerSizeInv;
	layerSizeInv._x = 1.0f / layerDesc._width;
	layerSizeInv._y = 1.0f / layerDesc._height;

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
	inputSizeMinusOne._x = prevLayerWidth - 1;
	inputSizeMinusOne._y = prevLayerHeight - 1;

	// Column weight update
	_layerColumnWeightUpdateKernel.setArg(0, prevLayerOutput);
	_layerColumnWeightUpdateKernel.setArg(1, layer._columnActivations);
	_layerColumnWeightUpdateKernel.setArg(2, layer._columnStates);
	_layerColumnWeightUpdateKernel.setArg(3, layer._columnWeightsPrev);
	_layerColumnWeightUpdateKernel.setArg(4, layer._columnDutyCyclesPrev);
	_layerColumnWeightUpdateKernel.setArg(5, layer._columnWeights);
	_layerColumnWeightUpdateKernel.setArg(6, layerSizeMinusOneInv);
	_layerColumnWeightUpdateKernel.setArg(7, inputReceptiveFieldRadius);
	_layerColumnWeightUpdateKernel.setArg(8, inputSize);
	_layerColumnWeightUpdateKernel.setArg(9, inputSizeMinusOne);
	_layerColumnWeightUpdateKernel.setArg(10, columnConnectionAlpha);
	_layerColumnWeightUpdateKernel.setArg(11, widthAlpha);
	_layerColumnWeightUpdateKernel.setArg(12, seed);

	cs.getQueue().enqueueNDRangeKernel(_layerColumnWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	Int2 lateralConnectionRadii;
	lateralConnectionRadii._x = layerDesc._lateralConnectionRadius;
	lateralConnectionRadii._y = layerDesc._lateralConnectionRadius;

	Float2 traceFactors;
	traceFactors._x = traceFactorA;
	traceFactors._y = traceFactorB;

	// Lateral weight update
	_layerCellWeightUpdateLastKernel.setArg(0, layer._columnStates);
	_layerCellWeightUpdateLastKernel.setArg(1, layer._cellPredictionsPrev);
	_layerCellWeightUpdateLastKernel.setArg(2, layer._cellStates);
	_layerCellWeightUpdateLastKernel.setArg(3, layer._cellStatesPrev);
	_layerCellWeightUpdateLastKernel.setArg(4, layer._cellWeightsPrev);
	_layerCellWeightUpdateLastKernel.setArg(5, layer._columnPredictionsPrev);
	_layerCellWeightUpdateLastKernel.setArg(6, layer._cellWeights);
	_layerCellWeightUpdateLastKernel.setArg(7, layerDesc._cellsInColumn);
	_layerCellWeightUpdateLastKernel.setArg(8, layerSize);
	_layerCellWeightUpdateLastKernel.setArg(9, lateralConnectionRadii);
	_layerCellWeightUpdateLastKernel.setArg(10, cellConnectionAlpha);
	_layerCellWeightUpdateLastKernel.setArg(11, cellConnectionBeta);
	_layerCellWeightUpdateLastKernel.setArg(12, cellConnectionTemperature);
	_layerCellWeightUpdateLastKernel.setArg(13, cellWeightEligibilityDecay);
	_layerCellWeightUpdateLastKernel.setArg(14, traceFactors);

	cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateLastKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));

	cs.getQueue().flush();
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

void HTMRL::learnTemporal(sys::ComputeSystem &cs, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, float traceFactorA, float traceFactorB, unsigned long seed) {
	std::mt19937 generator(seed);

	cl::Image2D* pPrevLayerOutput = &_inputImage;
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		if (l == _layers.size() - 1)
			learnLayerTemporalLast(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], cellConnectionAlpha, cellConnectionBeta, cellConnectionTemperature, cellWeightEligibilityDecay, traceFactorA, traceFactorB, generator);
		else
			learnLayerTemporal(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l + 1]._columnPredictionsPrev, _layerDescs[l + 1]._width, _layerDescs[l + 1]._width, _layerDescs[l], cellConnectionAlpha, cellConnectionBeta, cellConnectionTemperature, cellWeightEligibilityDecay, traceFactorA, traceFactorB, generator);

		pPrevLayerOutput = &_layers[l]._columnStates;
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}
}

void HTMRL::learnSpatialTemporal(sys::ComputeSystem &cs, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, float traceFactorA, float traceFactorB, unsigned long seed) {
	std::mt19937 generator(seed);

	cl::Image2D* pPrevLayerOutput = &_inputImage;
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;
	
	for (int l = 0; l < _layers.size(); l++) {
		if (l == _layers.size() - 1)
			learnLayerSpatialTemporalLast(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layerDescs[l], columnConnectionAlpha, widthAlpha, cellConnectionAlpha, cellConnectionBeta, cellConnectionTemperature, cellWeightEligibilityDecay, traceFactorA, traceFactorB, generator);
		else
			learnLayerSpatialTemporal(cs, _layers[l], *pPrevLayerOutput, prevLayerWidth, prevLayerHeight, _layers[l + 1]._columnPredictionsPrev, _layerDescs[l + 1]._width, _layerDescs[l + 1]._width, _layerDescs[l], columnConnectionAlpha, widthAlpha, cellConnectionAlpha, cellConnectionBeta, cellConnectionTemperature, cellWeightEligibilityDecay, traceFactorA, traceFactorB, generator);
	
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

void HTMRL::layerNodeActivate(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, const LayerDesc &inputDesc, cl::Image3D &inputImage) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Int2 nodeFieldRadius;
	nodeFieldRadius._x = nodeFieldRadius._y = layerDesc._nodeFieldRadius;

	Float2 layerSizeMinusOneInv;
	layerSizeMinusOneInv._x = 1.0f / (layerDesc._width - 1);
	layerSizeMinusOneInv._y = 1.0f / (layerDesc._height - 1);

	Int2 inputSize;
	inputSize._x = inputDesc._width;
	inputSize._y = inputDesc._height;

	Int2 inputSizeMinusOne;
	inputSizeMinusOne._x = inputDesc._width - 1;
	inputSizeMinusOne._y = inputDesc._height - 1;

	_layerNodeActivateKernel.setArg(0, inputImage);
	_layerNodeActivateKernel.setArg(1, layer._cellStates);
	_layerNodeActivateKernel.setArg(2, layer._nodeBiasesPrev);
	_layerNodeActivateKernel.setArg(3, layer._nodeWeightsPrev);
	_layerNodeActivateKernel.setArg(4, layer._nodeOutputs);
	_layerNodeActivateKernel.setArg(5, layerDesc._cellsInColumn);
	_layerNodeActivateKernel.setArg(6, inputDesc._cellsInColumn);
	_layerNodeActivateKernel.setArg(7, nodeFieldRadius);
	_layerNodeActivateKernel.setArg(8, layerSizeMinusOneInv);
	_layerNodeActivateKernel.setArg(9, inputSize);
	_layerNodeActivateKernel.setArg(10, inputSizeMinusOne);

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

	Float2 layerSizeMinusOneInv;
	layerSizeMinusOneInv._x = 1.0f / (layerDesc._width - 1);
	layerSizeMinusOneInv._y = 1.0f / (layerDesc._height - 1);

	Int2 inputSize;
	inputSize._x = _inputWidth;
	inputSize._y = _inputHeight;

	Int2 inputSizeMinusOne;
	inputSizeMinusOne._x = _inputWidth - 1;
	inputSizeMinusOne._y = _inputHeight - 1;

	_layerNodeActivateFirstKernel.setArg(0, inputImage);
	_layerNodeActivateFirstKernel.setArg(1, layer._cellStates);
	_layerNodeActivateFirstKernel.setArg(2, layer._nodeBiasesPrev);
	_layerNodeActivateFirstKernel.setArg(3, layer._nodeWeightsPrev);
	_layerNodeActivateFirstKernel.setArg(4, layer._nodeOutputs);
	_layerNodeActivateFirstKernel.setArg(5, layerDesc._cellsInColumn);
	_layerNodeActivateFirstKernel.setArg(6, nodeFieldRadius);
	_layerNodeActivateFirstKernel.setArg(7, layerSizeMinusOneInv);
	_layerNodeActivateFirstKernel.setArg(8, inputSize);
	_layerNodeActivateFirstKernel.setArg(9, inputSizeMinusOne);

	cs.getQueue().enqueueNDRangeKernel(_layerNodeActivateFirstKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

float HTMRL::getQ(sys::ComputeSystem &cs) {
	_weighOutputKernel.setArg(0, _layers.back()._nodeOutputs);
	_weighOutputKernel.setArg(1, _outputWeightsPrev);
	_weighOutputKernel.setArg(2, _partialSums);
	_weighOutputKernel.setArg(3, _layerDescs.back()._cellsInColumn);

	cs.getQueue().enqueueNDRangeKernel(_weighOutputKernel, cl::NullRange, cl::NDRange(_layerDescs.back()._width, _layerDescs.back()._height));

	// Read partial sums and combine them
	float totalSum = 0.0f;// _outputBias;

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

void HTMRL::nodeActivate(std::vector<float> &input, sys::ComputeSystem &cs) {
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

	layerNodeActivateFirst(cs, _layers.front(), _layerDescs.front(), _inputImage);

	for (int l = 1; l < _layers.size(); l++)
		layerNodeActivate(cs, _layers[l], _layerDescs[l], _layerDescs[l - 1], _layers[l - 1]._nodeOutputs);
}

void HTMRL::backpropagateToInputs(sys::ComputeSystem &cs, float qError, std::vector<float> &inputs) {
	layerNodeBackpropagateLast(cs, qError);

	for (int l = _layers.size() - 2; l >= 0; l--)
		layerNodeBackpropagate(cs, _layers[l], _layers[l + 1], _layerDescs[l], _layerDescs[l + 1]);

	layerNodeBackpropagateToInput(cs);

	if (inputs.size() != _input.size())
		inputs.resize(_input.size());

	cl::size_t<3> origin;
	cl::size_t<3> region;

	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;

	region[0] = _inputWidth;
	region[1] = _inputHeight;
	region[2] = 1;

	cs.getQueue().enqueueReadImage(_inputErrors, CL_TRUE, origin, region, 0, 0, &inputs[0]);
}

void HTMRL::backpropagate(sys::ComputeSystem &cs, float qError) {
	layerNodeBackpropagateLast(cs, qError);

	for (int l = _layers.size() - 2; l >= 0; l--)
		layerNodeBackpropagate(cs, _layers[l], _layers[l + 1], _layerDescs[l], _layerDescs[l + 1]);
}

void HTMRL::nodeLearn(sys::ComputeSystem &cs, float qError, float outputAlpha, float outputBeta, float outputTemperature, float eligibilityDecay) {
	layerNodeWeightUpdateLast(cs, qError, outputAlpha, outputBeta, outputTemperature, eligibilityDecay);

	for (int l = _layers.size() - 1; l >= 1; l--)
		layerNodeWeightUpdate(cs, _layers[l], _layerDescs[l], _layerDescs[l - 1], _layers[l - 1]._nodeOutputs, qError * _layerDescs[l]._nodeAlpha, outputBeta, outputTemperature, eligibilityDecay);

	layerNodeWeightUpdateFirst(cs, _layers.front(), _layerDescs.front(), _inputImage, qError * _layerDescs.front()._nodeAlpha, outputBeta, outputTemperature, eligibilityDecay);

	_outputBias += outputAlpha * qError;
}

void HTMRL::layerNodeBackpropagate(sys::ComputeSystem &cs, Layer &layer, Layer &nextLayer, const LayerDesc &layerDesc, const LayerDesc &nextDesc) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Int2 layerSizeMinusOne;
	layerSizeMinusOne._x = layerDesc._width - 1;
	layerSizeMinusOne._y = layerDesc._height - 1;

	Float2 layerSizeMinusOneInv;
	layerSizeMinusOneInv._x = 1.0f / (layerDesc._width - 1);
	layerSizeMinusOneInv._y = 1.0f / (layerDesc._height - 1);

	Int2 nextLayerSize;
	nextLayerSize._x = nextDesc._width;
	nextLayerSize._y = nextDesc._height;

	Int2 reverseNodeFieldSize;
	reverseNodeFieldSize._x = std::ceil(static_cast<float>(nextDesc._width) / layerDesc._width * nextDesc._receptiveFieldRadius);
	reverseNodeFieldSize._y = std::ceil(static_cast<float>(nextDesc._height) / layerDesc._height * nextDesc._receptiveFieldRadius);

	Int2 nextNodeFieldSize;
	nextNodeFieldSize._x = nextNodeFieldSize._y = nextDesc._receptiveFieldRadius;

	Int2 nextLayerSizeMinusOne;
	nextLayerSizeMinusOne._x = nextDesc._width - 1;
	nextLayerSizeMinusOne._y = nextDesc._height - 1;

	Float2 nextLayerSizeMinusOneInv;
	nextLayerSizeMinusOneInv._x = 1.0f / (nextDesc._width - 1);
	nextLayerSizeMinusOneInv._y = 1.0f / (nextDesc._height - 1);

	_layerNodeBackpropagateKernel.setArg(0, nextLayer._nodeErrors);
	_layerNodeBackpropagateKernel.setArg(1, nextLayer._nodeWeightsPrev);
	_layerNodeBackpropagateKernel.setArg(2, layer._nodeOutputs);
	_layerNodeBackpropagateKernel.setArg(3, layer._cellStates);
	_layerNodeBackpropagateKernel.setArg(4, layer._nodeErrors);
	_layerNodeBackpropagateKernel.setArg(5, layerDesc._cellsInColumn);
	_layerNodeBackpropagateKernel.setArg(6, layerSizeMinusOne);
	_layerNodeBackpropagateKernel.setArg(7, layerSizeMinusOneInv);
	_layerNodeBackpropagateKernel.setArg(8, nextLayerSize);
	_layerNodeBackpropagateKernel.setArg(9, nextLayerSizeMinusOne);
	_layerNodeBackpropagateKernel.setArg(10, nextLayerSizeMinusOneInv);
	_layerNodeBackpropagateKernel.setArg(11, nextDesc._cellsInColumn);
	_layerNodeBackpropagateKernel.setArg(12, reverseNodeFieldSize);
	_layerNodeBackpropagateKernel.setArg(13, nextNodeFieldSize);

	cs.getQueue().enqueueNDRangeKernel(_layerNodeBackpropagateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::layerNodeBackpropagateLast(sys::ComputeSystem &cs, float qError) {
	_layerNodeBackpropagateLastKernel.setArg(0, _outputWeightsPrev);
	_layerNodeBackpropagateLastKernel.setArg(1, _layers.back()._nodeOutputs);
	_layerNodeBackpropagateLastKernel.setArg(2, _layers.back()._cellStates);
	_layerNodeBackpropagateLastKernel.setArg(3, _layers.back()._nodeErrors);
	_layerNodeBackpropagateLastKernel.setArg(4, _layerDescs.back()._cellsInColumn);
	_layerNodeBackpropagateLastKernel.setArg(5, qError);

	cs.getQueue().enqueueNDRangeKernel(_layerNodeBackpropagateLastKernel, cl::NullRange, cl::NDRange(_layerDescs.back()._width, _layerDescs.back()._height));
}

void HTMRL::layerNodeBackpropagateToInput(sys::ComputeSystem &cs) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Int2 layerSizeMinusOne;
	layerSizeMinusOne._x = _inputWidth - 1;
	layerSizeMinusOne._y = _inputHeight - 1;

	Float2 layerSizeMinusOneInv;
	layerSizeMinusOneInv._x = 1.0f / (_inputWidth - 1);
	layerSizeMinusOneInv._y = 1.0f / (_inputHeight - 1);

	Int2 nextLayerSize;
	nextLayerSize._x = _layerDescs.front()._width;
	nextLayerSize._y = _layerDescs.front()._height;

	Int2 reverseNodeFieldSize;
	reverseNodeFieldSize._x = std::ceil(static_cast<float>(_layerDescs.front()._width) / _inputWidth * _layerDescs.front()._receptiveFieldRadius);
	reverseNodeFieldSize._y = std::ceil(static_cast<float>(_layerDescs.front()._height) / _inputHeight * _layerDescs.front()._receptiveFieldRadius);

	Int2 nextNodeFieldSize;
	nextNodeFieldSize._x = nextNodeFieldSize._y = _layerDescs.front()._receptiveFieldRadius;

	Float2 nextOverReverseNodeFieldSize;
	nextOverReverseNodeFieldSize._x = static_cast<float>(_layerDescs.front()._receptiveFieldRadius * 2 + 1) / (reverseNodeFieldSize._x * 2 + 1);
	nextOverReverseNodeFieldSize._y = static_cast<float>(_layerDescs.front()._receptiveFieldRadius * 2 + 1) / (reverseNodeFieldSize._y * 2 + 1);

	Int2 nextLayerSizeMinusOne;
	nextLayerSizeMinusOne._x = _layerDescs.front()._width - 1;
	nextLayerSizeMinusOne._y = _layerDescs.front()._height - 1;

	Float2 nextLayerSizeMinusOneInv;
	nextLayerSizeMinusOneInv._x = 1.0f / (_layerDescs.front()._width - 1);
	nextLayerSizeMinusOneInv._y = 1.0f / (_layerDescs.front()._height - 1);

	_layerNodeBackpropagateToInputKernel.setArg(0, _layers.front()._nodeErrors);
	_layerNodeBackpropagateToInputKernel.setArg(1, _layers.front()._nodeWeightsPrev);
	_layerNodeBackpropagateToInputKernel.setArg(2, _inputImage);
	_layerNodeBackpropagateToInputKernel.setArg(3, _inputErrors);
	_layerNodeBackpropagateToInputKernel.setArg(4, layerSizeMinusOne);
	_layerNodeBackpropagateToInputKernel.setArg(5, layerSizeMinusOneInv);
	_layerNodeBackpropagateToInputKernel.setArg(6, nextLayerSize);
	_layerNodeBackpropagateToInputKernel.setArg(7, nextLayerSizeMinusOne);
	_layerNodeBackpropagateToInputKernel.setArg(8, nextLayerSizeMinusOneInv);
	_layerNodeBackpropagateToInputKernel.setArg(9, _layerDescs.front()._cellsInColumn);
	_layerNodeBackpropagateToInputKernel.setArg(10, reverseNodeFieldSize);
	_layerNodeBackpropagateToInputKernel.setArg(11, nextNodeFieldSize);

	cs.getQueue().enqueueNDRangeKernel(_layerNodeBackpropagateToInputKernel, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));
}

void HTMRL::layerNodeWeightUpdate(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, const LayerDesc &inputDesc, cl::Image3D &inputImage, float alpha, float beta, float temperature, float eligibilityDecay) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Int2 nodeFieldRadius;
	nodeFieldRadius._x = nodeFieldRadius._y = layerDesc._nodeFieldRadius;

	Float2 layerSizeMinusOneInv;
	layerSizeMinusOneInv._x = 1.0f / (layerDesc._width - 1);
	layerSizeMinusOneInv._y = 1.0f / (layerDesc._height - 1);

	Int2 inputSize;
	inputSize._x = inputDesc._width;
	inputSize._y = inputDesc._height;

	Int2 inputSizeMinusOne;
	inputSizeMinusOne._x = inputDesc._width - layerDesc._nodeFieldRadius;
	inputSizeMinusOne._y = inputDesc._height - layerDesc._nodeFieldRadius;

	_layerNodeWeightUpdateKernel.setArg(0, layer._nodeErrors);
	_layerNodeWeightUpdateKernel.setArg(1, inputImage);
	_layerNodeWeightUpdateKernel.setArg(2, layer._nodeBiasesPrev);
	_layerNodeWeightUpdateKernel.setArg(3, layer._nodeWeightsPrev);
	_layerNodeWeightUpdateKernel.setArg(4, layer._nodeBiases);
	_layerNodeWeightUpdateKernel.setArg(5, layer._nodeWeights);
	_layerNodeWeightUpdateKernel.setArg(6, layerDesc._cellsInColumn);
	_layerNodeWeightUpdateKernel.setArg(7, inputDesc._cellsInColumn);
	_layerNodeWeightUpdateKernel.setArg(8, nodeFieldRadius);
	_layerNodeWeightUpdateKernel.setArg(9, layerSizeMinusOneInv);
	_layerNodeWeightUpdateKernel.setArg(10, inputSize);
	_layerNodeWeightUpdateKernel.setArg(11, inputSizeMinusOne);
	_layerNodeWeightUpdateKernel.setArg(12, alpha);
	_layerNodeWeightUpdateKernel.setArg(13, beta);
	_layerNodeWeightUpdateKernel.setArg(14, temperature);
	_layerNodeWeightUpdateKernel.setArg(15, eligibilityDecay);

	cs.getQueue().enqueueNDRangeKernel(_layerNodeWeightUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::layerNodeWeightUpdateFirst(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, cl::Image2D &inputImage, float alpha, float beta, float temperature, float eligibilityDecay) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	Int2 nodeFieldRadius;
	nodeFieldRadius._x = nodeFieldRadius._y = layerDesc._nodeFieldRadius;

	Float2 layerSizeMinusOneInv;
	layerSizeMinusOneInv._x = 1.0f / (layerDesc._width - 1);
	layerSizeMinusOneInv._y = 1.0f / (layerDesc._height - 1);

	Int2 inputSize;
	inputSize._x = _inputWidth;
	inputSize._y = _inputHeight;

	Int2 inputSizeMinusOne;
	inputSizeMinusOne._x = _inputWidth - layerDesc._nodeFieldRadius;
	inputSizeMinusOne._y = _inputHeight - layerDesc._nodeFieldRadius;

	_layerNodeWeightUpdateFirstKernel.setArg(0, layer._nodeErrors);
	_layerNodeWeightUpdateFirstKernel.setArg(1, inputImage);
	_layerNodeWeightUpdateFirstKernel.setArg(2, layer._nodeBiasesPrev);
	_layerNodeWeightUpdateFirstKernel.setArg(3, layer._nodeWeightsPrev);
	_layerNodeWeightUpdateFirstKernel.setArg(4, layer._nodeBiases);
	_layerNodeWeightUpdateFirstKernel.setArg(5, layer._nodeWeights);
	_layerNodeWeightUpdateFirstKernel.setArg(6, layerDesc._cellsInColumn);
	_layerNodeWeightUpdateFirstKernel.setArg(7, nodeFieldRadius);
	_layerNodeWeightUpdateFirstKernel.setArg(8, layerSizeMinusOneInv);
	_layerNodeWeightUpdateFirstKernel.setArg(9, inputSize);
	_layerNodeWeightUpdateFirstKernel.setArg(10, inputSizeMinusOne);
	_layerNodeWeightUpdateFirstKernel.setArg(11, alpha);
	_layerNodeWeightUpdateFirstKernel.setArg(12, beta);
	_layerNodeWeightUpdateFirstKernel.setArg(13, temperature);
	_layerNodeWeightUpdateFirstKernel.setArg(14, eligibilityDecay);

	cs.getQueue().enqueueNDRangeKernel(_layerNodeWeightUpdateFirstKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
}

void HTMRL::layerNodeWeightUpdateLast(sys::ComputeSystem &cs, float qError, float alpha, float beta, float temperature, float eligibilityDecay) {
	_layerNodeWeightUpdateLastKernel.setArg(0, _layers.back()._nodeOutputs);
	_layerNodeWeightUpdateLastKernel.setArg(1, _outputWeightsPrev);
	_layerNodeWeightUpdateLastKernel.setArg(2, _outputWeights);
	_layerNodeWeightUpdateLastKernel.setArg(3, _layerDescs.back()._cellsInColumn);
	_layerNodeWeightUpdateLastKernel.setArg(4, alpha * qError);
	_layerNodeWeightUpdateLastKernel.setArg(5, beta);
	_layerNodeWeightUpdateLastKernel.setArg(6, temperature);
	_layerNodeWeightUpdateLastKernel.setArg(7, eligibilityDecay);

	cs.getQueue().enqueueNDRangeKernel(_layerNodeWeightUpdateLastKernel, cl::NullRange, cl::NDRange(_layerDescs.back()._width, _layerDescs.back()._height));
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

void HTMRL::step(sys::ComputeSystem &cs, float reward, float outputAlpha, float outputBeta, float outputTemperature, float nodeEligibilityDecay, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, float activationDutyCycleDecay, float stateDutyCycleDecay, float reconstructionAlpha, float qBiasAlpha, int deriveMaxQIterations, float deriveMaxQAlpha, float deriveMaxQError, float deriveQMutationStdDev, float deriveMaxQMutationDecay, float deriveMaxQMomentum, float alpha, float gamma, float tauInv, float breakChance, float perturbationStdDev, float maxTdError, std::mt19937 &generator) {
	std::uniform_int_distribution<int> seedDist(0, 10000);

	unsigned long seed = seedDist(generator);

	stepBegin();

	for (int i = 0; i < _input.size(); i++)
	if (_inputTypes[i] == _action)
		_input[i] = std::min<float>(1.0f, std::max<float>(0.0f, _prevOutputExploratory[i]));
	else if (_inputTypes[i] == _unused)
		_input[i] = 0.0f;

	std::vector<float> learnInputExploratory(_input.size());

	for (int i = 0; i < _input.size(); i++)
	if (_inputTypes[i] == _action)
		learnInputExploratory[i] = std::min<float>(1.0f, std::max<float>(0.0f, _prevOutputExploratory[i]));
	else if (_inputTypes[i] == _unused)
		learnInputExploratory[i] = 0.0f;
	else
		learnInputExploratory[i] = _input[i];

	std::vector<float> learnInputMaxQ(_input.size());

	for (int i = 0; i < _input.size(); i++)
	if (_inputTypes[i] == _action)
		learnInputMaxQ[i] = std::min<float>(1.0f, std::max<float>(0.0f, _prevOutput[i]));
	else if (_inputTypes[i] == _unused)
		learnInputMaxQ[i] = 0.0f;
	else
		learnInputMaxQ[i] = _input[i];

	activate(_input, cs, true, seed);
	nodeActivate(_input, cs);

	dutyCycleUpdate(cs, activationDutyCycleDecay, stateDutyCycleDecay);
	
	getReconstructedPrediction(_output, cs);

	// Exploratory action
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, perturbationStdDev);

	for (int i = 0; i < _output.size(); i++)
	if (_inputTypes[i] == _action) {
		if (dist01(generator) < breakChance)
			_exploratoryOutput[i] = dist01(generator);
		else
			_exploratoryOutput[i] = std::min<float>(1.0f, std::max<float>(0.0f, std::min<float>(1.0f, std::max<float>(0.0f, _output[i])) + pertDist(generator)));
	}

	float maxQ = getQ(cs);

	float newQ = reward + gamma * maxQ;

	float tdError = alpha * (newQ - _prevValue);

	learnSpatial(cs, columnConnectionAlpha, widthAlpha, seed + 1);

	if (tdError > 0.0f) {
		activate(learnInputExploratory, cs, false, seed);
	
		learnTemporal(cs, cellConnectionAlpha * tdError, cellConnectionBeta, cellConnectionTemperature, cellWeightEligibilityDecay, 1.0f, 0.0f, seed + 2);

		std::cout << "Exp" << std::endl;
	}
	else {
		activate(learnInputMaxQ, cs, false, seed);
		
		learnTemporal(cs, cellConnectionAlpha * -tdError, cellConnectionBeta, cellConnectionTemperature, cellWeightEligibilityDecay, 0.0f, 1.0f, seed + 2);

		std::cout << "Max" << std::endl;
	}

	activate(_input, cs, true, seed);
	nodeActivate(_input, cs);

	_prevValue = getQ(cs);

	std::cout << "Q: " << newQ << " Err: " << tdError << std::endl;

	backpropagate(cs, 1.0f);

	nodeLearn(cs, tdError, outputAlpha, outputBeta, outputTemperature, nodeEligibilityDecay);

	_prevOutput = _output;
	_prevOutputExploratory = _exploratoryOutput;

	/*stepBegin();

	std::uniform_int_distribution<int> seedDist(0, 10000);

	unsigned long seed = seedDist(generator);

	for (int i = 0; i < _input.size(); i++)
	if (_inputTypes[i] == _action)
		_input[i] = std::min<float>(1.0f, std::max<float>(0.0f, _output[i]));
	else if (_inputTypes[i] == _unused)
		_input[i] = 0.0f;

	for (int j = 0; j < _output.size(); j++)
	if (_inputTypes[j] == _action)
		_output[j] = std::min<float>(1.0f, std::max<float>(0.0f, _output[j]));
	else if (_inputTypes[j] == _unused)
		_output[j] = 0.0f;
	else
		_output[j] = _input[j];

	activate(_output, cs, false, seed);
	nodeActivate(_output, cs);

	std::cout << "Start" << std::endl;

	std::normal_distribution<float> deriveQMutationDist(0.0f, deriveQMutationStdDev);

	float maxQ = getQ(cs);

	std::vector<float> suggestedInputs;
	backpropagateToInputs(cs, deriveMaxQError, suggestedInputs);

	float mutationAmount = 1.0f;

	std::vector<float> prevDeltas(_output.size(), 0.0f);

	for (int i = 0; i < deriveMaxQIterations; i++) {
		std::vector<float> tOutput = _output;

		for (int j = 0; j < _output.size(); j++)
		if (_inputTypes[j] == _action)
			tOutput[j] = std::min<float>(1.0f, std::max<float>(0.0f, _output[j] + deriveMaxQAlpha * suggestedInputs[j] + mutationAmount * deriveQMutationDist(generator) + deriveMaxQMomentum * prevDeltas[j]));

		activate(tOutput, cs, false, seed);
		nodeActivate(tOutput, cs);

		float tQ = getQ(cs);

		if (tQ > maxQ) {
			for (int j = 0; j < _output.size(); j++)
			if (_inputTypes[j] == _action)
				prevDeltas[j] = tOutput[j] - _output[j];

			_output = tOutput;

			std::cout << tQ << " " << maxQ << std::endl;

			maxQ = tQ;

			backpropagateToInputs(cs, deriveMaxQError, suggestedInputs);
		}

		mutationAmount *= deriveMaxQMutationDecay;
	}

	std::cout << "End" << std::endl;

	//activate(_output, cs, seed);
	//nodeActivate(_output, cs);

	// Exploratory action
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, perturbationStdDev);

	for (int i = 0; i < _output.size(); i++)
	if (_inputTypes[i] == _action) {
		if (dist01(generator) < breakChance)
			_exploratoryOutput[i] = dist01(generator);
		else
			_exploratoryOutput[i] = std::min<float>(1.0f, std::max<float>(0.0f, _output[i] + pertDist(generator)));

			_input[i] = _exploratoryOutput[i];
	}
	else
		_exploratoryOutput[i] = _input[i];

	activate(_exploratoryOutput, cs, true, seed);
	nodeActivate(_exploratoryOutput, cs);

	dutyCycleUpdate(cs, activationDutyCycleDecay, stateDutyCycleDecay);

	learnSpatialTemporal(cs, columnConnectionAlpha, widthAlpha, cellConnectionAlpha, cellConnectionBeta, cellConnectionTemperature, 1.0f, seed + 1);

	std::vector<float> recon;
	getReconstructedPrevPrediction(recon, cs);
	learnReconstruction(cs, reconstructionAlpha);

	getReconstructedPrediction(_output, cs);

	float value = getQ(cs);

	float newQ = reward + gamma * value;

	float tdError = alpha * (newQ - _prevValue);

	float q = _prevValue + tdError;

	std::cout << "Q: " << q << " Err: " << tdError << std::endl;

	backpropagate(cs, 1.0f);

	if (tdError > maxTdError)
		tdError = maxTdError;
	else if (tdError < -maxTdError)
		tdError = -maxTdError;

	nodeLearn(cs, tdError, outputAlpha, outputBeta, outputTemperature, nodeEligibilityDecay);

	_prevOutput = _output;
	_prevOutputExploratory = _exploratoryOutput;
	_prevValue = value;*/
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
}