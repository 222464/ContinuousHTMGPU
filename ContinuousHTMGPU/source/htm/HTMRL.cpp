#include <htm/HTMRL.h>

#include <iostream>

using namespace htm;

void HTMRL::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<bool> &actionMask, float minInitWeight, float maxInitWeight, float minInitWidth, float maxInitWidth, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;
	
	_layerDescs = layerDescs;
	
	_layers.resize(_layerDescs.size());

	_actionMask = actionMask;

	std::uniform_real_distribution<float> weightDist(minInitWeight, maxInitWeight);
	std::uniform_real_distribution<float> actionDist(-1.0f, 1.0f);

	_qBias = weightDist(generator);
	_prevMaxQ = 0.0f;
	_prevValue = 0.0f;

	cl::Kernel initPartOneKernel = cl::Kernel(program.getProgram(), "initializePartOne");
	cl::Kernel initPartTwoKernel = cl::Kernel(program.getProgram(), "initializePartTwo");

	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;

	int maxWidth = 0;
	int maxHeight = 0;

	_input.clear();
	_input.assign(_inputWidth * _inputHeight, 0.0f);

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

	std::uniform_int_distribution<int> uniformDist(0, 10000);

	struct Uint2 {
		unsigned int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	for (int l = 0; l < _layers.size(); l++) {
		maxWidth = std::max<int>(maxWidth, _layerDescs[l]._width);
		maxHeight = std::max<int>(maxHeight, _layerDescs[l]._height);

		int receptiveFieldSize = std::pow(_layerDescs[l]._receptiveFieldRadius * 2 + 1, 2);
		int lateralConnectionsSize;
		
		// If not the last layer, add weights for additional context from next layer
		if (l == _layers.size() - 1)
			lateralConnectionsSize = std::pow(_layerDescs[l]._lateralConnectionRadius * 2 + 1, 2) * (_layerDescs[l]._cellsInColumn + 1) + 1; // + 1 for bias
		else
			lateralConnectionsSize = std::pow(_layerDescs[l]._lateralConnectionRadius * 2 + 1, 2) * _layerDescs[l]._cellsInColumn + 1; // + 1 for bias

		_layers[l]._columnActivations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		
		_layers[l]._columnDutyCycles = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._columnDutyCyclesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._columnAttentions = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._columnAttentionsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._columnStates = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._columnStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._columnWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, receptiveFieldSize);
		_layers[l]._columnWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, receptiveFieldSize);

		_layers[l]._cellStates = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._cellsInColumn);
		_layers[l]._cellStatesPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._cellsInColumn);

		_layers[l]._cellPredictions = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._cellsInColumn);
		_layers[l]._cellPredictionsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._cellsInColumn);

		_layers[l]._cellWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height * _layerDescs[l]._cellsInColumn, lateralConnectionsSize);
		_layers[l]._cellWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height * _layerDescs[l]._cellsInColumn, lateralConnectionsSize);

		_layers[l]._columnPredictions = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._columnPredictionsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._cellQWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._cellsInColumn);
		_layers[l]._cellQWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._cellsInColumn);

		_layers[l]._partialQSums = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		Uint2 seed1;
		seed1._x = uniformDist(generator);
		seed1._y = uniformDist(generator);

		initPartOneKernel.setArg(0, _layers[l]._columnActivations);
		initPartOneKernel.setArg(1, _layers[l]._columnDutyCycles);
		initPartOneKernel.setArg(2, _layers[l]._columnStates);
		initPartOneKernel.setArg(3, _layers[l]._columnWeights);
		initPartOneKernel.setArg(4, _layerDescs[l]._cellsInColumn);
		initPartOneKernel.setArg(5, receptiveFieldSize);
		initPartOneKernel.setArg(6, lateralConnectionsSize);
		initPartOneKernel.setArg(7, seed1);
		initPartOneKernel.setArg(8, minInitWeight);
		initPartOneKernel.setArg(9, maxInitWeight);
		initPartOneKernel.setArg(10, minInitWidth);
		initPartOneKernel.setArg(11, maxInitWidth);

		cs.getQueue().enqueueNDRangeKernel(initPartOneKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		Uint2 seed2;
		seed2._x = uniformDist(generator);
		seed2._y = uniformDist(generator);

		initPartTwoKernel.setArg(0, _layers[l]._cellStates);
		initPartTwoKernel.setArg(1, _layers[l]._cellWeights);
		initPartTwoKernel.setArg(2, _layers[l]._cellPredictions);
		initPartTwoKernel.setArg(3, _layers[l]._cellQWeights);
		initPartTwoKernel.setArg(4, _layerDescs[l]._cellsInColumn);
		initPartTwoKernel.setArg(5, receptiveFieldSize);
		initPartTwoKernel.setArg(6, lateralConnectionsSize);
		initPartTwoKernel.setArg(7, seed2);
		initPartTwoKernel.setArg(8, minInitWeight);
		initPartTwoKernel.setArg(9, maxInitWeight);
		initPartTwoKernel.setArg(10, minInitWidth);
		initPartTwoKernel.setArg(11, maxInitWidth);

		cs.getQueue().enqueueNDRangeKernel(initPartTwoKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		{
			cl::size_t<3> origin;
			cl::size_t<3> region;

			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = 1;

			cs.getQueue().enqueueCopyImage(_layers[l]._columnStates, _layers[l]._columnStatesPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			cl::size_t<3> region;

			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = 1;

			cs.getQueue().enqueueCopyImage(_layers[l]._columnDutyCycles, _layers[l]._columnDutyCyclesPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			cl::size_t<3> region;

			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = 1;

			cs.getQueue().enqueueCopyImage(_layers[l]._columnAttentions, _layers[l]._columnAttentionsPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			cl::size_t<3> region;

			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = receptiveFieldSize;

			cs.getQueue().enqueueCopyImage(_layers[l]._columnWeights, _layers[l]._columnWeightsPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			cl::size_t<3> region;

			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = _layerDescs[l]._cellsInColumn;

			cs.getQueue().enqueueCopyImage(_layers[l]._cellStates, _layers[l]._cellStatesPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			cl::size_t<3> region;

			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height * _layerDescs[l]._cellsInColumn;
			region[2] = lateralConnectionsSize;

			cs.getQueue().enqueueCopyImage(_layers[l]._cellWeights, _layers[l]._cellWeightsPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			cl::size_t<3> region;

			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = _layerDescs[l]._cellsInColumn;

			cs.getQueue().enqueueCopyImage(_layers[l]._cellPredictions, _layers[l]._cellPredictionsPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			cl::size_t<3> region;

			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = _layerDescs[l]._cellsInColumn;

			cs.getQueue().enqueueCopyImage(_layers[l]._cellQWeights, _layers[l]._cellQWeightsPrev, origin, origin, region);
		}

		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
	}

	_maxBufferPing = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), maxWidth, maxHeight);
	_maxBufferPong = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), maxWidth, maxHeight);

	_reconstructionReceptiveRadius = std::ceil(static_cast<float>(_layerDescs.front()._width) / static_cast<float>(_inputWidth) * static_cast<float>(_layerDescs.front()._receptiveFieldRadius));
	int reconstructionNumWeights = std::pow(_reconstructionReceptiveRadius * 2 + 1, 2) + 1; // + 1 for bias

	_reconstructionWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight, reconstructionNumWeights);
	_reconstructionWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight, reconstructionNumWeights);

	_reconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

	cl::Kernel reconstructionInit = cl::Kernel(program.getProgram(), "reconstructionInit");

	Uint2 seed;
	seed._x = uniformDist(generator);
	seed._y = uniformDist(generator);

	reconstructionInit.setArg(0, _reconstructionWeights);
	reconstructionInit.setArg(1, reconstructionNumWeights);
	reconstructionInit.setArg(2, seed);
	reconstructionInit.setArg(3, minInitWeight);
	reconstructionInit.setArg(4, maxInitWeight);

	cs.getQueue().enqueueNDRangeKernel(reconstructionInit, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));

	{
		cl::size_t<3> origin;
		cl::size_t<3> region;

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

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
	_layerRetrieveQKernel = cl::Kernel(program.getProgram(), "layerRetrieveQ");
	_layerUpdateQWeightsKernel = cl::Kernel(program.getProgram(), "layerUpdateQWeights");
	_reconstructInputKernel = cl::Kernel(program.getProgram(), "reconstructInput");
	_updateReconstructionKernel = cl::Kernel(program.getProgram(), "updateReconstruction");
}

void HTMRL::stepBegin() {
	for (int l = 0; l < _layers.size(); l++) {
		std::swap(_layers[l]._columnStates, _layers[l]._columnStatesPrev);
		std::swap(_layers[l]._columnDutyCycles, _layers[l]._columnDutyCyclesPrev);
		std::swap(_layers[l]._columnAttentions, _layers[l]._columnAttentionsPrev);
		std::swap(_layers[l]._columnWeights, _layers[l]._columnWeightsPrev);
		std::swap(_layers[l]._cellStates, _layers[l]._cellStatesPrev);
		std::swap(_layers[l]._cellPredictions, _layers[l]._cellPredictionsPrev);
		std::swap(_layers[l]._cellWeights, _layers[l]._cellWeightsPrev);
		std::swap(_layers[l]._columnPredictions, _layers[l]._columnPredictionsPrev);
		std::swap(_layers[l]._cellQWeights, _layers[l]._cellQWeightsPrev);
	}

	std::swap(_reconstructionWeights, _reconstructionWeightsPrev);
}

void HTMRL::activate(std::vector<float> &input, sys::ComputeSystem &cs, unsigned long seed) {
	std::mt19937 generator(seed);

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

	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;
	cl::Image2D* pPrevColumnStates = &_inputImage;

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

	for (int l = 0; l < _layers.size(); l++) {
		Uint2 seed;
		seed._x = uniformDist(generator);
		seed._y = uniformDist(generator);

		Int2 inputSize;
		inputSize._x = prevLayerWidth;
		inputSize._y = prevLayerHeight;

		Int2 layerSize;
		layerSize._x = _layerDescs[l]._width;
		layerSize._y = _layerDescs[l]._height;

		Float2 inputSizeInv;
		inputSizeInv._x = 1.0f / prevLayerWidth;
		inputSizeInv._y = 1.0f / prevLayerHeight;

		Float2 layerSizeInv;
		layerSizeInv._x = 1.0f / _layerDescs[l]._width;
		layerSizeInv._y = 1.0f / _layerDescs[l]._height;

		Int2 inputReceptiveFieldRadius;
		inputReceptiveFieldRadius._x = _layerDescs[l]._receptiveFieldRadius;
		inputReceptiveFieldRadius._y = _layerDescs[l]._receptiveFieldRadius;

		Float2 inputReceptiveFieldStep;
		inputReceptiveFieldStep._x = inputSizeInv._x * _layerDescs[l]._receptiveFieldRadius;
		inputReceptiveFieldStep._y = inputSizeInv._y * _layerDescs[l]._receptiveFieldRadius;

		// Activation
		_layerColumnActivateKernel.setArg(0, *pPrevColumnStates);
		_layerColumnActivateKernel.setArg(1, _layers[l]._columnWeightsPrev);
		_layerColumnActivateKernel.setArg(2, _layers[l]._columnActivations);
		_layerColumnActivateKernel.setArg(3, layerSizeInv);
		_layerColumnActivateKernel.setArg(4, inputReceptiveFieldRadius);
		_layerColumnActivateKernel.setArg(5, inputReceptiveFieldStep);
		_layerColumnActivateKernel.setArg(6, inputSize);

		cs.getQueue().enqueueNDRangeKernel(_layerColumnActivateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		Int2 layerInhibitionRadius;
		layerInhibitionRadius._x = _layerDescs[l]._inhibitionRadius;
		layerInhibitionRadius._y = _layerDescs[l]._inhibitionRadius;

		Float2 layerInhibitionStep;
		layerInhibitionStep._x = layerSizeInv._x * _layerDescs[l]._inhibitionRadius;
		layerInhibitionStep._y = layerSizeInv._y * _layerDescs[l]._inhibitionRadius;

		// Inhibition
		_layerColumnInhibitKernel.setArg(0, _layers[l]._columnActivations);
		_layerColumnInhibitKernel.setArg(1, _layers[l]._columnDutyCyclesPrev);
		_layerColumnInhibitKernel.setArg(2, _layers[l]._columnStates);
		_layerColumnInhibitKernel.setArg(3, layerSize);
		_layerColumnInhibitKernel.setArg(4, layerSizeInv);
		_layerColumnInhibitKernel.setArg(5, layerInhibitionRadius);
		_layerColumnInhibitKernel.setArg(6, layerInhibitionStep);
		_layerColumnInhibitKernel.setArg(7, seed);

		cs.getQueue().enqueueNDRangeKernel(_layerColumnInhibitKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		int layerWidth = _layerDescs[l]._width;

		Int2 lateralConnectionRadii;
		lateralConnectionRadii._x = _layerDescs[l]._lateralConnectionRadius;
		lateralConnectionRadii._y = _layerDescs[l]._lateralConnectionRadius;

		// Cell activation
		_layerCellActivateKernel.setArg(0, _layers[l]._columnStates);
		_layerCellActivateKernel.setArg(1, _layers[l]._cellStatesPrev);
		_layerCellActivateKernel.setArg(2, _layers[l]._cellPredictionsPrev);
		_layerCellActivateKernel.setArg(3, _layers[l]._cellWeightsPrev);
		_layerCellActivateKernel.setArg(4, _layers[l]._columnPredictionsPrev);
		_layerCellActivateKernel.setArg(5, _layers[l]._cellStates);
		_layerCellActivateKernel.setArg(6, _layerDescs[l]._cellsInColumn);
		_layerCellActivateKernel.setArg(7, lateralConnectionRadii);

		cs.getQueue().enqueueNDRangeKernel(_layerCellActivateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		cs.getQueue().flush();

		// Update prevs
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
		pPrevColumnStates = &_layers[l]._columnStates;
	}

	// Predict in reverse order
	for (int l = _layers.size() - 1; l >= 0; l--) {
		Int2 layerSize;
		layerSize._x = _layerDescs[l]._width;
		layerSize._y = _layerDescs[l]._height;

		Float2 layerSizeInv;
		layerSizeInv._x = 1.0f / _layerDescs[l]._width;
		layerSizeInv._y = 1.0f / _layerDescs[l]._height;

		Int2 lateralConnectionRadii;
		lateralConnectionRadii._x = _layerDescs[l]._lateralConnectionRadius;
		lateralConnectionRadii._y = _layerDescs[l]._lateralConnectionRadius;

		// Cell prediction
		if (l == _layers.size() - 1) {
			_layerCellPredictLastKernel.setArg(0, _layers[l]._columnAttentions);
			_layerCellPredictLastKernel.setArg(1, _layers[l]._columnStates);
			_layerCellPredictLastKernel.setArg(2, _layers[l]._cellStates);
			_layerCellPredictLastKernel.setArg(3, _layers[l]._cellWeightsPrev);
			_layerCellPredictLastKernel.setArg(4, _layers[l]._cellPredictions);
			_layerCellPredictLastKernel.setArg(5, _layerDescs[l]._cellsInColumn);
			_layerCellPredictLastKernel.setArg(6, layerSize);
			_layerCellPredictLastKernel.setArg(7, lateralConnectionRadii);

			cs.getQueue().enqueueNDRangeKernel(_layerCellPredictLastKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
		}
		else {
			Int2 nextLayerSize;
			nextLayerSize._x = _layerDescs[l]._width;
			nextLayerSize._y = _layerDescs[l]._height;

			_layerCellPredictKernel.setArg(0, _layers[l]._columnAttentions);
			_layerCellPredictKernel.setArg(1, _layers[l]._columnStates);
			_layerCellPredictKernel.setArg(2, _layers[l]._cellStates);
			_layerCellPredictKernel.setArg(3, _layers[l]._cellWeightsPrev);
			_layerCellPredictKernel.setArg(4, _layers[l + 1]._columnPredictions);
			_layerCellPredictKernel.setArg(5, _layers[l]._cellPredictions);
			_layerCellPredictKernel.setArg(6, _layerDescs[l]._cellsInColumn);
			_layerCellPredictKernel.setArg(7, layerSize);
			_layerCellPredictKernel.setArg(8, lateralConnectionRadii);
			_layerCellPredictKernel.setArg(9, layerSizeInv);
			_layerCellPredictKernel.setArg(10, nextLayerSize);

			cs.getQueue().enqueueNDRangeKernel(_layerCellPredictKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
		}

		// Column prediction
		_layerColumnPredictionKernel.setArg(0, _layers[l]._cellPredictions);
		_layerColumnPredictionKernel.setArg(1, _layers[l]._cellStates);
		_layerColumnPredictionKernel.setArg(2, _layers[l]._columnPredictions);
		_layerColumnPredictionKernel.setArg(3, _layerDescs[l]._cellsInColumn);

		cs.getQueue().enqueueNDRangeKernel(_layerColumnPredictionKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		cs.getQueue().flush();
	}
}

float HTMRL::retrieveQ(sys::ComputeSystem &cs) {
	float totalSum = _qBias;

	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	for (int l = 0; l < _layers.size(); l++) {
		_layerRetrieveQKernel.setArg(0, _layers[l]._columnAttentions);
		_layerRetrieveQKernel.setArg(1, _layers[l]._cellStates);
		_layerRetrieveQKernel.setArg(2, _layers[l]._cellQWeightsPrev);
		_layerRetrieveQKernel.setArg(3, _layers[l]._partialQSums);
		_layerRetrieveQKernel.setArg(4, _layerDescs[l]._cellsInColumn);

		cs.getQueue().enqueueNDRangeKernel(_layerRetrieveQKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		cs.getQueue().flush();
	}

	for (int l = 0; l < _layers.size(); l++) {
		// Retrieve result
		std::vector<float> result(_layerDescs[l]._width * _layerDescs[l]._height * 2);

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = 1;

			cs.getQueue().enqueueReadImage(_layers[l]._partialQSums, CL_TRUE, origin, region, 0, 0, &result[0]);
		}

		for (int i = 0; i < result.size(); i++)
			totalSum += result[i] * _layerDescs[l]._qInfluenceMultiplier;
	}

	return totalSum;
}

void HTMRL::learnSpatialTemporal(sys::ComputeSystem &cs, float columnConnectionAlpha, float cellConnectionAlpha, float reconstructionAlpha, bool learnSDR, bool learnPrediction, bool learnReconstruction) {
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;
	cl::Image2D* pPrevColumnStates = &_inputImage;

	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	for (int l = 0; l < _layers.size(); l++) {
		Int2 inputSize;
		inputSize._x = prevLayerWidth;
		inputSize._y = prevLayerHeight;

		Int2 layerSize;
		layerSize._x = _layerDescs[l]._width;
		layerSize._y = _layerDescs[l]._height;

		Float2 inputSizeInv;
		inputSizeInv._x = 1.0f / prevLayerWidth;
		inputSizeInv._y = 1.0f / prevLayerHeight;

		Float2 layerSizeInv;
		layerSizeInv._x = 1.0f / _layerDescs[l]._width;
		layerSizeInv._y = 1.0f / _layerDescs[l]._height;

		Int2 inputReceptiveFieldRadius;
		inputReceptiveFieldRadius._x = _layerDescs[l]._receptiveFieldRadius;
		inputReceptiveFieldRadius._y = _layerDescs[l]._receptiveFieldRadius;

		Float2 inputReceptiveFieldStep;
		inputReceptiveFieldStep._x = inputSizeInv._x * _layerDescs[l]._receptiveFieldRadius;
		inputReceptiveFieldStep._y = inputSizeInv._y * _layerDescs[l]._receptiveFieldRadius;

		Int2 layerReceptiveFieldRadius;
		layerReceptiveFieldRadius._x = _layerDescs[l]._receptiveFieldRadius;
		layerReceptiveFieldRadius._y = _layerDescs[l]._receptiveFieldRadius;

		Float2 layerReceptiveFieldStep;
		layerReceptiveFieldStep._x = layerSizeInv._x * _layerDescs[l]._receptiveFieldRadius;
		layerReceptiveFieldStep._y = layerSizeInv._y * _layerDescs[l]._receptiveFieldRadius;

		// Column weight update
		if (learnSDR) {
			_layerColumnWeightUpdateKernel.setArg(0, *pPrevColumnStates);
			_layerColumnWeightUpdateKernel.setArg(1, _layers[l]._columnStates);
			_layerColumnWeightUpdateKernel.setArg(2, _layers[l]._columnWeightsPrev);
			_layerColumnWeightUpdateKernel.setArg(3, _layers[l]._columnWeights);
			_layerColumnWeightUpdateKernel.setArg(4, layerSize);
			_layerColumnWeightUpdateKernel.setArg(5, layerSizeInv);
			_layerColumnWeightUpdateKernel.setArg(6, inputReceptiveFieldRadius);
			_layerColumnWeightUpdateKernel.setArg(7, inputReceptiveFieldStep);
			_layerColumnWeightUpdateKernel.setArg(8, inputSize);
			_layerColumnWeightUpdateKernel.setArg(9, columnConnectionAlpha);

			cs.getQueue().enqueueNDRangeKernel(_layerColumnWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
		}

		Int2 lateralConnectionRadii;
		lateralConnectionRadii._x = _layerDescs[l]._lateralConnectionRadius;
		lateralConnectionRadii._y = _layerDescs[l]._lateralConnectionRadius;

		// Lateral weight update
		if (learnPrediction) {
			if (l == _layers.size() - 1) {
				_layerCellWeightUpdateLastKernel.setArg(0, _layers[l]._columnAttentionsPrev);
				_layerCellWeightUpdateLastKernel.setArg(1, _layers[l]._columnStates);
				_layerCellWeightUpdateLastKernel.setArg(2, _layers[l]._cellStatesPrev);
				_layerCellWeightUpdateLastKernel.setArg(3, _layers[l]._cellWeightsPrev);
				_layerCellWeightUpdateLastKernel.setArg(4, _layers[l]._columnPredictionsPrev);
				_layerCellWeightUpdateLastKernel.setArg(5, _layers[l]._cellWeights);
				_layerCellWeightUpdateLastKernel.setArg(6, _layerDescs[l]._cellsInColumn);
				_layerCellWeightUpdateLastKernel.setArg(7, layerSize);
				_layerCellWeightUpdateLastKernel.setArg(8, lateralConnectionRadii);
				_layerCellWeightUpdateLastKernel.setArg(9, cellConnectionAlpha);

				cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateLastKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
			}
			else {
				Int2 nextLayerSize;
				nextLayerSize._x = _layerDescs[l]._width;
				nextLayerSize._y = _layerDescs[l]._height;

				_layerCellWeightUpdateKernel.setArg(0, _layers[l]._columnAttentionsPrev);
				_layerCellWeightUpdateKernel.setArg(1, _layers[l]._columnStates);
				_layerCellWeightUpdateKernel.setArg(2, _layers[l]._cellStatesPrev);
				_layerCellWeightUpdateKernel.setArg(3, _layers[l + 1]._columnPredictionsPrev);
				_layerCellWeightUpdateKernel.setArg(4, _layers[l]._cellWeightsPrev);
				_layerCellWeightUpdateKernel.setArg(5, _layers[l]._columnPredictionsPrev);
				_layerCellWeightUpdateKernel.setArg(6, _layers[l]._cellWeights);
				_layerCellWeightUpdateKernel.setArg(7, _layerDescs[l]._cellsInColumn);
				_layerCellWeightUpdateKernel.setArg(8, layerSize);
				_layerCellWeightUpdateKernel.setArg(9, lateralConnectionRadii);
				_layerCellWeightUpdateKernel.setArg(10, layerSizeInv);
				_layerCellWeightUpdateKernel.setArg(11, nextLayerSize);
				_layerCellWeightUpdateKernel.setArg(12, cellConnectionAlpha);

				cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
			}
		}

		cs.getQueue().flush();

		// Update prevs
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
		pPrevColumnStates = &_layers[l]._columnStates;
	}

	// Learn input reconstruction
	if (learnReconstruction) {
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

		_reconstructInputKernel.setArg(0, _reconstructionWeights);
		_reconstructInputKernel.setArg(1, _layers.front()._columnStates);
		_reconstructInputKernel.setArg(2, _reconstruction);
		_reconstructInputKernel.setArg(3, reconstructionReceptiveFieldRadii);
		_reconstructInputKernel.setArg(4, inputSizeInv);
		_reconstructInputKernel.setArg(5, layerSize);
		_reconstructInputKernel.setArg(6, layerSizeInv);

		cs.getQueue().enqueueNDRangeKernel(_reconstructInputKernel, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));

		_updateReconstructionKernel.setArg(0, _inputImage);
		_updateReconstructionKernel.setArg(1, _reconstruction);
		_updateReconstructionKernel.setArg(2, _reconstructionWeightsPrev);
		_updateReconstructionKernel.setArg(3, _layers.front()._columnStates);
		_updateReconstructionKernel.setArg(4, _reconstructionWeights);
		_updateReconstructionKernel.setArg(5, reconstructionReceptiveFieldRadii);
		_updateReconstructionKernel.setArg(6, inputSizeInv);
		_updateReconstructionKernel.setArg(7, layerSize);
		_updateReconstructionKernel.setArg(8, layerSizeInv);
		_updateReconstructionKernel.setArg(9, reconstructionAlpha);

		cs.getQueue().enqueueNDRangeKernel(_updateReconstructionKernel, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));

		cs.getQueue().flush();
	}
}

void HTMRL::updateQWeights(sys::ComputeSystem &cs, float tdError, float cellQWeightEligibilityDecay, float qBiasAlpha) {
	_qBias += qBiasAlpha * tdError;

	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	for (int l = _layers.size() - 1; l >= 0; l--) {
		_layerUpdateQWeightsKernel.setArg(0, _layers[l]._columnAttentions);
		_layerUpdateQWeightsKernel.setArg(1, _layers[l]._cellStates);
		_layerUpdateQWeightsKernel.setArg(2, _layers[l]._cellQWeightsPrev);
		_layerUpdateQWeightsKernel.setArg(3, _layers[l]._cellQWeights);
		_layerUpdateQWeightsKernel.setArg(4, _layerDescs[l]._cellsInColumn);
		_layerUpdateQWeightsKernel.setArg(5, cellQWeightEligibilityDecay);
		_layerUpdateQWeightsKernel.setArg(6, tdError * _layerDescs[l]._qInfluenceMultiplier);

		cs.getQueue().enqueueNDRangeKernel(_layerUpdateQWeightsKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
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

	_reconstructInputKernel.setArg(0, _reconstructionWeights);
	_reconstructInputKernel.setArg(1, _layers.front()._columnPredictions);
	_reconstructInputKernel.setArg(2, _reconstruction);
	_reconstructInputKernel.setArg(3, reconstructionReceptiveFieldRadii);
	_reconstructInputKernel.setArg(4, inputSizeInv);
	_reconstructInputKernel.setArg(5, layerSize);
	_reconstructInputKernel.setArg(6, layerSizeInv);

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

void HTMRL::updateDutyCycles(sys::ComputeSystem &cs, float dutyCycleDecay) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	for (int l = 0; l < _layers.size(); l++) {
		Int2 layerSize;
		layerSize._x = _layerDescs[l]._width;
		layerSize._y = _layerDescs[l]._height;

		Float2 layerSizeInv;
		layerSizeInv._x = 1.0f / _layerDescs[l]._width;
		layerSizeInv._y = 1.0f / _layerDescs[l]._height;

		Int2 layerAttentionRadius;
		layerAttentionRadius._x = _layerDescs[l]._attentionRadius;
		layerAttentionRadius._y = _layerDescs[l]._attentionRadius;

		Float2 layerAttentionStep;
		layerAttentionStep._x = layerSizeInv._x * _layerDescs[l]._attentionRadius;
		layerAttentionStep._y = layerSizeInv._y * _layerDescs[l]._attentionRadius;

		_layerColumnDutyCycleUpdateKernel.setArg(0, _layers[l]._columnDutyCyclesPrev);
		_layerColumnDutyCycleUpdateKernel.setArg(1, _layers[l]._columnStates);
		_layerColumnDutyCycleUpdateKernel.setArg(2, _layers[l]._columnDutyCycles);
		_layerColumnDutyCycleUpdateKernel.setArg(3, _layers[l]._columnAttentions);
		_layerColumnDutyCycleUpdateKernel.setArg(4, layerAttentionRadius);
		_layerColumnDutyCycleUpdateKernel.setArg(5, layerAttentionStep);
		_layerColumnDutyCycleUpdateKernel.setArg(6, layerSize);
		_layerColumnDutyCycleUpdateKernel.setArg(7, layerSizeInv);
		_layerColumnDutyCycleUpdateKernel.setArg(8, dutyCycleDecay);

		cs.getQueue().enqueueNDRangeKernel(_layerColumnDutyCycleUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		cs.getQueue().flush();
	}
}

void HTMRL::step(sys::ComputeSystem &cs, float reward, float columnConnectionAlpha, float dutyCycleDecay, float cellConnectionAlpha, float reconstructionAlpha, float cellQWeightEligibilityDecay, float qBiasAlpha, int annealingIterations, float annealingStdDev, float annealingBreakChance, float annealingDecay, float annealingMomentum, float alpha, float gamma, float tauInv, float outputBreakChance, float outputPerturbationStdDev, std::mt19937 &generator) {
	stepBegin();
	
	// Complete input
	std::vector<float> maxQInput = _input;

	for (int j = 0; j < _input.size(); j++)
	if (_actionMask[j]) {
		_input[j] = _prevOutputExploratory[j];

		maxQInput[j] = _prevOutput[j];
	}

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
	std::uniform_int_distribution<int> seedDist(0, 10000);

	int seed = seedDist(generator);
	
	activate(maxQInput, cs, seed);

	float maxQ = retrieveQ(cs);

	activate(_input, cs, seed);

	float exploratoryQ = retrieveQ(cs);

	// Get prediction for next action
	std::vector<float> output;
	
	getReconstructedPrediction(output, cs);

	// Exploratory action
	std::normal_distribution<float> outputPerturbationDist(0.0f, outputPerturbationStdDev);

	for (int j = 0; j < _input.size(); j++)
	if (_actionMask[j]) {
		if (uniformDist(generator) < outputBreakChance)
			_exploratoryOutput[j] = uniformDist(generator) * 2.0f - 1.0f;
		else
			_exploratoryOutput[j] = std::min<float>(1.0f, std::max<float>(-1.0f, output[j] + outputPerturbationDist(generator)));
	}
	else
		_exploratoryOutput[j] = output[j];

	float newQ = reward + gamma * maxQ;

	float suboptimality = std::max<float>(0.0f, (_prevMaxQ - newQ) * tauInv);

	float adv = newQ - suboptimality;

	float tdError = alpha * (adv - _prevValue);

	std::cout << tdError << " " << exploratoryQ << std::endl;

	_prevMaxQ = maxQ;
	_prevValue = exploratoryQ;

	//activate(_input, cs, seed);

	updateQWeights(cs, tdError, cellQWeightEligibilityDecay, qBiasAlpha);

	if (tdError < 0.0f)
		activate(maxQInput, cs, seed);

	learnSpatialTemporal(cs, columnConnectionAlpha, cellConnectionAlpha, reconstructionAlpha, true, true, true);

	activate(_input, cs, seed);

	updateDutyCycles(cs, dutyCycleDecay);

	_prevOutput = output;
	_prevOutputExploratory = _exploratoryOutput;

	/*stepBegin();

	// Get initial Q
	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
	std::uniform_int_distribution<int> seedDist(0, 10000);

	int seed = seedDist(generator);

	std::vector<float> oldOutput = _exploratoryOutput;

	_exploratoryOutput = _input;

	for (int j = 0; j < _exploratoryOutput.size(); j++)
	if (_actionMask[j])
		_exploratoryOutput[j] = oldOutput[j];

	activate(_exploratoryOutput, cs, seed);
	
	float maxQ = retrieveQ(cs);
	
	std::vector<float> testOutput(_exploratoryOutput.size());
	
	float perturbationMultiplier = 1.0f;
	
	std::normal_distribution<float> perturbationDist(0.0f, annealingStdDev);
	
	std::vector<float> prevDOutput(_exploratoryOutput.size(), 0.0f);
	
	for (int i = 0; i < annealingIterations; i++) {
		for (int j = 0; j < _exploratoryOutput.size(); j++)
		if (_actionMask[j]) {
			if (uniformDist(generator) < annealingBreakChance)
				testOutput[j] = uniformDist(generator) * 2.0f - 1.0f;
			else
				testOutput[j] = std::min<float>(1.0f, std::max<float>(-1.0f, _exploratoryOutput[j] + prevDOutput[j] * annealingMomentum + perturbationMultiplier * perturbationDist(generator)));
		}
		else
			testOutput[j] = _input[j];
	
		activate(testOutput, cs, seed);
	
		float result = retrieveQ(cs);
	
		if (result >= maxQ) {
			maxQ = result;
	
			for (int j = 0; j < _exploratoryOutput.size(); j++)
				prevDOutput[j] = testOutput[j] - _exploratoryOutput[j];
	
			_exploratoryOutput = testOutput;
		}
	
		perturbationMultiplier *= annealingDecay;
	}

	// Exploratory action
	std::normal_distribution<float> outputPerturbationDist(0.0f, outputPerturbationStdDev);

	for (int j = 0; j < _input.size(); j++)
	if (_actionMask[j]) {
		if (uniformDist(generator) < outputBreakChance)
			_exploratoryOutput[j] = uniformDist(generator) * 2.0f - 1.0f;
		else
			_exploratoryOutput[j] = std::min<float>(1.0f, std::max<float>(-1.0f, _exploratoryOutput[j] + outputPerturbationDist(generator)));
	}

	activate(_exploratoryOutput, cs, seed);

	float exploratoryQ = retrieveQ(cs);

	float newQ = reward + gamma * exploratoryQ;

	float suboptimality = std::max<float>(0.0f, (_prevMaxQ - newQ) * tauInv);

	float adv = newQ - suboptimality;

	float tdError = alpha * (adv - _prevValue);

	std::cout << tdError << " " << maxQ << std::endl;

	_prevMaxQ = maxQ;
	_prevValue = exploratoryQ;

	updateDutyCycles(cs, dutyCycleDecay);

	updateQWeights(cs, tdError, cellQWeightEligibilityDecay, qBiasAlpha);

	learnSpatialTemporal(cs, columnConnectionAlpha, cellConnectionAlpha, reconstructionAlpha, true, tdError > 0.0f, true);

	getReconstructedPrediction(_prevOutput, cs);*/
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

			color.a = _exploratoryOutput[x + y * _inputWidth] * (255.0f - 3.0f) + 3;

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

		cs.getQueue().enqueueReadImage(_layers[l]._columnPredictions, CL_TRUE, origin, region, 0, 0, &state[0]);

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