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

	_qBias = weightDist(generator);
	_prevMaxQ = 0.0f;
	_prevValue = 0.0f;

	cl::Kernel initKernel = cl::Kernel(program.getProgram(), "initialize");

	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;

	int maxWidth = 0;
	int maxHeight = 0;

	_input.clear();
	_input.assign(_inputWidth * _inputHeight, 0.0f);

	_output.clear();
	_output.assign(_inputWidth * _inputHeight, 0.0f);

	_prevPrediction.clear();
	_prevPrediction.assign(_inputWidth * _inputHeight, 0.0f);

	_prevInput.clear();
	_prevInput.assign(_inputWidth * _inputHeight, 0.0f);

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
		
		_layers[l]._columnStates = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._columnStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._columnWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, receptiveFieldSize);
		_layers[l]._columnWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, receptiveFieldSize);

		_layers[l]._cellStates = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._cellsInColumn);
		_layers[l]._cellStatesPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._cellsInColumn);

		_layers[l]._cellPredictions = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._cellsInColumn);
		_layers[l]._cellPredictionsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._cellsInColumn);

		_layers[l]._cellWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height * _layerDescs[l]._cellsInColumn, lateralConnectionsSize);
		_layers[l]._cellWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height * _layerDescs[l]._cellsInColumn, lateralConnectionsSize);

		_layers[l]._columnPredictions = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._columnPredictionsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._cellQWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._cellsInColumn + 1);
		_layers[l]._cellQWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._cellsInColumn + 1);

		_layers[l]._columnOutputs = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._columnOutputsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		Uint2 seed;
		seed._x = uniformDist(generator);
		seed._y = uniformDist(generator);

		initKernel.setArg(0, _layers[l]._columnActivations);
		initKernel.setArg(1, _layers[l]._columnStates);
		initKernel.setArg(2, _layers[l]._columnWeights);
		initKernel.setArg(3, _layers[l]._cellStates);
		initKernel.setArg(4, _layers[l]._cellWeights);
		initKernel.setArg(5, _layers[l]._cellPredictions);
		initKernel.setArg(6, _layers[l]._cellQWeights);
		initKernel.setArg(7, _layers[l]._columnOutputs);
		initKernel.setArg(8, _layerDescs[l]._cellsInColumn);
		initKernel.setArg(9, _layerDescs[l]._width);
		initKernel.setArg(10, receptiveFieldSize);
		initKernel.setArg(11, lateralConnectionsSize);
		initKernel.setArg(12, seed);
		initKernel.setArg(13, minInitWeight);
		initKernel.setArg(14, maxInitWeight);
		initKernel.setArg(15, minInitWidth);
		initKernel.setArg(16, maxInitWidth);

		cs.getQueue().enqueueNDRangeKernel(initKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

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
			region[2] = _layerDescs[l]._cellsInColumn + 1;

			cs.getQueue().enqueueCopyImage(_layers[l]._cellQWeights, _layers[l]._cellQWeightsPrev, origin, origin, region);
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

			cs.getQueue().enqueueCopyImage(_layers[l]._columnOutputs, _layers[l]._columnOutputsPrev, origin, origin, region);
		}

		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
	}

	_qSummationBuffer = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), maxWidth, maxHeight);
	_halfQSummationBuffer = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), std::ceil(maxWidth * 0.5f), std::ceil(maxHeight * 0.5f));

	_reconstructionReceptiveRadius = std::ceil(static_cast<float>(_layerDescs.front()._width) / static_cast<float>(_inputWidth) / static_cast<float>(_layerDescs.front()._receptiveFieldRadius));
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
	_layerCellActivateKernel = cl::Kernel(program.getProgram(), "layerCellActivate");
	_layerCellWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerCellWeightUpdate");
	_layerCellWeightUpdateLastKernel = cl::Kernel(program.getProgram(), "layerCellWeightUpdateLast");
	_layerCellPredictKernel = cl::Kernel(program.getProgram(), "layerCellPredict");
	_layerCellPredictLastKernel = cl::Kernel(program.getProgram(), "layerCellPredictLast");
	_layerColumnWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerColumnWeightUpdate");
	_layerColumnPredictionKernel = cl::Kernel(program.getProgram(), "layerColumnPrediction");
	_layerColumnOutputKernel = cl::Kernel(program.getProgram(), "layerColumnOutput");
	_layerRetrievePartialQSumsKernel = cl::Kernel(program.getProgram(), "layerRetrievePartialQSums");
	_layerDownsampleKernel = cl::Kernel(program.getProgram(), "layerDownsample");
	_layerUpdateQWeightsKernel = cl::Kernel(program.getProgram(), "layerUpdateQWeights");

	_reconstructInputKernel = cl::Kernel(program.getProgram(), "reconstructInput");
	_updateReconstructionKernel = cl::Kernel(program.getProgram(), "updateReconstruction");

	cs.getQueue().finish();
}

void HTMRL::stepBegin() {
	for (int l = 0; l < _layers.size(); l++) {
		std::swap(_layers[l]._columnStates, _layers[l]._columnStatesPrev);
		std::swap(_layers[l]._columnWeights, _layers[l]._columnWeightsPrev);
		std::swap(_layers[l]._cellStates, _layers[l]._cellStatesPrev);
		std::swap(_layers[l]._cellPredictions, _layers[l]._cellPredictionsPrev);
		std::swap(_layers[l]._cellWeights, _layers[l]._cellWeightsPrev);
		std::swap(_layers[l]._columnPredictions, _layers[l]._columnPredictionsPrev);
		std::swap(_layers[l]._cellQWeights, _layers[l]._cellQWeightsPrev);
		std::swap(_layers[l]._columnOutputs, _layers[l]._columnOutputsPrev);
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
		_layerColumnActivateKernel.setArg(1, _layers[l]._columnActivations);
		_layerColumnActivateKernel.setArg(2, _layers[l]._columnWeightsPrev);
		_layerColumnActivateKernel.setArg(3, layerSizeInv);
		_layerColumnActivateKernel.setArg(4, inputReceptiveFieldRadius);
		_layerColumnActivateKernel.setArg(5, inputReceptiveFieldStep);

		cs.getQueue().enqueueNDRangeKernel(_layerColumnActivateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		Int2 layerInhibitionRadius;
		layerInhibitionRadius._x = _layerDescs[l]._inhibitionRadius;
		layerInhibitionRadius._y = _layerDescs[l]._inhibitionRadius;

		Float2 layerInhibitionStep;
		layerInhibitionStep._x = layerSizeInv._x * _layerDescs[l]._inhibitionRadius;
		layerInhibitionStep._y = layerSizeInv._y * _layerDescs[l]._inhibitionRadius;

		// Inhibition
		_layerColumnInhibitKernel.setArg(0, _layers[l]._columnActivations);
		_layerColumnInhibitKernel.setArg(1, _layers[l]._columnStates);
		_layerColumnInhibitKernel.setArg(2, layerSizeInv);
		_layerColumnInhibitKernel.setArg(3, layerInhibitionRadius);
		_layerColumnInhibitKernel.setArg(4, layerInhibitionStep);
		_layerColumnInhibitKernel.setArg(5, seed);

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
		_layerCellActivateKernel.setArg(7, layerWidth);
		_layerCellActivateKernel.setArg(8, lateralConnectionRadii);

		cs.getQueue().enqueueNDRangeKernel(_layerCellActivateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		// Cell prediction
		if (l == _layers.size() - 1) {
			_layerCellPredictLastKernel.setArg(0, _layers[l]._columnStates);
			_layerCellPredictLastKernel.setArg(1, _layers[l]._cellStates);
			_layerCellPredictLastKernel.setArg(2, _layers[l]._cellWeightsPrev);
			_layerCellPredictLastKernel.setArg(3, _layers[l]._cellPredictions);
			_layerCellPredictLastKernel.setArg(4, _layerDescs[l]._cellsInColumn);
			_layerCellPredictLastKernel.setArg(5, layerWidth);
			_layerCellPredictLastKernel.setArg(6, lateralConnectionRadii);

			cs.getQueue().enqueueNDRangeKernel(_layerCellPredictLastKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
		}
		else {
			_layerCellPredictKernel.setArg(0, _layers[l]._columnStates);
			_layerCellPredictKernel.setArg(1, _layers[l]._cellStates);
			_layerCellPredictKernel.setArg(2, _layers[l]._cellWeightsPrev);
			_layerCellPredictKernel.setArg(3, _layers[l + 1]._columnOutputsPrev);
			_layerCellPredictKernel.setArg(4, _layers[l]._cellPredictions);
			_layerCellPredictKernel.setArg(5, _layerDescs[l]._cellsInColumn);
			_layerCellPredictKernel.setArg(6, layerWidth);
			_layerCellPredictKernel.setArg(7, lateralConnectionRadii);
			_layerCellPredictKernel.setArg(8, layerSizeInv);

			cs.getQueue().enqueueNDRangeKernel(_layerCellPredictKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
		}

		// Column prediction
		_layerColumnPredictionKernel.setArg(0, _layers[l]._cellPredictions);
		_layerColumnPredictionKernel.setArg(1, _layers[l]._cellStates);
		_layerColumnPredictionKernel.setArg(2, _layers[l]._columnPredictions);
		_layerColumnPredictionKernel.setArg(3, _layerDescs[l]._cellsInColumn);

		cs.getQueue().enqueueNDRangeKernel(_layerColumnPredictionKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		// Column output
		_layerColumnOutputKernel.setArg(0, _layers[l]._columnStates);
		_layerColumnOutputKernel.setArg(1, _layers[l]._columnPredictions);
		_layerColumnOutputKernel.setArg(2, _layers[l]._columnOutputs);

		cs.getQueue().enqueueNDRangeKernel(_layerColumnOutputKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		cs.getQueue().flush();

		// Update prevs
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
		pPrevColumnStates = &_layers[l]._columnOutputs;
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

	Int2 downsampleSize;

	downsampleSize._x = 2;
	downsampleSize._y = 2;

	for (int l = 0; l < _layers.size(); l++) {
		_layerRetrievePartialQSumsKernel.setArg(0, _layers[l]._cellStates);
		_layerRetrievePartialQSumsKernel.setArg(1, _layers[l]._columnStates);
		_layerRetrievePartialQSumsKernel.setArg(2, _layers[l]._cellQWeightsPrev);
		_layerRetrievePartialQSumsKernel.setArg(3, _qSummationBuffer);
		_layerRetrievePartialQSumsKernel.setArg(4, _layerDescs[l]._cellsInColumn);

		cs.getQueue().enqueueNDRangeKernel(_layerRetrievePartialQSumsKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

#if HTMRL_GPU_Q_SUMMATION
		// Downsample
		int width = _layerDescs[l]._width;
		int height = _layerDescs[l]._height;

		cl::Image2D* pPing = &_qSummationBuffer;
		cl::Image2D* pPong = &_halfQSummationBuffer;

		while (width > 1 && height > 1) {
			_layerDownsampleKernel.setArg(0, *pPing);
			_layerDownsampleKernel.setArg(1, *pPong);
			_layerDownsampleKernel.setArg(2, downsampleSize);

			cs.getQueue().enqueueNDRangeKernel(_layerDownsampleKernel, cl::NullRange, cl::NDRange(width / 2, height / 2));

			std::swap(pPing, pPong);

			width /= 2;
			height /= 2;
		}

		// Retrieve result
		std::vector<float> result(width * height);

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = width;
			region[1] = height;
			region[2] = 1;

			cs.getQueue().enqueueReadImage(*pPing, CL_TRUE, origin, region, 0, 0, &result[0]);
		}

		for (int i = 0; i < result.size(); i++)
			totalSum += result[i];
#else
		// Retrieve result
		std::vector<float> result(_layerDescs[l]._width * _layerDescs[l]._height);

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = 1;

			cs.getQueue().enqueueReadImage(_qSummationBuffer, CL_TRUE, origin, region, 0, 0, &result[0]);
		}

		for (int i = 0; i < result.size(); i++)
			totalSum += result[i];
#endif
	}

	return totalSum;
}

void HTMRL::learnSpatialTemporal(sys::ComputeSystem &cs, float columnConnectionAlpha, float columnWidthAlpha, float cellConnectionAlpha, float reconstructionAlpha) {
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
		_layerColumnWeightUpdateKernel.setArg(0, *pPrevColumnStates);
		_layerColumnWeightUpdateKernel.setArg(1, _layers[l]._columnStates);
		_layerColumnWeightUpdateKernel.setArg(2, _layers[l]._columnWeightsPrev);
		_layerColumnWeightUpdateKernel.setArg(3, _layers[l]._columnWeights);
		_layerColumnWeightUpdateKernel.setArg(4, layerSizeInv);
		_layerColumnWeightUpdateKernel.setArg(5, inputReceptiveFieldRadius);
		_layerColumnWeightUpdateKernel.setArg(6, inputReceptiveFieldStep);
		_layerColumnWeightUpdateKernel.setArg(7, columnConnectionAlpha);
		_layerColumnWeightUpdateKernel.setArg(8, columnWidthAlpha);

		cs.getQueue().enqueueNDRangeKernel(_layerColumnWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		int layerWidth = _layerDescs[l]._width;

		Int2 lateralConnectionRadii;
		lateralConnectionRadii._x = _layerDescs[l]._lateralConnectionRadius;
		lateralConnectionRadii._y = _layerDescs[l]._lateralConnectionRadius;

		// Lateral weight update
		if (l == _layers.size() - 1) {
			_layerCellWeightUpdateLastKernel.setArg(0, _layers[l]._columnStates);
			_layerCellWeightUpdateLastKernel.setArg(1, _layers[l]._cellStatesPrev);
			_layerCellWeightUpdateLastKernel.setArg(2, _layers[l]._cellPredictionsPrev);
			_layerCellWeightUpdateLastKernel.setArg(3, _layers[l]._cellWeightsPrev);
			_layerCellWeightUpdateLastKernel.setArg(4, _layers[l]._columnPredictionsPrev);
			_layerCellWeightUpdateLastKernel.setArg(5, _layers[l]._cellWeights);
			_layerCellWeightUpdateLastKernel.setArg(6, _layerDescs[l]._cellsInColumn);
			_layerCellWeightUpdateLastKernel.setArg(7, layerWidth);
			_layerCellWeightUpdateLastKernel.setArg(8, lateralConnectionRadii);
			_layerCellWeightUpdateLastKernel.setArg(9, cellConnectionAlpha);

			cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateLastKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
		}
		else {
			_layerCellWeightUpdateKernel.setArg(0, _layers[l]._columnStates);
			_layerCellWeightUpdateKernel.setArg(1, _layers[l]._cellStatesPrev);
			_layerCellWeightUpdateKernel.setArg(2, _layers[l]._cellPredictionsPrev);
			_layerCellWeightUpdateKernel.setArg(3, _layers[l + 1]._columnOutputsPrev);
			_layerCellWeightUpdateKernel.setArg(4, _layers[l]._cellWeightsPrev);
			_layerCellWeightUpdateKernel.setArg(5, _layers[l]._columnPredictionsPrev);
			_layerCellWeightUpdateKernel.setArg(6, _layers[l]._cellWeights);
			_layerCellWeightUpdateKernel.setArg(7, _layerDescs[l]._cellsInColumn);
			_layerCellWeightUpdateKernel.setArg(8, layerWidth);
			_layerCellWeightUpdateKernel.setArg(9, lateralConnectionRadii);
			_layerCellWeightUpdateKernel.setArg(10, layerSizeInv);
			_layerCellWeightUpdateKernel.setArg(11, cellConnectionAlpha);

			cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
		}

		cs.getQueue().flush();

		// Update prevs
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
		pPrevColumnStates = &_layers[l]._columnOutputs;
	}

	// Learn input reconstruction
	Float2 inputSizeInv;
	inputSizeInv._x = 1.0f / _inputWidth;
	inputSizeInv._y = 1.0f / _inputHeight;

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
	_reconstructInputKernel.setArg(5, layerSizeInv);

	cs.getQueue().enqueueNDRangeKernel(_reconstructInputKernel, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));

	_updateReconstructionKernel.setArg(0, _inputImage);
	_updateReconstructionKernel.setArg(1, _reconstruction);
	_updateReconstructionKernel.setArg(2, _reconstructionWeightsPrev);
	_updateReconstructionKernel.setArg(3, _layers.front()._columnStates);
	_updateReconstructionKernel.setArg(4, _reconstructionWeights);
	_updateReconstructionKernel.setArg(5, reconstructionReceptiveFieldRadii);
	_updateReconstructionKernel.setArg(6, inputSizeInv);
	_updateReconstructionKernel.setArg(7, layerSizeInv);
	_updateReconstructionKernel.setArg(8, reconstructionAlpha);

	cs.getQueue().enqueueNDRangeKernel(_updateReconstructionKernel, cl::NullRange, cl::NDRange(_inputWidth, _inputHeight));
}

void HTMRL::learnQ(sys::ComputeSystem &cs, float tdError, float cellQWeightEligibilityDecay, float qBiasAlpha) {
	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	_qBias += qBiasAlpha * tdError;

	for (int l = 0; l < _layers.size(); l++) {
		// Cell Q weights update
		_layerUpdateQWeightsKernel.setArg(0, _layers[l]._cellStates);
		_layerUpdateQWeightsKernel.setArg(1, _layers[l]._columnStates);
		_layerUpdateQWeightsKernel.setArg(2, _layers[l]._cellQWeightsPrev);
		_layerUpdateQWeightsKernel.setArg(3, _layers[l]._cellQWeights);
		_layerUpdateQWeightsKernel.setArg(4, tdError);
		_layerUpdateQWeightsKernel.setArg(5, cellQWeightEligibilityDecay);
		_layerUpdateQWeightsKernel.setArg(6, _layerDescs[l]._cellsInColumn);

		cs.getQueue().enqueueNDRangeKernel(_layerUpdateQWeightsKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		cs.getQueue().flush();
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
	_reconstructInputKernel.setArg(5, layerSizeInv);

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

void HTMRL::step(sys::ComputeSystem &cs, float reward, float columnConnectionAlpha, float columnWidthAlpha, float cellConnectionAlpha, float reconstructionAlpha, float cellQWeightEligibilityDecay, float qBiasAlpha, int annealingIterations, float annealingStdDev, float annealingBreakChance, float annealingDecay, float annealingMomentum, float alpha, float gamma, float tauInv, float outputBreakChance, float outputPerturbationStdDev, std::mt19937 &generator) {
	stepBegin();
	
	_output = _input;

	for (int j = 0; j < _output.size(); j++)
	if (_actionMask[j])
		_output[j] = std::min<float>(1.0f, std::max<float>(-1.0f, _prevPrediction[j]));

	std::uniform_int_distribution<int> seedDist(0, 10000);

	int seed = seedDist(generator);

	// Get initial Q
	activate(_output, cs, seed);

	float maxQ = retrieveQ(cs);

	std::vector<float> testOutput(_output.size());

	float perturbationMultiplier = 1.0f;

	std::normal_distribution<float> perturbationDist(0.0f, annealingStdDev);
	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	std::vector<float> prevDOutput(_output.size(), 0.0f);

	for (int i = 0; i < annealingIterations; i++) {
		for (int j = 0; j < _output.size(); j++)
		if (_actionMask[j]) {
			if (uniformDist(generator) < annealingBreakChance)
				testOutput[j] = uniformDist(generator) * 2.0f - 1.0f;
			else
				testOutput[j] = std::min<float>(1.0f, std::max<float>(-1.0f, _output[j] + prevDOutput[j] * annealingMomentum + perturbationMultiplier * perturbationDist(generator)));
		}
		else
			testOutput[j] = _input[j];

		activate(testOutput, cs, seed);

		float result = retrieveQ(cs);

		if (result >= maxQ) {
			maxQ = result;

			for (int j = 0; j < _output.size(); j++)
				prevDOutput[j] = testOutput[j] - _output[j];

			_output = testOutput;
		}

		perturbationMultiplier *= annealingDecay;
	}

	// Exploratory action
	std::normal_distribution<float> outputPerturbationDist(0.0f, outputPerturbationStdDev);

	for (int j = 0; j < _output.size(); j++)
	if (_actionMask[j]) {
		if (uniformDist(generator) < outputBreakChance)
			_output[j] = uniformDist(generator) * 2.0f - 1.0f;
		else
			_output[j] = std::min<float>(1.0f, std::max<float>(-1.0f, _output[j] + outputPerturbationDist(generator)));
	}
	else
		_output[j] = _input[j];

	activate(_output, cs, seed);

	float exploratoryQ = retrieveQ(cs);

	maxQ = std::max<float>(maxQ, exploratoryQ);

	float newQ = reward + gamma * maxQ;

	float suboptimality = (_prevMaxQ - newQ) * tauInv;

	float adv = newQ - suboptimality;

	float tdError = alpha * (adv - _prevValue);

	std::cout << tdError << " " << exploratoryQ << " " << _output[4] << std::endl;

	_prevMaxQ = maxQ;
	_prevValue = exploratoryQ;

	learnSpatialTemporal(cs, columnConnectionAlpha, columnWidthAlpha, cellConnectionAlpha, reconstructionAlpha);

	learnQ(cs, tdError, cellQWeightEligibilityDecay, qBiasAlpha);

	// Get prediction for next action
	getReconstructedPrediction(_prevPrediction, cs);

	_prevInput = _output;

	cs.getQueue().finish();
}

void HTMRL::exportCellData(sys::ComputeSystem &cs, std::vector<std::shared_ptr<sf::Image>> &images, unsigned long seed) const {
	std::mt19937 generator(seed);
	
	int maxWidth = 0;
	int maxHeight = 0;

	for (int l = 0; l < _layers.size(); l++) {
		maxWidth = std::max<int>(maxWidth, _layerDescs[l]._width);
		maxHeight = std::max<int>(maxHeight, _layerDescs[l]._height);
	}
	
	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
	
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

				color.a = state[x + y * _layerDescs[l]._width + ci * _layerDescs[l]._width * _layerDescs[l]._height] * (255.0f - 3.0f) + 3;

				image->setPixel(x - _layerDescs[l]._width / 2 + maxWidth / 2, y - _layerDescs[l]._height / 2 + maxHeight / 2, color);
			}

			images.push_back(image);
		}
	}
}