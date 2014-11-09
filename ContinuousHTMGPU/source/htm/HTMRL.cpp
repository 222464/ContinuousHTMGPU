#include <htm/HTMRL.h>

using namespace htm;

void HTMRL::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<bool> &actionMask, float minInitWeight, float maxInitWeight, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;
	
	_layerDescs = layerDescs;
	
	_layers.resize(_layerDescs.size());

	_actionMask = actionMask;

	std::uniform_real_distribution<float> weightDist(minInitWeight, maxInitWeight);

	_qBias = weightDist(generator);
	_qEligibility = 0.0f;
	_prevQ = 0.0f;

	cl::Kernel initKernel = cl::Kernel(program.getProgram(), "initialize");

	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;

	int maxWidth = 0;
	int maxHeight = 0;

	_input.clear();
	_input.assign(_inputWidth * _inputHeight, 0.0f);

	_output.clear();
	_output.assign(_inputWidth * _inputHeight, 0.0f);

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
		int lateralConnectionsSize = std::pow(_layerDescs[l]._lateralConnectionRadius * 2 + 1, 2) * _layerDescs[l]._cellsInColumn + 1; // + 1 for bias

		_layers[l]._columnActivations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		
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
		initKernel.setArg(7, _layerDescs[l]._cellsInColumn);
		initKernel.setArg(8, _layerDescs[l]._width);
		initKernel.setArg(9, receptiveFieldSize);
		initKernel.setArg(10, lateralConnectionsSize);
		initKernel.setArg(11, seed);
		initKernel.setArg(12, minInitWeight);
		initKernel.setArg(13, maxInitWeight);

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
			region[2] = _layerDescs[l]._cellsInColumn;

			cs.getQueue().enqueueCopyImage(_layers[l]._cellQWeights, _layers[l]._cellQWeightsPrev, origin, origin, region);
		}

		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
	}

	_qSummationBuffer = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), maxWidth, maxHeight);
	_halfQSummationBuffer = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), std::ceil(maxWidth * 0.5f), std::ceil(maxHeight * 0.5f));

	_layerColumnActivateKernel = cl::Kernel(program.getProgram(), "layerColumnActivate");
	_layerColumnInhibitKernel = cl::Kernel(program.getProgram(), "layerColumnInhibit");
	_layerCellActivateKernel = cl::Kernel(program.getProgram(), "layerCellActivate");
	_layerCellWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerCellWeightUpdate");
	_layerCellPredictKernel = cl::Kernel(program.getProgram(), "layerCellPredict");
	_layerColumnWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerColumnWeightUpdate");
	_layerColumnPredictionKernel = cl::Kernel(program.getProgram(), "layerColumnPrediction");
	_layerRetrievePartialQSumsKernel = cl::Kernel(program.getProgram(), "layerRetrievePartialQSums");
	_layerDownsampleKernel = cl::Kernel(program.getProgram(), "layerDownsample");
	_layerUpdateQWeightsKernel = cl::Kernel(program.getProgram(), "layerUpdateQWeights");
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
	}
}

void HTMRL::activate(std::vector<float> &input, sys::ComputeSystem &cs) {
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

		int inputDiameter = 2 * _layerDescs[l]._receptiveFieldRadius + 1;

		Float2 inputReceptiveFieldStep;
		inputReceptiveFieldStep._x = inputSizeInv._x * _layerDescs[l]._receptiveFieldRadius / inputDiameter;
		inputReceptiveFieldStep._y = inputSizeInv._y * _layerDescs[l]._receptiveFieldRadius / inputDiameter;

		// Activation
		_layerColumnActivateKernel.setArg(0, *pPrevColumnStates);
		_layerColumnActivateKernel.setArg(1, _layers[l]._columnActivations);
		_layerColumnActivateKernel.setArg(2, _layers[l]._columnWeightsPrev);
		_layerColumnActivateKernel.setArg(3, layerSizeInv);
		_layerColumnActivateKernel.setArg(4, inputReceptiveFieldRadius);
		_layerColumnActivateKernel.setArg(5, inputReceptiveFieldStep);

		cs.getQueue().enqueueNDRangeKernel(_layerColumnActivateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		Int2 layerReceptiveFieldRadius;
		layerReceptiveFieldRadius._x = _layerDescs[l]._receptiveFieldRadius;
		layerReceptiveFieldRadius._y = _layerDescs[l]._receptiveFieldRadius;

		Float2 layerReceptiveFieldStep;
		layerReceptiveFieldStep._x = layerSizeInv._x * _layerDescs[l]._receptiveFieldRadius / inputDiameter;
		layerReceptiveFieldStep._y = layerSizeInv._y * _layerDescs[l]._receptiveFieldRadius / inputDiameter;

		// Inhibition
		_layerColumnInhibitKernel.setArg(0, _layers[l]._columnActivations);
		_layerColumnInhibitKernel.setArg(1, _layers[l]._columnStates);
		_layerColumnInhibitKernel.setArg(2, layerSizeInv);
		_layerColumnInhibitKernel.setArg(3, layerReceptiveFieldRadius);
		_layerColumnInhibitKernel.setArg(4, layerReceptiveFieldStep);

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
		_layerCellPredictKernel.setArg(0, _layers[l]._columnStates);
		_layerCellPredictKernel.setArg(1, _layers[l]._cellStates);
		_layerCellPredictKernel.setArg(2, _layers[l]._cellWeights);
		_layerCellPredictKernel.setArg(3, _layers[l]._cellPredictions);
		_layerCellPredictKernel.setArg(4, _layerDescs[l]._cellsInColumn);
		_layerCellPredictKernel.setArg(5, layerWidth);
		_layerCellPredictKernel.setArg(6, lateralConnectionRadii);

		cs.getQueue().enqueueNDRangeKernel(_layerCellPredictKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		_layerColumnPredictionKernel.setArg(0, _layers[l]._cellPredictions);
		_layerColumnPredictionKernel.setArg(1, _layers[l]._cellStates);
		_layerColumnPredictionKernel.setArg(2, _layers[l]._columnPredictions);
		_layerColumnPredictionKernel.setArg(3, _layerDescs[l]._cellsInColumn);

		cs.getQueue().enqueueNDRangeKernel(_layerColumnPredictionKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		// Update prevs
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
		pPrevColumnStates = &_layers[l]._columnStates;
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
		_layerRetrievePartialQSumsKernel.setArg(0, _layers[l]._cellStates);
		_layerRetrievePartialQSumsKernel.setArg(1, _layers[l]._cellQWeightsPrev);
		_layerRetrievePartialQSumsKernel.setArg(2, _qSummationBuffer);
		_layerRetrievePartialQSumsKernel.setArg(3, _layerDescs[l]._cellsInColumn);

		cs.getQueue().enqueueNDRangeKernel(_layerRetrievePartialQSumsKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		// Downsample
		int width = _layerDescs[l]._width;
		int height = _layerDescs[l]._height;

		cl::Image2D* pPing = &_qSummationBuffer;
		cl::Image2D* pPong = &_halfQSummationBuffer;

		while (width > 1 || height > 1) {
			_layerDownsampleKernel.setArg(0, *pPing);
			_layerDownsampleKernel.setArg(1, *pPong);
			_layerDownsampleKernel.setArg(2, 2);

			cs.getQueue().enqueueNDRangeKernel(_layerDownsampleKernel, cl::NullRange, cl::NDRange(width, height));

			std::swap(pPing, pPong);

			width /= 2;
			height /= 2;
		}

		// Retrieve result
		float result;

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = 1;
			region[1] = 1;
			region[2] = 1;

			cs.getQueue().enqueueReadImage(*pPing, CL_TRUE, origin, region, 0, 0, &result);
		}

		totalSum += result;
	}

	return totalSum;
}

void HTMRL::learn(sys::ComputeSystem &cs, float columnConnectionAlpha, float cellConnectionAlpha, float tdError, float cellQWeightEligibilityDecay) {
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;
	cl::Image2D* pPrevColumnStates = &_inputImage;

	struct Int2 {
		int _x, _y;
	};

	struct Float2 {
		float _x, _y;
	};

	_qBias += tdError * _qEligibility;
	_qEligibility += -cellQWeightEligibilityDecay * _qEligibility + 1.0f;

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

		int inputDiameter = 2 * _layerDescs[l]._receptiveFieldRadius + 1;

		Float2 inputReceptiveFieldStep;
		inputReceptiveFieldStep._x = inputSizeInv._x * _layerDescs[l]._receptiveFieldRadius / inputDiameter;
		inputReceptiveFieldStep._y = inputSizeInv._y * _layerDescs[l]._receptiveFieldRadius / inputDiameter;

		Int2 layerReceptiveFieldRadius;
		layerReceptiveFieldRadius._x = _layerDescs[l]._receptiveFieldRadius;
		layerReceptiveFieldRadius._y = _layerDescs[l]._receptiveFieldRadius;

		Float2 layerReceptiveFieldStep;
		layerReceptiveFieldStep._x = layerSizeInv._x * _layerDescs[l]._receptiveFieldRadius / inputDiameter;
		layerReceptiveFieldStep._y = layerSizeInv._y * _layerDescs[l]._receptiveFieldRadius / inputDiameter;

		// Column weight update
		_layerColumnWeightUpdateKernel.setArg(0, *pPrevColumnStates);
		_layerColumnWeightUpdateKernel.setArg(1, _layers[l]._columnStates);
		_layerColumnWeightUpdateKernel.setArg(2, _layers[l]._columnWeightsPrev);
		_layerColumnWeightUpdateKernel.setArg(3, _layers[l]._columnWeights);
		_layerColumnWeightUpdateKernel.setArg(4, layerSizeInv);
		_layerColumnWeightUpdateKernel.setArg(5, inputReceptiveFieldRadius);
		_layerColumnWeightUpdateKernel.setArg(6, inputReceptiveFieldStep);
		_layerColumnWeightUpdateKernel.setArg(7, columnConnectionAlpha);

		cs.getQueue().enqueueNDRangeKernel(_layerColumnWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		int layerWidth = _layerDescs[l]._width;

		Int2 lateralConnectionRadii;
		lateralConnectionRadii._x = _layerDescs[l]._lateralConnectionRadius;
		lateralConnectionRadii._y = _layerDescs[l]._lateralConnectionRadius;

		// Lateral weight update
		_layerCellWeightUpdateKernel.setArg(0, _layers[l]._columnStates);
		_layerCellWeightUpdateKernel.setArg(1, _layers[l]._cellStates);
		_layerCellWeightUpdateKernel.setArg(2, _layers[l]._cellStatesPrev);
		_layerCellWeightUpdateKernel.setArg(3, _layers[l]._cellPredictionsPrev);
		_layerCellWeightUpdateKernel.setArg(4, _layers[l]._cellWeightsPrev);
		_layerCellWeightUpdateKernel.setArg(5, _layers[l]._columnPredictionsPrev);
		_layerCellWeightUpdateKernel.setArg(6, _layers[l]._cellWeights);
		_layerCellWeightUpdateKernel.setArg(7, _layerDescs[l]._cellsInColumn);
		_layerCellWeightUpdateKernel.setArg(8, layerWidth);
		_layerCellWeightUpdateKernel.setArg(9, lateralConnectionRadii);
		_layerCellWeightUpdateKernel.setArg(10, cellConnectionAlpha);

		cs.getQueue().enqueueNDRangeKernel(_layerCellWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		// Cell Q weights update
		_layerUpdateQWeightsKernel.setArg(0, _layers[l]._cellStates);
		_layerUpdateQWeightsKernel.setArg(1, _layers[l]._cellQWeightsPrev);
		_layerUpdateQWeightsKernel.setArg(2, _layers[l]._cellQWeights);
		_layerUpdateQWeightsKernel.setArg(3, tdError);
		_layerUpdateQWeightsKernel.setArg(4, cellQWeightEligibilityDecay);
		_layerUpdateQWeightsKernel.setArg(5, _layerDescs[l]._cellsInColumn);

		cs.getQueue().enqueueNDRangeKernel(_layerUpdateQWeightsKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		// Update prevs
		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
		pPrevColumnStates = &_layers[l]._columnStates;
	}
}

void HTMRL::step(sys::ComputeSystem &cs, float reward, float columnConnectionAlpha, float cellConnectionAlpha, float cellQWeightEligibilityDecay, int annealingIterations, float annealingStdDev, float annealingDecay, float alpha, float gamma, float outputBreakChance, float outputPerturbationStdDev, std::mt19937 &generator) {
	_output = _input;

	// Get initial Q
	activate(_output, cs);

	float maxQ = retrieveQ(cs);

	std::vector<float> testOutput(_output.size());

	float perturbationMultiplier = 1.0f;

	std::normal_distribution<float> perturbationDist(0.0f, annealingStdDev);

	for (int i = 0; i < annealingIterations; i++) {
		for (int j = 0; j < _output.size(); j++)
		if (_actionMask[j])
			testOutput[j] = std::min<int>(1.0f, std::max<int>(-1.0f, _output[j] + perturbationMultiplier * perturbationDist(generator)));
		else
			testOutput[j] = _input[j];

		activate(testOutput, cs);

		float result = retrieveQ(cs);

		if (result > maxQ) {
			maxQ = result;

			_output = testOutput;
		}

		perturbationMultiplier *= annealingDecay;
	}

	// Exploratory action
	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
	std::normal_distribution<float> outputPerturbationDist(0.0f, outputPerturbationStdDev);

	for (int j = 0; j < _output.size(); j++)
	if (_actionMask[j]) {
		if (uniformDist(generator) < outputBreakChance)
			_output[j] = uniformDist(generator) * 2.0f - 1.0f;
		else
			_output[j] = std::min<int>(1.0f, std::max<int>(-1.0f, _output[j] + outputPerturbationDist(generator)));
	}
	else
		_output[j] = _input[j];

	activate(_output, cs);

	float exploratoryQ = retrieveQ(cs);

	float tdError = alpha * (reward + gamma * exploratoryQ - _prevQ);

	_prevQ = exploratoryQ;

	learn(cs, columnConnectionAlpha, cellConnectionAlpha, tdError, cellQWeightEligibilityDecay);
}

void HTMRL::exportCellData(sys::ComputeSystem &cs, const std::string &rootName) {
	std::vector<float> state(_layerDescs.back()._width * _layerDescs.back()._height * _layerDescs.back()._cellsInColumn);

	cl::size_t<3> origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;

	cl::size_t<3> region;
	region[0] = _layerDescs.back()._width;
	region[1] = _layerDescs.back()._height;
	region[2] = _layerDescs.back()._cellsInColumn;

	cs.getQueue().enqueueReadImage(_layers.back()._cellStates, CL_TRUE, origin, region, 0, 0, &state[0]);

	// Convert to colors
	for (int ci = 0; ci < _layerDescs.back()._cellsInColumn; ci++) {
		sf::Image image;
		image.create(_layerDescs.back()._width, _layerDescs.back()._height);

		for (int x = 0; x < _layerDescs.back()._width; x++)
		for (int y = 0; y < _layerDescs.back()._height; y++) {
			sf::Color color;

			color = sf::Color::White;

			color.a = color.g = color.b = state[x + y * _layerDescs.back()._width + ci * _layerDescs.back()._width * _layerDescs.back()._height] * 255.0f;

			image.setPixel(x, y, color);
		}

		image.saveToFile(rootName + std::to_string(ci) + ".png");
	}
}