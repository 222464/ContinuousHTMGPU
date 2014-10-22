#include <htm/HTMFeatureExtractor.h>

using namespace htm;

void HTMFeatureExtractor::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, float minInitWeight, float maxInitWeight, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;
	
	_layerDescs = layerDescs;
	
	_layers.resize(_layerDescs.size());

	cl::Kernel initKernel = cl::Kernel(program.getProgram(), "weightInit");

	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;

	_input.clear();
	_input.assign(_inputWidth * _inputHeight, 0.0f);

	_output.clear();
	_output.assign(_layerDescs.back()._width * _layerDescs.back()._height, 0.0f);

	_inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

	std::uniform_int_distribution<int> uniformDist(0, 10000);

	for (int l = 0; l < _layers.size(); l++) {
		_layers[l]._columnActivations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._columnStates = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		int receptiveFieldSize = std::pow(_layerDescs[l]._receptiveFieldRadius * 2 + 1, 2);

		_layers[l]._columnWeights = cl::Image2DArray(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), receptiveFieldSize, _layerDescs[l]._width, _layerDescs[l]._height, 0, 0);
		_layers[l]._columnWeightsPrev = cl::Image2DArray(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), receptiveFieldSize, _layerDescs[l]._width, _layerDescs[l]._height, 0, 0);

		struct Uint2 {
			unsigned int x, y;
		};

		Uint2 seed;
		seed.x = uniformDist(generator);
		seed.y = uniformDist(generator);

		initKernel.setArg(0, _layers[l]._columnStates);
		initKernel.setArg(1, _layers[l]._columnWeights);
		initKernel.setArg(2, receptiveFieldSize);
		initKernel.setArg(3, seed);
		initKernel.setArg(4, minInitWeight);
		initKernel.setArg(5, maxInitWeight);

		cs.getQueue().enqueueNDRangeKernel(initKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _layerDescs[l]._width;
		region[1] = _layerDescs[l]._height;
		region[2] = receptiveFieldSize;

		cs.getQueue().enqueueCopyImage(_layers[l]._columnWeights, _layers[l]._columnWeightsPrev, origin, origin, region);

		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
	}

	_layerActivateKernel = cl::Kernel(program.getProgram(), "layerActivate");

	_layerInhibitKernel = cl::Kernel(program.getProgram(), "layerInhibit");

	_layerWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerWeightUpdate");
}

void HTMFeatureExtractor::step(sys::ComputeSystem &cs, float alpha) {
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

		cs.getQueue().enqueueWriteImage(_inputImage, CL_TRUE, origin, region, 0, 0, &_input[0]);
	}

	// First layer
	{
		int l = 0;

		struct Float2 {
			float x, y;
		};

		Float2 inputSizeInv;
		inputSizeInv.x = 1.0f / _inputWidth;
		inputSizeInv.y = 1.0f / _inputHeight;

		Float2 layerSizeInv;
		layerSizeInv.x = 1.0f / _layerDescs[l]._width;
		layerSizeInv.y = 1.0f / _layerDescs[l]._height;

		Float2 inputReceptiveFieldRadius;
		inputReceptiveFieldRadius.x = inputSizeInv.x * _layerDescs[l]._receptiveFieldRadius;
		inputReceptiveFieldRadius.y = inputSizeInv.y * _layerDescs[l]._receptiveFieldRadius;

		int inputDiameter = 2 * _layerDescs[l]._receptiveFieldRadius + 1;

		Float2 inputReceptiveFieldStep;
		inputReceptiveFieldStep.x = inputReceptiveFieldRadius.x / inputDiameter;
		inputReceptiveFieldStep.y = inputReceptiveFieldRadius.y / inputDiameter;

		// Activation
		_layerActivateKernel.setArg(0, _inputImage);
		_layerActivateKernel.setArg(1, _layers[l]._columnActivations);
		_layerActivateKernel.setArg(2, _layers[l]._columnWeightsPrev);
		_layerActivateKernel.setArg(3, layerSizeInv);
		_layerActivateKernel.setArg(4, inputReceptiveFieldRadius);
		_layerActivateKernel.setArg(5, inputReceptiveFieldStep);

		cs.getQueue().enqueueNDRangeKernel(_layerActivateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		Float2 layerReceptiveFieldRadius;
		layerReceptiveFieldRadius.x = layerSizeInv.x * _layerDescs[l]._receptiveFieldRadius;
		layerReceptiveFieldRadius.y = layerSizeInv.y * _layerDescs[l]._receptiveFieldRadius;

		Float2 layerReceptiveFieldStep;
		layerReceptiveFieldStep.x = layerReceptiveFieldRadius.x / inputDiameter;
		layerReceptiveFieldStep.y = layerReceptiveFieldRadius.y / inputDiameter;

		// Inhibition
		_layerInhibitKernel.setArg(0, _layers[l]._columnActivations);
		_layerInhibitKernel.setArg(1, _layers[l]._columnStates);
		_layerInhibitKernel.setArg(2, layerSizeInv);
		_layerInhibitKernel.setArg(3, layerReceptiveFieldRadius);
		_layerInhibitKernel.setArg(4, layerReceptiveFieldStep);

		cs.getQueue().enqueueNDRangeKernel(_layerInhibitKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		_layerWeightUpdateKernel.setArg(0, _inputImage);
		_layerWeightUpdateKernel.setArg(1, _layers[l]._columnStates);
		_layerWeightUpdateKernel.setArg(2, _layers[l]._columnWeightsPrev);
		_layerWeightUpdateKernel.setArg(3, _layers[l]._columnWeights);
		_layerWeightUpdateKernel.setArg(4, layerSizeInv);
		_layerWeightUpdateKernel.setArg(5, inputReceptiveFieldRadius);
		_layerWeightUpdateKernel.setArg(6, inputReceptiveFieldStep);
		_layerWeightUpdateKernel.setArg(7, alpha);

		cs.getQueue().enqueueNDRangeKernel(_layerWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		std::swap(_layers[l]._columnWeights, _layers[l]._columnWeightsPrev);
	}

	// All other layers
	for (int l = 1; l < _layers.size(); l++) {
		struct Float2 {
			float x, y;
		};

		Float2 inputSizeInv;
		inputSizeInv.x = 1.0f / _layerDescs[l - 1]._width;
		inputSizeInv.y = 1.0f / _layerDescs[l - 1]._height;

		Float2 layerSizeInv;
		layerSizeInv.x = 1.0f / _layerDescs[l]._width;
		layerSizeInv.y = 1.0f / _layerDescs[l]._height;

		Float2 inputReceptiveFieldRadius;
		inputReceptiveFieldRadius.x = inputSizeInv.x * _layerDescs[l]._receptiveFieldRadius;
		inputReceptiveFieldRadius.y = inputSizeInv.y * _layerDescs[l]._receptiveFieldRadius;

		int inputDiameter = 2 * _layerDescs[l]._receptiveFieldRadius + 1;

		Float2 inputReceptiveFieldStep;
		inputReceptiveFieldStep.x = inputReceptiveFieldRadius.x / inputDiameter;
		inputReceptiveFieldStep.y = inputReceptiveFieldRadius.y / inputDiameter;

		// Activation
		_layerActivateKernel.setArg(0, _layers[l - 1]._columnStates);
		_layerActivateKernel.setArg(1, _layers[l]._columnActivations);
		_layerActivateKernel.setArg(2, _layers[l]._columnWeightsPrev);
		_layerActivateKernel.setArg(3, layerSizeInv);
		_layerActivateKernel.setArg(4, inputReceptiveFieldRadius);
		_layerActivateKernel.setArg(5, inputReceptiveFieldStep);

		cs.getQueue().enqueueNDRangeKernel(_layerActivateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		Float2 layerReceptiveFieldRadius;
		layerReceptiveFieldRadius.x = layerSizeInv.x * _layerDescs[l]._receptiveFieldRadius;
		layerReceptiveFieldRadius.y = layerSizeInv.y * _layerDescs[l]._receptiveFieldRadius;

		Float2 layerReceptiveFieldStep;
		layerReceptiveFieldStep.x = layerReceptiveFieldRadius.x / inputDiameter;
		layerReceptiveFieldStep.y = layerReceptiveFieldRadius.y / inputDiameter;

		// Inhibition
		_layerInhibitKernel.setArg(0, _layers[l]._columnActivations);
		_layerInhibitKernel.setArg(1, _layers[l]._columnStates);
		_layerInhibitKernel.setArg(2, layerSizeInv);
		_layerInhibitKernel.setArg(3, layerReceptiveFieldRadius);
		_layerInhibitKernel.setArg(4, layerReceptiveFieldStep);

		cs.getQueue().enqueueNDRangeKernel(_layerInhibitKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		_layerWeightUpdateKernel.setArg(0, _layers[l - 1]._columnStates);
		_layerWeightUpdateKernel.setArg(1, _layers[l]._columnStates);
		_layerWeightUpdateKernel.setArg(2, _layers[l]._columnWeightsPrev);
		_layerWeightUpdateKernel.setArg(3, _layers[l]._columnWeights);
		_layerWeightUpdateKernel.setArg(4, layerSizeInv);
		_layerWeightUpdateKernel.setArg(5, inputReceptiveFieldRadius);
		_layerWeightUpdateKernel.setArg(6, inputReceptiveFieldStep);
		_layerWeightUpdateKernel.setArg(7, alpha);

		cs.getQueue().enqueueNDRangeKernel(_layerWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		std::swap(_layers[l]._columnWeights, _layers[l]._columnWeightsPrev);
	}

	// Get output
	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _layerDescs.back()._width;
		region[1] = _layerDescs.back()._height;
		region[2] = 1;

		cs.getQueue().enqueueReadImage(_layers.back()._columnStates, CL_TRUE, origin, region, 0, 0, &_output[0]);
	}

	cs.getQueue().finish();
}