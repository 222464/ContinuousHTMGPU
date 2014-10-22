#include <cae/ConvAutoEncoder.h>

#include <algorithm>

using namespace cae;

void ConvAutoEncoder::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, int inputNumMaps, const std::vector<LayerDesc> &layerDescs, float minInitWeight, float maxInitWeight, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;
	_inputNumMaps = inputNumMaps;

	_layerDescs = layerDescs;

	_layers.resize(_layerDescs.size());

	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;

	_input.clear();
	_input.assign(_inputWidth * _inputHeight * _inputNumMaps, 0.0f);

	_inputMaps = cl::Image2DArray(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputNumMaps, _inputWidth, _inputHeight, 0, 0);

	int prevMapWidth = _inputWidth;
	int prevMapHeight = _inputHeight;
	int prevNumMaps = _inputNumMaps;

	std::uniform_real_distribution<float> weightDist(minInitWeight, maxInitWeight);

	int maxNumWeights = 0;

	for (int l = 0; l < _layers.size(); l++) {
		//_layers[l]._mapWidth = (prevMapWidth - _layerDescs[l]._convKernelWidth + 1) / (_layerDescs[l]._convKernelStrideWidth);
		//_layers[l]._mapHeight = (prevMapHeight - _layerDescs[l]._convKernelHeight + 1) / (_layerDescs[l]._convKernelStrideHeight);
		//_layers[l]._downsampledWidth = _layers[l]._mapWidth / _layerDescs[l]._downsampleWidth;
		//_layers[l]._downsampledHeight = _layers[l]._mapHeight / _layerDescs[l]._downsampleHeight;
		_layers[l]._numWeights = _layerDescs[l]._convKernelWidth * _layerDescs[l]._convKernelHeight * prevNumMaps + 1; // + 1 for bias

		maxNumWeights = std::max<float>(maxNumWeights, _layers[l]._numWeights * _layerDescs[l]._numMaps);

		_layers[l]._mapOutputs = cl::Image2DArray(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._numMaps, _layerDescs[l]._mapWidth, _layerDescs[l]._mapHeight, 0, 0);
		//_layers[l]._downsampledOutputs = cl::Image2DArray(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._numMaps, _layers[l]._downsampledWidth, _layers[l]._downsampledHeight, 0, 0);

		cl::size_t<3> origin;
		cl::size_t<3> region;

		std::vector<float> weights(_layerDescs[l]._numMaps * _layers[l]._numWeights);

		for (int i = 0; i < weights.size(); i++)
			weights[i] = weightDist(generator);

		_layers[l]._mapWeights = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layers[l]._numWeights, _layerDescs[l]._numMaps, 0);
		_layers[l]._newMapWeights = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layers[l]._numWeights, _layerDescs[l]._numMaps, 0);

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = _layers[l]._numWeights;
		region[1] = _layerDescs[l]._numMaps;
		region[2] = 1;

		cs.getQueue().enqueueWriteImage(_layers[l]._mapWeights, CL_TRUE, origin, region, 0, 0, &weights[0]);

		_layers[l]._prevLayerBiases = cl::Image2DArray(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevNumMaps, prevMapWidth, prevMapHeight, 0, 0);
		_layers[l]._newPrevLayerBiases = cl::Image2DArray(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevNumMaps, prevMapWidth, prevMapHeight, 0, 0);

		std::vector<float> prevLayerBiases(prevMapWidth * prevMapHeight * prevNumMaps);

		for (int i = 0; i < prevLayerBiases.size(); i++)
			prevLayerBiases[i] = weightDist(generator);

		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		region[0] = prevMapWidth;
		region[1] = prevMapHeight;
		region[2] = prevNumMaps;

		cs.getQueue().enqueueWriteImage(_layers[l]._prevLayerBiases, CL_TRUE, origin, region, 0, 0, &prevLayerBiases[0]);

		_layers[l]._prevLayerErrors = cl::Image2DArray(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevNumMaps, prevMapWidth, prevMapHeight, 0, 0);

		prevMapWidth = _layerDescs[l]._mapWidth;
		prevMapHeight = _layerDescs[l]._mapHeight;
		prevNumMaps = _layerDescs[l]._numMaps;
	}

	_output.clear();
	_output.assign(prevMapWidth * prevMapHeight * prevNumMaps, 0.0f);

	// Make large reduction image that can contain all reductions
	_reduceImagePing = cl::Image2DArray(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), maxNumWeights, _layerDescs.front()._mapWidth, _layerDescs.front()._mapHeight, 0, 0);
	_reduceImagePong = cl::Image2DArray(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), maxNumWeights, _layerDescs.front()._mapWidth, _layerDescs.front()._mapHeight, 0, 0);

	_layerActivateForwardKernel = cl::Kernel(program.getProgram(), "layerActivateForward");
	_layerActivateBackwardKernel = cl::Kernel(program.getProgram(), "layerActivateBackward");
	_layerWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerWeightUpdate");
	_weightDeltaReduceKernel = cl::Kernel(program.getProgram(), "weightDeltaReduce");
	_mapsDeltaUpdateKernel = cl::Kernel(program.getProgram(), "mapsDeltaUpdate");
	//_layerDownsampleKernel = cl::Kernel(program.getProgram(), "layerDownsample");
}

void ConvAutoEncoder::step(sys::ComputeSystem &cs, float alpha) {
	// Create buffer from input
	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = _inputNumMaps;

		cs.getQueue().enqueueWriteImage(_inputMaps, CL_TRUE, origin, region, 0, 0, &_input[0]);
	}

	cl::Image2DArray* pSourceMaps = &_inputMaps;
	int prevMapWidth = _inputWidth;
	int prevMapHeight = _inputHeight;
	int prevNumMaps = _inputNumMaps;

	for (int l = 0; l < _layers.size(); l++) {
		struct Int2 {
			int _x, _y;

			Int2(int x, int y) 
				: _x(x), _y(y)
			{}
		};

		struct Float2 {
			float _x, _y;
			
			Float2(float x, float y)
				: _x(x), _y(y)
			{}
		};

		Float2 layerSizeInv(1.0f / _layerDescs[l]._mapWidth, 1.0f / _layerDescs[l]._mapHeight);
		Float2 prevLayerSizeInv(1.0f / prevMapWidth, 1.0f / prevMapHeight);
		Float2 sizeRatio(static_cast<float>(_layerDescs[l]._mapWidth) / prevMapWidth, static_cast<float>(_layerDescs[l]._mapHeight) / prevMapHeight);

		_layerActivateForwardKernel.setArg(0, *pSourceMaps);
		_layerActivateForwardKernel.setArg(1, _layers[l]._mapWeights);
		_layerActivateForwardKernel.setArg(2, _layers[l]._mapOutputs);
		_layerActivateForwardKernel.setArg(3, Int2(_layerDescs[l]._convKernelWidth, _layerDescs[l]._convKernelHeight));
		_layerActivateForwardKernel.setArg(4, prevNumMaps);
		_layerActivateForwardKernel.setArg(5, layerSizeInv);
		_layerActivateForwardKernel.setArg(6, prevLayerSizeInv);

		cs.getQueue().enqueueNDRangeKernel(_layerActivateForwardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._mapWidth, _layerDescs[l]._mapHeight, _layerDescs[l]._numMaps));
	
		if (alpha != 0.0f) {
			_layerActivateBackwardKernel.setArg(0, _layers[l]._mapOutputs);
			_layerActivateBackwardKernel.setArg(1, *pSourceMaps);
			_layerActivateBackwardKernel.setArg(2, _layers[l]._mapWeights);
			_layerActivateBackwardKernel.setArg(3, _layers[l]._prevLayerBiases);
			_layerActivateBackwardKernel.setArg(4, _layers[l]._newPrevLayerBiases);
			_layerActivateBackwardKernel.setArg(5, _layers[l]._prevLayerErrors);
			_layerActivateBackwardKernel.setArg(6, Int2(_layerDescs[l]._convKernelWidth, _layerDescs[l]._convKernelHeight));
			_layerActivateBackwardKernel.setArg(7, Int2(std::round(_layerDescs[l]._convKernelWidth * sizeRatio._x), std::round(_layerDescs[l]._convKernelHeight * sizeRatio._y)));
			_layerActivateBackwardKernel.setArg(8, _layerDescs[l]._numMaps);
			_layerActivateBackwardKernel.setArg(9, prevNumMaps);
			_layerActivateBackwardKernel.setArg(10, layerSizeInv);
			_layerActivateBackwardKernel.setArg(11, prevLayerSizeInv);
			_layerActivateBackwardKernel.setArg(12, alpha);

			cs.getQueue().enqueueNDRangeKernel(_layerActivateBackwardKernel, cl::NullRange, cl::NDRange(prevMapWidth, prevMapHeight, prevNumMaps));

			_layerWeightUpdateKernel.setArg(0, _layers[l]._mapOutputs);
			_layerWeightUpdateKernel.setArg(1, *pSourceMaps);
			_layerWeightUpdateKernel.setArg(2, _layers[l]._prevLayerErrors);
			_layerWeightUpdateKernel.setArg(3, _layers[l]._mapWeights);
			_layerWeightUpdateKernel.setArg(4, _reduceImagePing);
			_layerWeightUpdateKernel.setArg(5, Int2(_layerDescs[l]._convKernelWidth, _layerDescs[l]._convKernelHeight));
			_layerWeightUpdateKernel.setArg(6, _layerDescs[l]._numMaps);
			_layerWeightUpdateKernel.setArg(7, prevNumMaps);
			_layerWeightUpdateKernel.setArg(8, layerSizeInv);
			_layerWeightUpdateKernel.setArg(9, prevLayerSizeInv);
			_layerWeightUpdateKernel.setArg(10, _layers[l]._numWeights);
			_layerWeightUpdateKernel.setArg(11, alpha);

			cs.getQueue().enqueueNDRangeKernel(_layerWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._mapWidth, _layerDescs[l]._mapHeight, _layerDescs[l]._numMaps));

			// Reduce weight updates
			int w = _layerDescs[l]._mapWidth;
			int h = _layerDescs[l]._mapHeight;

			int totalNumWeightsPerLayer = _layerDescs[l]._numMaps * (_layerDescs[l]._convKernelWidth * _layerDescs[l]._convKernelHeight * prevNumMaps + 1);

			for (; w > 1 || h > 1; w /= 2, h /= 2) {
				_weightDeltaReduceKernel.setArg(0, _reduceImagePing);
				_weightDeltaReduceKernel.setArg(1, _reduceImagePong);
				_weightDeltaReduceKernel.setArg(2, totalNumWeightsPerLayer);
				_weightDeltaReduceKernel.setArg(3, Int2(2, 2));

				cs.getQueue().enqueueNDRangeKernel(_weightDeltaReduceKernel, cl::NullRange, cl::NDRange(w / 2, h / 2));

				std::swap(_reduceImagePing, _reduceImagePong);
			}

			// Apply weight changes
			_mapsDeltaUpdateKernel.setArg(0, _reduceImagePong);
			_mapsDeltaUpdateKernel.setArg(1, _layers[l]._mapWeights);
			_mapsDeltaUpdateKernel.setArg(2, _layers[l]._newMapWeights);
			_mapsDeltaUpdateKernel.setArg(3, _layers[l]._numWeights);

			cs.getQueue().enqueueNDRangeKernel(_mapsDeltaUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._convKernelWidth, _layerDescs[l]._convKernelHeight, _layerDescs[l]._numMaps));

			std::swap(_layers[l]._prevLayerBiases, _layers[l]._newPrevLayerBiases);
			std::swap(_layers[l]._mapWeights, _layers[l]._newMapWeights);
		}

		// Perform max-pooling (downsampling)
		//_layerDownsampleKernel.setArg(0, _layers[l]._mapOutputs);
		//_layerDownsampleKernel.setArg(1, _layers[l]._downsampledOutputs);
		//_layerDownsampleKernel.setArg(2, Int2(_layerDescs[l]._downsampleWidth, _layerDescs[l]._downsampleHeight));

		//cs.getQueue().enqueueNDRangeKernel(_layerDownsampleKernel, cl::NullRange, cl::NDRange(_layers[l]._downsampledWidth, _layers[l]._downsampledHeight, _layerDescs[l]._numMaps));

		pSourceMaps = &_layers[l]._mapOutputs;
		prevMapWidth = _layerDescs[l]._mapWidth;
		prevMapHeight = _layerDescs[l]._mapHeight;
		prevNumMaps = _layerDescs[l]._numMaps;
	}

	// Get output
	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = getOutputWidth();
		region[1] = getOutputHeight();
		region[2] = getOutputNumMaps();

		cs.getQueue().enqueueReadImage(_layers.back()._mapOutputs, CL_TRUE, origin, region, 0, 0, &_output[0]);
	}

	cs.getQueue().finish();
}