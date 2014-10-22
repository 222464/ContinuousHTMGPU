#pragma once

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <vector>

#include <random>

#include <memory>

namespace cae {
	class ConvAutoEncoder {
	public:
		struct LayerDesc {
			int _convKernelWidth, _convKernelHeight;
			//int _convKernelStrideWidth, _convKernelStrideHeight;

			int _mapWidth, _mapHeight;

			int _numMaps;

			LayerDesc()
				: _convKernelWidth(6), _convKernelHeight(6),
				//_convKernelStrideWidth(1), _convKernelStrideHeight(1),
				_mapWidth(32), _mapHeight(32),
				_numMaps(1)
			{}
		};
	private:
		struct Layer {
			int _numWeights; // Includes bias

			cl::Image2D _mapWeights;
			cl::Image2D _newMapWeights;

			cl::Image2DArray _prevLayerBiases;
			cl::Image2DArray _newPrevLayerBiases;

			cl::Image2DArray _prevLayerErrors;

			cl::Image2DArray _mapOutputs;
		};

		int _inputWidth, _inputHeight, _inputNumMaps;

		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		cl::Image2DArray _reduceImagePing;
		cl::Image2DArray _reduceImagePong;

		cl::Kernel _layerActivateForwardKernel;
		cl::Kernel _layerActivateBackwardKernel;
		cl::Kernel _layerWeightUpdateKernel;
		cl::Kernel _weightDeltaReduceKernel;
		cl::Kernel _mapsDeltaUpdateKernel;
		//cl::Kernel _layerDownsampleKernel;
	
		std::vector<float> _input;
		std::vector<float> _output;

		cl::Image2DArray _inputMaps;

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, int inputNumMaps, const std::vector<LayerDesc> &layerDescs, float minInitWeight, float maxInitWeight, std::mt19937 &generator);

		void step(sys::ComputeSystem &cs, float alpha);

		int getInputWidth() const {
			return _inputWidth;
		}

		int getInputHeight() const {
			return _inputHeight;
		}

		int getInputNumMaps() const {
			return _inputNumMaps;
		}

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		int getOutputWidth() const {
			return _layerDescs.back()._mapWidth;
		}

		int getOutputHeight() const {
			return _layerDescs.back()._mapHeight;
		}

		int getOutputNumMaps() const {
			return _layerDescs.back()._numMaps;
		}

		void setInput(int i, float value) {
			_input[i] = value;
		}

		void setInput(int x, int y, int m, float value) {
			setInput(x + y * _inputWidth + m * _inputWidth * _inputHeight, value);
		}

		float getOutput(int i) const {
			return _output[i];
		}

		float getOutput(int x, int y, int m) const {
			return getOutput(x + y * getOutputWidth() + m * getOutputWidth() * getOutputHeight());
		}
	};
}