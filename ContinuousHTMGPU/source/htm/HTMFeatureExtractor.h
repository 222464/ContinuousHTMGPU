#pragma once

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <vector>

#include <random>

#include <memory>

namespace htm {
	class HTMFeatureExtractor {
	public:
		struct LayerDesc {
			int _width, _height;

			int _receptiveFieldRadius;

			LayerDesc()
				: _width(16), _height(16), _receptiveFieldRadius(2)
			{}
		};
	private:
		struct Layer {
			cl::Image2DArray _columnWeightsPrev;
			cl::Image2DArray _columnWeights;

			cl::Image2D _columnActivations;
			cl::Image2D _columnStates;
		};

		int _inputWidth, _inputHeight;

		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		cl::Kernel _layerActivateKernel;
		cl::Kernel _layerInhibitKernel;
		cl::Kernel _layerWeightUpdateKernel;

		std::vector<float> _input;
		std::vector<float> _output;

		cl::Image2D _inputImage;

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, float minInitWeight, float maxInitWeight, std::mt19937 &generator);
	
		void step(sys::ComputeSystem &cs, float alpha);

		int getInputWidth() const {
			return _inputWidth;
		}

		int getInputHeight() const {
			return _inputHeight;
		}

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		void setInput(int i, float value) {
			_input[i] = value;
		}

		void setInput(int x, int y, float value) {
			setInput(x + y * _inputWidth, value);
		}

		float getOutput(int i) const {
			return _output[i];
		}

		float getOutput(int x, int y) const {
			return getOutput(x + y * _layerDescs.back()._width);
		}
	};
}