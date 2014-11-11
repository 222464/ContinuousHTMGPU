#pragma once

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <SFML/Graphics.hpp>

#include <vector>

#include <random>

#include <memory>

#define HTMRL_GPU_Q_SUMMATION 0

namespace htm {
	class HTMRL {
	public:
		struct LayerDesc {
			int _width, _height;

			int _receptiveFieldRadius;
			int _lateralConnectionRadius;
			int _inhibitionRadius;

			int _cellsInColumn;

			LayerDesc()
				: _width(16), _height(16), _receptiveFieldRadius(3), _lateralConnectionRadius(5), _inhibitionRadius(6), _cellsInColumn(4)
			{}
		};
	private:
		struct Layer {
			cl::Image3D _columnWeightsPrev;
			cl::Image3D _columnWeights;

			cl::Image2D _columnActivations;

			cl::Image2D _columnStatesPrev;
			cl::Image2D _columnStates;

			cl::Image3D _cellWeightsPrev;
			cl::Image3D _cellWeights;

			cl::Image3D _cellStatesPrev;
			cl::Image3D _cellStates;

			cl::Image3D _cellPredictionsPrev;
			cl::Image3D _cellPredictions;

			cl::Image2D _columnPredictionsPrev;
			cl::Image2D _columnPredictions;

			cl::Image3D _cellQWeightsPrev;
			cl::Image3D _cellQWeights;

			cl::Image2D _columnOutputsPrev;
			cl::Image2D _columnOutputs;
		};

		int _inputWidth, _inputHeight;

		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		cl::Kernel _layerColumnActivateKernel;
		cl::Kernel _layerColumnInhibitKernel;
		cl::Kernel _layerCellActivateKernel;
		cl::Kernel _layerCellWeightUpdateKernel;
		cl::Kernel _layerCellWeightUpdateLastKernel;
		cl::Kernel _layerCellPredictKernel;
		cl::Kernel _layerCellPredictLastKernel;
		cl::Kernel _layerColumnWeightUpdateKernel;
		cl::Kernel _layerColumnPredictionKernel;
		cl::Kernel _layerColumnOutputKernel;
		cl::Kernel _layerRetrievePartialQSumsKernel;
		cl::Kernel _layerDownsampleKernel;
		cl::Kernel _layerUpdateQWeightsKernel;

		cl::Kernel _reconstructInputKernel;
		cl::Kernel _updateReconstructionKernel;

		std::vector<float> _input;

		std::vector<bool> _actionMask;

		std::vector<float> _output;
		std::vector<float> _prevPrediction;
		std::vector<float> _prevInput;

		float _qBias;
		float _qEligibility;
		float _prevQ;
		float _prevValue;

		cl::Image2D _inputImage;

		cl::Image2D _qSummationBuffer;
		cl::Image2D _halfQSummationBuffer;

		cl::Image3D _reconstructionWeightsPrev;
		cl::Image3D _reconstructionWeights;
		cl::Image2D _reconstruction;

		int _reconstructionReceptiveRadius;

		void stepBegin();

		void activate(std::vector<float> &input, sys::ComputeSystem &cs, std::mt19937 &generator);

		float retrieveQ(sys::ComputeSystem &cs);

		void learn(sys::ComputeSystem &cs, float columnConnectionAlpha, float columnWidthAlpha, float cellConnectionAlpha, float reconstructionAlpha, float tdError, float cellQWeightEligibilityDecay);

		void getReconstructedPrediction(std::vector<float> &prediction, sys::ComputeSystem &cs);

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<bool> &actionMask, float minInitWeight, float maxInitWeight, float minInitWidth, float maxInitWidth, std::mt19937 &generator);
	
		void step(sys::ComputeSystem &cs, float reward, float columnConnectionAlpha, float columnWidthAlpha, float cellConnectionAlpha, float reconstructionAlpha, float cellQWeightEligibilityDecay, int annealingIterations, float annealingStdDev, float annealingDecay, float alpha, float gamma, float tauInv, float outputBreakChance, float outputPerturbationStdDev, std::mt19937 &generator);

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

		void exportCellData(sys::ComputeSystem &cs, std::vector<std::shared_ptr<sf::Image>> &images, unsigned long seed) const;
	};
}