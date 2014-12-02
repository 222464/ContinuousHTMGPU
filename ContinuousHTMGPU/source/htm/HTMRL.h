#pragma once

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <SFML/Graphics.hpp>

#include <vector>

#include <random>

#include <memory>

namespace htm {
	class HTMRL {
	public:
		struct LayerDesc {
			int _width, _height;

			int _receptiveFieldRadius;
			int _lateralConnectionRadius;
			int _inhibitionRadius;

			int _cellsInColumn;

			float _qInfluenceMultiplier;

			LayerDesc()
				: _width(16), _height(16), _receptiveFieldRadius(4), _lateralConnectionRadius(4), _inhibitionRadius(6), _cellsInColumn(4), _qInfluenceMultiplier(1.0f)
			{}
		};
	private:
		struct Layer {
			cl::Image3D _columnWeightsPrev;
			cl::Image3D _columnWeights;

			cl::Image2D _columnDutyCyclesPrev;
			cl::Image2D _columnDutyCycles;

			cl::Image2D _columnActivations;

			cl::Image2D _columnStatesPrev;
			cl::Image2D _columnStates;

			cl::Image2D _columnPredictionsExploratoryPrev;
			cl::Image2D _columnPredictionsExploratory;

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

			cl::Image2D _partialQSums;
		};

		int _inputWidth, _inputHeight;

		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		cl::Kernel _layerColumnActivateKernel;
		cl::Kernel _layerColumnInhibitKernel;
		cl::Kernel _layerExploreKernel;
		cl::Kernel _layerColumnDutyCycleUpdateKernel;
		cl::Kernel _layerCellActivateKernel;
		cl::Kernel _layerCellWeightUpdateKernel;
		cl::Kernel _layerCellWeightUpdateLastKernel;
		cl::Kernel _layerCellPredictKernel;
		cl::Kernel _layerCellPredictLastKernel;
		cl::Kernel _layerColumnWeightUpdateKernel;
		cl::Kernel _layerColumnPredictionKernel;
		cl::Kernel _layerRetrieveQKernel;
		cl::Kernel _layerUpdateQWeightsKernel;

		cl::Kernel _reconstructInputKernel;

		std::vector<float> _input;

		std::vector<bool> _actionMask;

		std::vector<float> _output;
		std::vector<float> _exploratoryOutput;
		std::vector<float> _prevOutput;
		std::vector<float> _prevOutputExploratory;
		std::vector<float> _prevInput;

		float _qBias;
		float _prevMaxQ;
		float _prevValue;

		cl::Image2D _inputImage;

		cl::Image2D _reconstruction;

		int _reconstructionReceptiveRadius;

		void stepBegin();

		void activate(std::vector<float> &input, sys::ComputeSystem &cs, unsigned long seed);

		float retrieveQ(sys::ComputeSystem &cs);

		void learnSpatialTemporal(sys::ComputeSystem &cs, float columnConnectionAlpha, float cellConnectionAlpha, bool learnExploratory, unsigned long seed);
		
		void updateQWeights(sys::ComputeSystem &cs, float tdError, float cellQWeightEligibilityDecay, float qBiasAlpha);

		void getReconstructedPrediction(std::vector<float> &prediction, sys::ComputeSystem &cs);
		void getReconstructedExploratoryPrediction(std::vector<float> &prediction, sys::ComputeSystem &cs);

		void dutyCycleUpdate(sys::ComputeSystem &cs, float activationDutyCycleDecay, float stateDutyCycleDecay);

		void explorePrediction(sys::ComputeSystem &cs, float mutateChance, unsigned long seed);

		void initLayer(sys::ComputeSystem &cs, cl::Kernel &initPartOneKernel, cl::Kernel &initPartTwoKernel, int inputWidth, int inputHeight, Layer &layer, const LayerDesc &layerDesc, bool isTopmost, float minInitWeight, float maxInitWeight, std::mt19937 &generator);
		void activateLayer(sys::ComputeSystem &cs, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void predictLayer(sys::ComputeSystem &cs, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void predictLayerLast(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		float retreiveLayerQ(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc);
		void learnLayerSpatialTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float cellConnectionAlpha, bool learnExploratory, std::mt19937 &generator);
		void learnLayerSpatialTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float cellConnectionAlpha, bool learnExploratory, std::mt19937 &generator);
		void updateLayerQWeights(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, float tdError, float cellQWeightEligibilityDecay);
		void dutyCycleLayerUpdate(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, float activationDutyCycleDecay, float stateDutyCycleDecay);

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<bool> &actionMask, float minInitWeight, float maxInitWeight, std::mt19937 &generator);
	
		void step(sys::ComputeSystem &cs, float reward, float columnConnectionAlpha, float cellConnectionAlpha, float cellQWeightEligibilityDecay, float activationDutyCycleDecay, float stateDutyCycleDecay, float qBiasAlpha, int annealingIterations, float annealingStdDev, float annealingBreakChance, float annealingDecay, float annealingMomentum, float alpha, float gamma, float tauInv, float mutateChance, std::mt19937 &generator);

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
			return getOutput(x + y * _inputWidth);
		}

		void exportCellData(sys::ComputeSystem &cs, std::vector<std::shared_ptr<sf::Image>> &images, unsigned long seed) const;
	};
}