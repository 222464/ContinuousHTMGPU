#pragma once

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <SFML/Graphics.hpp>

#include <htm/AnythingEncoder.h>

#include <vector>

#include <random>

#include <memory>

namespace htm {
	class HTMRL {
	public:
		enum InputType {
			_state, _action, _unused
		};

		struct LayerDesc {
			int _width, _height;

			int _receptiveFieldRadius;
			int _nodeFieldRadius;
			int _lateralConnectionRadius;
			int _inhibitionRadius;

			int _cellsInColumn;

			float _qInfluenceMultiplier;

			float _nodeAlpha;

			LayerDesc()
				: _width(16), _height(16), _receptiveFieldRadius(4), _nodeFieldRadius(4), _lateralConnectionRadius(4), _inhibitionRadius(2), _cellsInColumn(3),
				_qInfluenceMultiplier(1.0f), _nodeAlpha(0.01f)
			{}
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		struct Layer {
			cl::Image2D _columnActivations;
	
			cl::Image2D _columnStatesPrev;
			cl::Image2D _columnStates;

			cl::Image2D _columnPredictionsPrev;
			cl::Image2D _columnPredictions;

			cl::Image2D _columnDutyCyclesPrev;
			cl::Image2D _columnDutyCycles;

			cl::Image3D _columnWeightsPrev;
			cl::Image3D _columnWeights;

			cl::Image3D _cellWeightsPrev;
			cl::Image3D _cellWeights;

			cl::Image3D _cellStatesPrev;
			cl::Image3D _cellStates;

			cl::Image3D _cellPredictionsPrev;
			cl::Image3D _cellPredictions;

			// Nodes are like neurons in a conventional neural network
			cl::Image3D _nodeOutputs;
			cl::Image3D _nodeErrors;

			cl::Image3D _nodeBiasesPrev;
			cl::Image3D _nodeBiases;

			cl::Image3D _nodeWeightsPrev;
			cl::Image3D _nodeWeights;
		};

		int _inputWidth, _inputHeight;

		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		cl::Kernel _layerColumnActivateKernel;
		cl::Kernel _layerColumnInhibitKernel;
		cl::Kernel _layerColumnDutyCycleUpdateKernel;
		cl::Kernel _layerCellActivateKernel;
		cl::Kernel _layerCellWeightUpdateKernel;
		cl::Kernel _layerCellWeightUpdateLastKernel;
		cl::Kernel _layerCellPredictKernel;
		cl::Kernel _layerCellPredictLastKernel;
		cl::Kernel _layerColumnWeightUpdateKernel;
		cl::Kernel _layerColumnPredictionKernel;

		// For nodes
		cl::Kernel _layerNodeActivateKernel;
		cl::Kernel _layerNodeActivateFirstKernel;
		cl::Kernel _weighOutputKernel;
		cl::Kernel _layerNodeBackpropagateKernel;
		cl::Kernel _layerNodeBackpropagateLastKernel;
		cl::Kernel _layerNodeBackpropagateToInputKernel;
		cl::Kernel _layerNodeWeightUpdateKernel;
		cl::Kernel _layerNodeWeightUpdateFirstKernel;
		cl::Kernel _layerNodeWeightUpdateLastKernel;

		// For reconstruction
		cl::Kernel _reconstructInputKernel;

		std::vector<float> _input;

		std::vector<InputType> _inputTypes;

		std::vector<float> _output;
		std::vector<float> _exploratoryOutput;
		std::vector<float> _prevOutput;
		std::vector<float> _prevOutputExploratory;
		std::vector<float> _prevInput;

		float _prevMaxQ;
		float _prevValue;
		float _prevPrevValue;
		float _prevQ;
		float _prevTDError;

		cl::Image2D _inputImage;
		cl::Image2D _inputErrors;

		cl::Image3D _outputWeightsPrev;
		cl::Image3D _outputWeights;

		cl::Image2D _partialSums;

		cl::Image2D _reconstruction;

		float _outputBias;

		void stepBegin();

		void activate(std::vector<float> &input, sys::ComputeSystem &cs, unsigned long seed);
	
		void learnSpatial(sys::ComputeSystem &cs, float columnConnectionAlpha, float widthAlpha, unsigned long seed);
		
		void learnTemporal(sys::ComputeSystem &cs, float cellConnectionAlpha, float cellWeightEligibilityDecay, unsigned long seed);

		void learnSpatialTemporal(sys::ComputeSystem &cs, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellWeightEligibilityDecay, unsigned long seed);

		void dutyCycleUpdate(sys::ComputeSystem &cs, float activationDutyCycleDecay, float stateDutyCycleDecay);

		float getQ(sys::ComputeSystem &cs);
		void nodeActivate(std::vector<float> &input, sys::ComputeSystem &cs);
		void backpropagateToInputs(sys::ComputeSystem &cs, float qError, std::vector<float> &inputs);
		void backpropagate(sys::ComputeSystem &cs, float qError);
		void nodeLearn(sys::ComputeSystem &cs, float qError, float outputAlpha, float eligibilityDecay);

		void initLayer(sys::ComputeSystem &cs, cl::Kernel &initPartOneKernel, cl::Kernel &initPartTwoKernel, cl::Kernel &initPartThreeKernel, int inputWidth, int inputHeight, int inputCellsPerColumn, Layer &layer, const LayerDesc &layerDesc, bool isTopmost, float minInitWeight, float maxInitWeight, float minInitWidth, float maxInitWidth, std::mt19937 &generator);
		void activateLayer(sys::ComputeSystem &cs, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void predictLayer(sys::ComputeSystem &cs, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void predictLayerLast(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void learnLayerSpatial(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, std::mt19937 &generator);
		void learnLayerTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float cellConnectionAlpha, float cellWeightEligibilityDecay, std::mt19937 &generator);
		void learnLayerTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float cellConnectionAlpha, float cellWeightEligibilityDecay, std::mt19937 &generator);
		void learnLayerSpatialTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellWeightEligibilityDecay, std::mt19937 &generator);
		void learnLayerSpatialTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellWeightEligibilityDecay, std::mt19937 &generator);
		void dutyCycleLayerUpdate(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, float activationDutyCycleDecay, float stateDutyCycleDecay);
		
		// Node functions
		void layerNodeActivate(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, const LayerDesc &inputDesc, cl::Image3D &inputImage);
		void layerNodeActivateFirst(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, cl::Image2D &inputImage);
		void layerNodeBackpropagate(sys::ComputeSystem &cs, Layer &layer, Layer &nextLayer, const LayerDesc &layerDesc, const LayerDesc &nextDesc);
		void layerNodeBackpropagateLast(sys::ComputeSystem &cs, float qError);
		void layerNodeBackpropagateToInput(sys::ComputeSystem &cs);
		void layerNodeWeightUpdate(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, const LayerDesc &inputDesc, cl::Image3D &inputImage, float alpha, float eligibilityDecay);
		void layerNodeWeightUpdateFirst(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, cl::Image2D &inputImage, float alpha, float eligibilityDecay);
		void layerNodeWeightUpdateLast(sys::ComputeSystem &cs, float qError, float alpha, float eligibilityDecay);

		// Reconstruction
		void getReconstruction(std::vector<float> &reconstruction, sys::ComputeSystem &cs);
		void getReconstructedPrediction(std::vector<float> &prediction, sys::ComputeSystem &cs);

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, float minInitWidth, float maxInitWidth, std::mt19937 &generator);
	
		void step(sys::ComputeSystem &cs, float reward, float outputAlpha, float nodeEligibilityDecay, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellWeightEligibilityDecay, float cellQWeightEligibilityDecay, float activationDutyCycleDecay, float stateDutyCycleDecay, float reconstructionAlpha, float qBiasAlpha, int deriveMaxQIterations, float deriveMaxQAlpha, float deriveMaxQError, float deriveQMutationStdDev, float deriveMaxQMutationDecay, float alpha, float gamma, float tauInv, float breakChance, float perturbationStdDev, float maxTdError, std::mt19937 &generator);

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
			return _exploratoryOutput[i];
		}

		float getOutput(int x, int y) const {
			return getOutput(x + y * _inputWidth);
		}

		void exportCellData(sys::ComputeSystem &cs, std::vector<std::shared_ptr<sf::Image>> &images, unsigned long seed) const;
	};
}