#pragma once

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <SFML/Graphics.hpp>

#include <htm/AnythingEncoder.h>

#include <vector>
#include <list>

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
			int _lateralConnectionRadius;
			int _inhibitionRadius;

			int _cellsInColumn;

			int _numSegmentsPerCell;

			float _qInfluenceMultiplier;

			int _numColumnStateBlurPasses;
			float _columnStateBlurKernelWidthMultiplier;

			int _columnInfluenceRadius;

			float _qImportance;

			LayerDesc()
				: _width(16), _height(16), _receptiveFieldRadius(3), _lateralConnectionRadius(3), _inhibitionRadius(2), _cellsInColumn(3), _numSegmentsPerCell(4),
				_qInfluenceMultiplier(1.0f), _numColumnStateBlurPasses(1), _columnStateBlurKernelWidthMultiplier(1.0f), _columnInfluenceRadius(5), _qImportance(1.0f)
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

			cl::Image2D _columnStatesProbabilistic;

			cl::Image2D _columnActivationReconstruction;
			cl::Image2D _columnStateReconstruction;

			cl::Image2D _inputBiasesPrev;
			cl::Image2D _inputBiases;

			cl::Image2D _reconstruction;

			cl::Image2D _columnPredictionsPrev;
			cl::Image2D _columnPredictions;

			cl::Image3D _columnFeedForwardWeightsPrev;
			cl::Image3D _columnFeedForwardWeights;

			cl::Image3D _cellWeightsPrev;
			cl::Image3D _cellWeights;

			cl::Image3D _cellStatesPrev;
			cl::Image3D _cellStates;

			cl::Image3D _segmentStatesPrev;
			cl::Image3D _segmentStates;

			//cl::Image3D _segmentWeightsPrev;
			//cl::Image3D _segmentWeights;

			cl::Image3D _cellQValuesPrev;
			cl::Image3D _cellQValues;

			cl::Image2D _columnQValues;

			cl::Image2D _columnPrevValues;
			cl::Image2D _columnPrevValuesPrev;

			// Contains just tdError
			cl::Image2D _columnTdErrors;

			cl::Image3D _cellPredictionsPrev;
			cl::Image3D _cellPredictions;
		};

		int _inputWidth, _inputHeight;

		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		cl::Kernel _layerColumnActivateKernel;
		cl::Kernel _layerColumnInhibitBinaryKernel;
		cl::Kernel _layerColumnInhibitKernel;
		cl::Kernel _layerColumnInhibitProbablisticKernel;
		cl::Kernel _layerCellActivateKernel;
		cl::Kernel _layerCellWeightUpdateKernel;
		cl::Kernel _layerCellWeightUpdateLastKernel;
		cl::Kernel _layerCellPredictKernel;
		cl::Kernel _layerCellPredictLastKernel;
		cl::Kernel _layerColumnWeightUpdateKernel;
		cl::Kernel _layerColumnPredictionKernel;
		cl::Kernel _layerColumnQKernel;
		cl::Kernel _layerColumnQLastKernel;
		cl::Kernel _layerAssignQKernel;

		cl::Kernel _reconstructInputKernel;
		cl::Kernel _inputBiasUpdateKernel;
	
		// For blur
		cl::Kernel _gaussianBlurXKernel;
		cl::Kernel _gaussianBlurYKernel;

		std::vector<float> _input;

		std::vector<InputType> _inputTypes;

		std::vector<float> _output;
		std::vector<float> _prediction;
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
		cl::Image2D _reconstructedPrediction;

		int _addReplaySampleStepCounter;

		std::list<cl::Image2D> _inputReplayChain;

		void stepBegin(sys::ComputeSystem &cs, int addReplaySampleSteps, int maxReplayChainSize);

		void activate(std::vector<float> &input, sys::ComputeSystem &cs, float reward, float alpha, float gamma, float columnDecay, float cellStateDecay, float columnConnectionAlpha, float columnConnectionBeta, float columnConnectionGamma, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionGamma, float cellConnectionTemperature, float cellWeightEligibilityDecay, int maxReplayChainSize, int numReplaySamples, int addSampleSteps, unsigned long seed);
	
		void learnSpatialReplay(sys::ComputeSystem &cs, float cellStateDecay, float alpha, float beta, float gamma, int maxReplayChainSize, int numReplaySamples, unsigned long seed);
		
		void learnTemporal(sys::ComputeSystem &cs, float tdError, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionGamma, float cellConnectionTemperature, float cellWeightEligibilityDecay, unsigned long seed);

		void initLayer(sys::ComputeSystem &cs, cl::Kernel &initPartOneKernel, cl::Kernel &initPartTwoKernel, cl::Kernel &initPartThreeKernel, int inputWidth, int inputHeight, int inputCellsPerColumn, Layer &layer, const LayerDesc &layerDesc, bool isTopmost, float minInitWeight, float maxInitWeight, float minInitCenter, float maxInitCenter, float minInitWidth, float maxInitWidth, std::mt19937 &generator);
		void spatialPoolLayer(sys::ComputeSystem &cs, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, Layer &layer, const LayerDesc &layerDesc, float columnDecay, std::mt19937 &generator);
		void cellActivateLayer(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, float cellStateDecay, std::mt19937 &generator);
		void predictLayer(sys::ComputeSystem &cs, cl::Image2D &nextLayerPrediction, cl::Image2D &nextLayerPredictionPrev, int nextLayerWidth, int nextLayerHeight, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void predictLayerLast(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void determineLayerColumnQ(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc, Layer &nextLayer, LayerDesc &nextLayerDesc);
		void determineLayerColumnQLast(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc);
		void assignLayerQ(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc, float alpha);
		void learnLayerSpatial(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float alpha, float beta, float gamma, std::mt19937 &generator);
		void learnLayerTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float tdError, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionGamma, float cellConnectionTemperature, float cellWeightEligibilityDecay, std::mt19937 &generator);
		void learnLayerTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float tdError, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionGamma, float cellConnectionTemperature, float cellWeightEligibilityDecay, std::mt19937 &generator);
		void dutyCycleLayerUpdate(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, float activationDutyCycleDecay, float stateDutyCycleDecay);

		// Reconstruction
		void getReconstructedPrediction(std::vector<float> &prediction, sys::ComputeSystem &cs);
	
		// Blur
		void gaussianBlur(sys::ComputeSystem &cs, cl::Image2D &source, cl::Image2D &ping, cl::Image2D &pong, int imageSizeX, int imageSizeY, int passes, float kernelWidth);

		// Q
		float retreiveQ(sys::ComputeSystem &cs);

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, int reconstructionReceptiveRadius, const std::vector<LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, float minInitCenter, float maxInitCenter, std::mt19937 &generator);
	
		void step(sys::ComputeSystem &cs, float reward, float reconstructionAlpha, float columnDecay, float cellStateDecay, float columnConnectionAlpha, float columnConnectionBeta, float columnConnectionGamma, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionGamma, float cellConnectionTemperature, float cellWeightEligibilityDecay, float alpha, float gamma, float breakChance, float perturbationStdDev, int maxReplayChainSize, int numReplaySamples, int addReplaySampleSteps, std::mt19937 &generator);

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
			return _input[i];
		}

		float getOutput(int x, int y) const {
			return getOutput(x + y * _inputWidth);
		}

		void exportCellData(sys::ComputeSystem &cs, std::vector<std::shared_ptr<sf::Image>> &images, unsigned long seed) const;
	};
}