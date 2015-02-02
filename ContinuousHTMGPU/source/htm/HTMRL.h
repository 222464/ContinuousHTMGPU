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
			int _lateralConnectionRadius;
			int _inhibitionRadius;
			int _dutyCycleRadius; // Should be about 2 * _inhibitionRadius

			int _cellsInColumn;

			float _qInfluenceMultiplier;

			float _noMatchTolerance;

			int _numColumnStateBlurPasses;
			float _columnStateBlurKernelWidthMultiplier;

			int _numTdErrorBlurPasses;
			float _tdErrorBlurKernelWidthMultiplier;

			int _columnQRadius;

			LayerDesc()
				: _width(16), _height(16), _receptiveFieldRadius(4), _lateralConnectionRadius(4), _inhibitionRadius(3), _dutyCycleRadius(4), _cellsInColumn(5),
				_qInfluenceMultiplier(1.0f), _noMatchTolerance(0.2f), _numColumnStateBlurPasses(1), _columnStateBlurKernelWidthMultiplier(0.125f), _numTdErrorBlurPasses(4), _tdErrorBlurKernelWidthMultiplier(1.0f), _columnQRadius(5)
			{}
		};

		struct InputData {
			float _exploratory;
			float _maximum;

			InputData()
				: _exploratory(0.0f), _maximum(0.0f)
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

			cl::Image3D _cellQValuesPrev;
			cl::Image3D _cellQValues;

			cl::Image2D _columnQValues;

			cl::Image2D _columnPrevValues;
			cl::Image2D _columnPrevValuesPrev;

			// Contains just tdError
			cl::Image2D _columnTdErrors;

			cl::Image3D _cellPredictionsPrev;
			cl::Image3D _cellPredictions;

			cl::Image2D _blurPing;
			cl::Image2D _blurPong;
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
		cl::Kernel _layerColumnQKernel;
		cl::Kernel _layerTdErrorKernel;
		cl::Kernel _layerAssignQKernel;

		// For blur
		cl::Kernel _gaussianBlurXKernel;
		cl::Kernel _gaussianBlurYKernel;

		// For reconstruction
		cl::Kernel _reconstructInputKernel;

		std::vector<InputData> _input;

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

		cl::Image2D _reconstruction;

		void stepBegin();

		void activate(std::vector<InputData> &input, sys::ComputeSystem &cs, float reward, float alpha, float gamma, float cellStateDecay, float activationDutyCycleDecay, float stateDutyCycleDecay, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, unsigned long seed);
	
		void learnSpatial(sys::ComputeSystem &cs, float columnConnectionAlpha, float widthAlpha, unsigned long seed);
		
		void learnTemporal(sys::ComputeSystem &cs, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, unsigned long seed);

		void dutyCycleUpdate(sys::ComputeSystem &cs, float activationDutyCycleDecay, float stateDutyCycleDecay);

		void initLayer(sys::ComputeSystem &cs, cl::Kernel &initPartOneKernel, cl::Kernel &initPartTwoKernel, int inputWidth, int inputHeight, int inputCellsPerColumn, Layer &layer, const LayerDesc &layerDesc, bool isTopmost, float minInitWeight, float maxInitWeight, float minInitCenter, float maxInitCenter, float minInitWidth, float maxInitWidth, std::mt19937 &generator);
		void activateLayer(sys::ComputeSystem &cs, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, Layer &layer, const LayerDesc &layerDesc, float cellStateDecay, std::mt19937 &generator);
		void predictLayer(sys::ComputeSystem &cs, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void predictLayerLast(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void determineLayerColumnQ(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc);
		void assignLayerQ(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc, float alpha);
		void determineLayerTdError(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc, float reward, float alpha, float gamma);
		void learnLayerSpatial(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, std::mt19937 &generator);
		void learnLayerTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, std::mt19937 &generator);
		void learnLayerTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, std::mt19937 &generator);
		void dutyCycleLayerUpdate(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, float activationDutyCycleDecay, float stateDutyCycleDecay);
		
		// Reconstruction
		void getReconstruction(std::vector<float> &reconstruction, sys::ComputeSystem &cs);
		void getReconstructedPrediction(std::vector<float> &prediction, sys::ComputeSystem &cs);
		void getReconstructedPrevPrediction(std::vector<float> &prediction, sys::ComputeSystem &cs);

		// Blur
		void gaussianBlur(sys::ComputeSystem &cs, cl::Image2D &source, cl::Image2D &ping, cl::Image2D &pong, int imageSizeX, int imageSizeY, int passes, float kernelWidth);

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, float minInitCenter, float maxInitCenter, std::mt19937 &generator);
	
		void step(sys::ComputeSystem &cs, float reward, float cellStateDecay, float columnConnectionAlpha, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, float activationDutyCycleDecay, float stateDutyCycleDecay, float reconstructionAlpha, float alpha, float gamma, float tauInv, float breakChance, float perturbationStdDev, float maxTdError, std::mt19937 &generator);

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
			_input[i]._exploratory = _input[i]._maximum = value;
		}

		void setInput(int x, int y, float value) {
			setInput(x + y * _inputWidth, value);
		}

		float getOutput(int i) const {
			return _input[i]._exploratory;
		}

		float getOutput(int x, int y) const {
			return getOutput(x + y * _inputWidth);
		}

		void exportCellData(sys::ComputeSystem &cs, std::vector<std::shared_ptr<sf::Image>> &images, unsigned long seed) const;
	};
}