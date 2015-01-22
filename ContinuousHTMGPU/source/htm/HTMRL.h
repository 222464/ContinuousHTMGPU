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
			_state, _action, _q, _unused
		};

		struct LayerDesc {
			int _width, _height;

			int _receptiveFieldRadius;
			int _nodeFieldRadius;
			int _lateralConnectionRadius;
			int _inhibitionRadius;
			int _dutyCycleRadius; // Should be about 2 * _inhibitionRadius

			int _cellsInColumn;

			float _qInfluenceMultiplier;

			float _nodeAlpha;

			float _noMatchTolerance;

			int _numBlurPasses;
			float _blurKernelWidthMuliplier;

			LayerDesc()
				: _width(16), _height(16), _receptiveFieldRadius(4), _nodeFieldRadius(5), _lateralConnectionRadius(5), _inhibitionRadius(4), _dutyCycleRadius(5), _cellsInColumn(6),
				_qInfluenceMultiplier(1.0f), _nodeAlpha(0.1f), _noMatchTolerance(0.01f), _numBlurPasses(1), _blurKernelWidthMuliplier(0.25f)
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

			// Contain both Q and tdError
			cl::Image3D _cellQValuesPrev;
			cl::Image3D _cellQValues;

			// Contains just Q
			cl::Image2D _columnQValuesPrev;
			cl::Image2D _columnQValues;

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
		cl::Kernel _layerColumnTdErrorKernel;
		cl::Kernel _layerAssignQKernel;
		cl::Kernel _layerAssignQLastKernel;

		// For blur
		cl::Kernel _gaussianBlurXKernel;
		cl::Kernel _gaussianBlurYKernel;

		// For reconstruction
		cl::Kernel _reconstructInputKernel;

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

		AnythingEncoder _qEncoder;
		std::vector<int> _qIndices;

		cl::Image2D _inputImage;

		cl::Image2D _reconstruction;

		void stepBegin();

		void activate(std::vector<float> &input, sys::ComputeSystem &cs, float reward, float alpha, float gamma, float cellStateDecay, float activationDutyCycleDecay, float stateDutyCycleDecay, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, unsigned long seed);
	
		void learnSpatial(sys::ComputeSystem &cs, float columnConnectionAlpha, float widthAlpha, unsigned long seed);
		
		void learnTemporal(sys::ComputeSystem &cs, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, unsigned long seed);

		void dutyCycleUpdate(sys::ComputeSystem &cs, float activationDutyCycleDecay, float stateDutyCycleDecay);

		void initLayer(sys::ComputeSystem &cs, cl::Kernel &initPartOneKernel, cl::Kernel &initPartTwoKernel, int inputWidth, int inputHeight, int inputCellsPerColumn, Layer &layer, const LayerDesc &layerDesc, bool isTopmost, float minInitWeight, float maxInitWeight, float minInitCenter, float maxInitCenter, float minInitWidth, float maxInitWidth, std::mt19937 &generator);
		void activateLayer(sys::ComputeSystem &cs, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, Layer &layer, const LayerDesc &layerDesc, float cellStateDecay, std::mt19937 &generator);
		void predictLayer(sys::ComputeSystem &cs, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void predictLayerLast(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void determineLayerColumnQ(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc);
		void assignLayerQ(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc, Layer &nextLayer, LayerDesc &nextLayerDesc, float reward, float alpha, float gamma);
		void assignLayerQLast(sys::ComputeSystem &cs, Layer &layer, LayerDesc &layerDesc, float reward, float alpha, float gamma);
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

		// Q
		float reconstructQFromInput(const std::vector<float> &input);
		void assignInputsFromQ(std::vector<float> &input, float q, float encoderLocalActivity, float encoderOutputIntensity, float encoderDutyCycleDecay);
		void learnQReconstruction(float q, float encoderCenterAlpha, float encoderMaxDutyCycleForLearn, float encoderNoMatchIntensity);

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, float minInitCenter, float maxInitCenter, float minInitEncoderCenter, float maxInitEncoderCenter, std::mt19937 &generator);
	
		void step(sys::ComputeSystem &cs, float reward, float cellStateDecay, float columnConnectionAlpha, float cellConnectionAlpha, float cellConnectionBeta, float cellConnectionTemperature, float cellWeightEligibilityDecay, float encoderAlpha, float encoderLocalActivity, float encoderOutputIntensity, float encoderDutyCycleDecay, float encoderMaxDutyCycleForLearn, float encoderNoMatchIntensity, float activationDutyCycleDecay, float stateDutyCycleDecay, float reconstructionAlpha, float alpha, float gamma, float tauInv, float breakChance, float perturbationStdDev, float maxTdError, std::mt19937 &generator);

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