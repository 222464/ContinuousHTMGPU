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
			_state, _action, _q
		};

		struct LayerDesc {
			int _width, _height;

			int _receptiveFieldRadius;
			int _lateralConnectionRadius;
			int _inhibitionRadius;

			int _cellsInColumn;

			float _qInfluenceMultiplier;

			LayerDesc()
				: _width(16), _height(16), _receptiveFieldRadius(6), _lateralConnectionRadius(4), _inhibitionRadius(6), _cellsInColumn(6), _qInfluenceMultiplier(1.0f)
			{}
		};
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

			cl::Image3D _cellStatesPrevPrev;
			cl::Image3D _cellStatesPrev;
			cl::Image3D _cellStates;

			cl::Image3D _cellPredictionsPrev;
			cl::Image3D _cellPredictions;
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

		cl::Kernel _reconstructInputKernel;
		cl::Kernel _learnReconstructionKernel;

		std::vector<float> _input;

		std::vector<InputType> _inputTypes;

		AnythingEncoder _qEncoder;
		std::vector<int> _qInputIndices;

		std::vector<float> _output;
		std::vector<float> _exploratoryOutput;
		std::vector<float> _prevOutput;
		std::vector<float> _prevOutputExploratory;
		std::vector<float> _prevInput;

		float _qBias;
		float _prevMaxQ;
		float _prevValue;
		float _prevPrevValue;
		float _prevQ;

		cl::Image2D _inputImage;

		cl::Image3D _reconstructionWeightsPrev;
		cl::Image3D _reconstructionWeights;

		cl::Image2D _reconstruction;

		int _reconstructionReceptiveRadius;

		void stepBegin();

		void activate(std::vector<float> &input, sys::ComputeSystem &cs, unsigned long seed);
	
		void learnSpatial(sys::ComputeSystem &cs, float columnConnectionAlpha, float widthAlpha, unsigned long seed);
		
		void learnTemporal(sys::ComputeSystem &cs, float cellConnectionAlpha, float cellWeightEligibilityDecay, unsigned long seed);

		void learnSpatialTemporal(sys::ComputeSystem &cs, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellWeightEligibilityDecay, unsigned long seed);

		void getReconstruction(std::vector<float> &reconstruction, sys::ComputeSystem &cs);
		void getReconstructedPrediction(std::vector<float> &prediction, sys::ComputeSystem &cs);

		void learnReconstruction(sys::ComputeSystem &cs, float reconstructionAlpha);

		void dutyCycleUpdate(sys::ComputeSystem &cs, float activationDutyCycleDecay, float stateDutyCycleDecay);

		float retrieveQ(const std::vector<float> &sdr);
		void setQ(std::vector<float> &sdr, float q, float localActivity, float outputIntensity, float dutyCycleDecay, float boostThreshold, float boostIntensity);
		void learnQEncoder(float q, float prevQ, float centerAlpha, float widthAlpha, float widthScalar, float minWidth, float reconAlpha);

		void initLayer(sys::ComputeSystem &cs, cl::Kernel &initPartOneKernel, cl::Kernel &initPartTwoKernel, int inputWidth, int inputHeight, Layer &layer, const LayerDesc &layerDesc, bool isTopmost, float minInitWeight, float maxInitWeight, float minInitWidth, float maxInitWidth, std::mt19937 &generator);
		void activateLayer(sys::ComputeSystem &cs, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void predictLayer(sys::ComputeSystem &cs, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void predictLayerLast(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, std::mt19937 &generator);
		void learnLayerSpatial(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, std::mt19937 &generator);
		void learnLayerSpatialLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, std::mt19937 &generator);
		void learnLayerTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float cellConnectionAlpha, float cellWeightEligibilityDecay, std::mt19937 &generator);
		void learnLayerTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float cellConnectionAlpha, float cellWeightEligibilityDecay, std::mt19937 &generator);
		void learnLayerSpatialTemporal(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, cl::Image2D &nextLayerPrediction, int nextLayerWidth, int nextLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellWeightEligibilityDecay, std::mt19937 &generator);
		void learnLayerSpatialTemporalLast(sys::ComputeSystem &cs, Layer &layer, cl::Image2D &prevLayerOutput, int prevLayerWidth, int prevLayerHeight, const LayerDesc &layerDesc, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellWeightEligibilityDecay, std::mt19937 &generator);
		void updateLayerQWeights(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, float tdError, float cellQWeightEligibilityDecay);
		void dutyCycleLayerUpdate(sys::ComputeSystem &cs, Layer &layer, const LayerDesc &layerDesc, float activationDutyCycleDecay, float stateDutyCycleDecay);

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, float minInitWidth, float maxInitWidth, float minEncoderInitCenter, float maxEncoderInitCenter, float minEncoderInitWidth, float maxEncoderInitWidth, float minEncoderInitWeight, float maxEncoderInitWeight, std::mt19937 &generator);
	
		void step(sys::ComputeSystem &cs, float reward, float columnConnectionAlpha, float widthAlpha, float cellConnectionAlpha, float cellWeightEligibilityDecay, float cellQWeightEligibilityDecay, float activationDutyCycleDecay, float stateDutyCycleDecay, float reconstructionAlpha, float qBiasAlpha, float encoderLocalActivity, float encoderOutputIntensity, float encoderDutyCycleDecay, float encoderBoostThreshold, float encoderBoostIntensity, float encoderCenterAlpha, float encoderWidthAlpha, float encoderWidthScalar, float encoderMinWidth, float encoderReconAlpha, float alpha, float gamma, float tauInv, float breakChance, float perturbationStdDev, std::mt19937 &generator);

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