#pragma once

#include <vector>

#include <random>

#include <memory>

#include <algorithm>

namespace htm {
	class AnythingEncoder {
	public:
		float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		struct Node {
			std::vector<float> _center;

			float _activation;
			float _output;
			float _outputPrev;
			float _dutyCycle;

			Node()
				: _activation(0.0f), _output(0.0f), _outputPrev(0.0f), _dutyCycle(0.0f)
			{}
		};

		int _sdrSize;
		int _inputSize;

		int _boostCandidate;
		float _bestRepresentation;

		std::vector<Node> _nodes;
	
	public:
		AnythingEncoder()
			: _boostCandidate(0), _bestRepresentation(1.0f)
		{}

		void create(int sdrSize, int inputSize, float minInitCenter, float maxInitCenter, std::mt19937 &generator);

		void encode(const std::vector<float> &input, std::vector<float> &sdr, float localActivity, float outputIntensity, float dutyCycleDecay);
		void learn(const std::vector<float> &input, float centerAlpha, float maxDutyCycleForLearn, float noMatchIntensity);
		void decode(const std::vector<float> &sdr, std::vector<float> &recon);
	};
}