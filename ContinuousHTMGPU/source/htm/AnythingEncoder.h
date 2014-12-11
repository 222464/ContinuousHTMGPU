#pragma once

#include <vector>

#include <random>

#include <memory>

namespace htm {
	class AnythingEncoder {
	public:
		float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		struct Node {
			std::vector<float> _center;

			float _width;

			float _activation;
			float _output;

			Node()
				: _activation(0.0f), _output(0.0f)
			{}
		};

		struct Recon {
			std::vector<float> _reconWeights;

			float _reconBias;
		};

		int _sdrSize;
		int _inputSize;

		std::vector<Node> _nodes;
		std::vector<Recon> _recons;

	public:
		void create(int sdrSize, int inputSize, float minInitCenter, float maxInitCenter, float minInitWidth, float maxInitWidth, float minInitWeight, float maxInitWeight, std::mt19937 &generator);

		void encode(const std::vector<float> &input, std::vector<float> &sdr, float localActivity, float outputIntensity);
		void learn(const std::vector<float> &input, const std::vector<float> &recon, float centerAlpha, float widthAlpha, float widthScalar, float minWidth, float reconAlpha);
		void decode(const std::vector<float> &sdr, std::vector<float> &recon);
	};
}