#include <htm/AnythingEncoder.h>

#include <algorithm>

using namespace htm;

void AnythingEncoder::create(int sdrSize, int inputSize, float minInitCenter, float maxInitCenter, float minInitWidth, float maxInitWidth, float minInitWeight, float maxInitWeight, std::mt19937 &generator) {
	_sdrSize = sdrSize;
	_inputSize = inputSize;

	_nodes.resize(sdrSize);
	_recons.resize(inputSize);

	std::uniform_real_distribution<float> centerDist(minInitCenter, maxInitCenter);
	std::uniform_real_distribution<float> widthDist(minInitWidth, maxInitWidth);
	std::uniform_real_distribution<float> weightDist(minInitWeight, maxInitWeight);

	for (int i = 0; i < _sdrSize; i++) {
		_nodes[i]._center.resize(inputSize);

		for (int j = 0; j < _inputSize; j++)
			_nodes[i]._center[j] = centerDist(generator);
		
		_nodes[i]._width = widthDist(generator);
	}

	for (int i = 0; i < _inputSize; i++) {
		_recons[i]._reconWeights.resize(sdrSize);

		_recons[i]._reconBias = weightDist(generator);

		for (int j = 0; j < _sdrSize; j++)
			_recons[i]._reconWeights[j] = weightDist(generator);
	}
}

void AnythingEncoder::encode(const std::vector<float> &input, std::vector<float> &sdr, float localActivity, float outputIntensity) {
	if (sdr.size() != _sdrSize)
		sdr.resize(_sdrSize);

	for (int i = 0; i < _sdrSize; i++) {
		float sum = 0.0f;

		for (int j = 0; j < _inputSize; j++) {
			float difference = _nodes[i]._center[j] - input[j];

			sum += difference * difference;
		}

		_nodes[i]._activation = -sum * _nodes[i]._width;
	}

	// Inhibit
	for (int i = 0; i < _sdrSize; i++) {
		float numHigher = 0.0f;

		for (int j = 0; j < _sdrSize; j++) {
			if (_nodes[j]._activation >= _nodes[i]._activation)
				numHigher++;
		}

		sdr[i] = _nodes[i]._output = sigmoid((localActivity - numHigher) * outputIntensity);
	}
}

void AnythingEncoder::learn(const std::vector<float> &input, const std::vector<float> &recon, float centerAlpha, float widthAlpha, float widthScalar, float minWidth, float reconAlpha, float outputBaseline) {
	for (int i = 0; i < _sdrSize; i++) {
		float learnScalar = (_nodes[i]._output + outputBaseline) * (1.0f - outputBaseline);

		for (int j = 0; j < _inputSize; j++) {
			float difference = input[j] - _nodes[i]._center[j];

			_nodes[i]._center[j] += centerAlpha * learnScalar * difference;
		}

		_nodes[i]._width = std::max(0.0f, _nodes[i]._width + widthAlpha * learnScalar * (widthScalar / std::max(minWidth, -_nodes[i]._activation) - _nodes[i]._width));
	}

	for (int i = 0; i < _inputSize; i++) {
		float reconError = reconAlpha * (input[i] - recon[i]);

		_recons[i]._reconBias += reconError;

		for (int j = 0; j < _sdrSize; j++)
			_recons[i]._reconWeights[j] += reconError * _nodes[j]._output;
	}
}

void AnythingEncoder::decode(const std::vector<float> &sdr, std::vector<float> &recon) {
	if (recon.size() != _inputSize)
		recon.resize(_inputSize);

	for (int i = 0; i < _inputSize; i++) {
		float sum = _recons[i]._reconBias;

		for (int j = 0; j < _sdrSize; j++)
			sum += _recons[i]._reconWeights[j] * _nodes[j]._output;

		recon[i] = sum;
	}
}