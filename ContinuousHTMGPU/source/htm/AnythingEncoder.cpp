#include "AnythingEncoder.h"

using namespace htm;

void AnythingEncoder::create(int sdrSize, int inputSize, float minInitCenter, float maxInitCenter, std::mt19937 &generator) {
	_sdrSize = sdrSize;
	_inputSize = inputSize;

	_nodes.resize(sdrSize);
	
	std::uniform_real_distribution<float> centerDist(minInitCenter, maxInitCenter);

	for (int i = 0; i < _sdrSize; i++) {
		_nodes[i]._center.resize(inputSize);

		for (int j = 0; j < _inputSize; j++)
			_nodes[i]._center[j] = centerDist(generator);
	}
}

void AnythingEncoder::encode(const std::vector<float> &input, std::vector<float> &sdr, float localActivity, float outputIntensity, float dutyCycleDecay) {
	if (sdr.size() != _sdrSize)
		sdr.resize(_sdrSize);

	float maxActivation = -999999.0f;

	for (int i = 0; i < _sdrSize; i++) {
		float sum = 0.0f;

		for (int j = 0; j < _inputSize; j++) {
			float difference = _nodes[i]._center[j] - input[j];

			sum += difference * difference;
		}

		_nodes[i]._activation = -sum;

		maxActivation = std::max(maxActivation, _nodes[i]._activation);
	}

	_bestRepresentation = maxActivation;

	// Inhibit
	for (int i = 0; i < _sdrSize; i++) {
		float numHigher = 0.0f;

		for (int j = 0; j < _sdrSize; j++) {
			if (_nodes[j]._activation > _nodes[i]._activation)
				numHigher++;
		}

		_nodes[i]._outputPrev = _nodes[i]._output;

		sdr[i] = _nodes[i]._output = sigmoid((localActivity - numHigher) * outputIntensity);

		_nodes[i]._dutyCycle = std::max((1.0f - dutyCycleDecay) * _nodes[i]._dutyCycle, _nodes[i]._output);

		if (_nodes[i]._dutyCycle < _nodes[_boostCandidate]._dutyCycle)
			_boostCandidate = i;
	}
}

void AnythingEncoder::learn(const std::vector<float> &input, float centerAlpha, float maxDutyCycleForLearn, float noMatchIntensity) {
	float noMatch = 1.0f - exp(_bestRepresentation * noMatchIntensity);

	float boost = _nodes[_boostCandidate]._dutyCycle < maxDutyCycleForLearn ? noMatch : 0.0f;

	float learnScalar = (1.0f - boost) * std::max(0.0f, _nodes[_boostCandidate]._output - _nodes[_boostCandidate]._outputPrev) + boost;

	for (int j = 0; j < _inputSize; j++) {
		float difference = input[j] - _nodes[_boostCandidate]._center[j];

		_nodes[_boostCandidate]._center[j] += centerAlpha * learnScalar * difference;
	}
}

void AnythingEncoder::decode(const std::vector<float> &sdr, std::vector<float> &recon) {
	if (recon.size() != _inputSize)
		recon.resize(_inputSize);

	for (int i = 0; i < _inputSize; i++) {
		float sum = 0.0f;
		float divisor = 0.0f;

		for (int j = 0; j < _sdrSize; j++) {
			sum += _nodes[j]._center[i] * _nodes[j]._output;

			divisor += _nodes[j]._output;
		}

		if (divisor == 0.0f)
			recon[i] = 0.0f;
		else
			recon[i] = sum / divisor;
	}
}