constant sampler_t normalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;
	
constant float sparsity = 0.9f;
constant float intensity = 4.0f;

float randFloat(uint2* state)
{
    const float invMaxInt = 1.0f / 4294967296.0f;
    uint x = (*state).x * 17 + (*state).y * 13123;
    (*state).x = (x << 13) ^ x;
    (*state).y ^= (x << 7);

    uint tmp = x * (x * x * 15731 + 74323) + 871483;

    return convert_float(tmp) * invMaxInt;
}

float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

float logit(float x) {
	return -log(1.0f / x - 1.0f);
}

void kernel weightInit(write_only image2d_t states, write_only image2d_array_t weights, int receptiveFieldSize, uint2 seed, float minWeight, float maxWeight) {
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1));

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));

	write_imagef(states, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));

	for (int wi = 0; wi < receptiveFieldSize; wi++) {
		int4 weightPosition = (int4)(columnPosition.x, columnPosition.y, wi, 0);
	
		float weight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;
	
		write_imagef(weights, weightPosition, (float4)(weight, weight, weight, weight));
	}
}

void kernel layerActivate(read_only image2d_t prevStates, write_only image2d_t activations, read_only image2d_array_t weights, float2 layerSizeInv, float2 inputReceptiveFieldRadius, float2 inputReceptiveFieldStep) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeInv.x, columnPosition.y * layerSizeInv.y);

	float sum = 0.0f;
	
	int weightIndex = 0;
	
	for (float dx = -inputReceptiveFieldRadius.x; dx <= inputReceptiveFieldRadius.x; dx += inputReceptiveFieldStep.x)
	for (float dy = -inputReceptiveFieldRadius.y; dy <= inputReceptiveFieldRadius.y; dy += inputReceptiveFieldStep.y) {
		float2 inputPositionNormalized = inputCenterPositionNormalized + (float2)(dx, dy);
		
		float weight = read_imagef(weights, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
		float prevState = read_imagef(prevStates, normalizedClampedNearestSampler, inputPositionNormalized).x;
		
		sum += weight * prevState;
		
		weightIndex++;
	}
	
	float activation = sigmoid(sum) * 2.0f - 1.0f;
	
	write_imagef(activations, columnPosition, (float4)(activation, activation, activation, activation));
}

void kernel layerInhibit(read_only image2d_t activations, write_only image2d_t states, float2 layerSizeInv, float2 layerReceptiveFieldRadius, float2 layerReceptiveFieldStep) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 layerCenterPositionNormalized = (float2)(columnPosition.x * layerSizeInv.x, columnPosition.y * layerSizeInv.y);

	float average = 0.0f;
	float maximum = 0.0f;
	float minimum = 1.0f;
	
	int weightIndex = 0;
	
	for (float dx = -layerReceptiveFieldRadius.x; dx <= layerReceptiveFieldRadius.x; dx += layerReceptiveFieldStep.x)
	for (float dy = -layerReceptiveFieldRadius.y; dy <= layerReceptiveFieldRadius.y; dy += layerReceptiveFieldStep.y) {
		float2 layerPositionNormalized = layerCenterPositionNormalized + (float2)(dx, dy);
		
		float activation = read_imagef(activations, normalizedClampedNearestSampler, layerPositionNormalized).x;
		
		average += activation;
		maximum = max(maximum, activation);
		minimum = min(minimum, activation);
		
		weightIndex++;
	}
	
	average /= weightIndex;
	
	float thisActivation = read_imagef(activations, normalizedClampedNearestSampler, layerCenterPositionNormalized).x;
	
	// If this activation is above average
	float error = thisActivation - (sparsity * maximum + (1.0f - sparsity) * average);
	float inhibitedResult = sigmoid(error * intensity) * 2.0f - 1.0f;

	write_imagef(states, columnPosition, (float4)(inhibitedResult, inhibitedResult, inhibitedResult, inhibitedResult));
}

void kernel layerWeightUpdate(read_only image2d_t prevStates, read_only image2d_t states, read_only image2d_array_t prevWeights, write_only image2d_array_t weights, float2 layerSizeInv, float2 inputReceptiveFieldRadius, float2 inputReceptiveFieldStep, float alpha) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeInv.x, columnPosition.y * layerSizeInv.y);

	float state = read_imagef(states, columnPosition).x;
		
	// Adjust weights by their source activations and error
	int weightIndex = 0;
	
	for (float dx = -inputReceptiveFieldRadius.x; dx <= inputReceptiveFieldRadius.x; dx += inputReceptiveFieldStep.x)
	for (float dy = -inputReceptiveFieldRadius.y; dy <= inputReceptiveFieldRadius.y; dy += inputReceptiveFieldStep.y) {
		float2 inputPositionNormalized = inputCenterPositionNormalized + (float2)(dx, dy);
		
		float prevState = read_imagef(prevStates, normalizedClampedNearestSampler, inputPositionNormalized).x;
		
		float prevWeight = read_imagef(prevWeights, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
		
		float change = alpha * (state * prevState);
		
		float newWeight = prevWeight + change;
		
		write_imagef(weights, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0), (float4)(newWeight, newWeight, newWeight, newWeight));
		
		weightIndex++;
	}
}