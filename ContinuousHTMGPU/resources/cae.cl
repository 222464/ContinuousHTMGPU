constant sampler_t normalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;
	
constant sampler_t unnormalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;
	
float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

void kernel layerActivateForward(read_only image2d_array_t prevLayerOutputs, read_only image2d_t layerWeights,
	write_only image2d_array_t layerOutputs, int2 kernelSize, int prevNumMaps, float2 layerSizeInv, float2 prevLayerSizeInv)
{
	int3 position = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	
	float2 positionNormalized2D = (float2)(position.x * layerSizeInv.x, position.y * layerSizeInv.y);

	// First weight is bias
	float sum = read_imagef(layerWeights, (int2)(0, position.z)).x;
	
	int weightIndex = 1;
	
	for (int x = 0; x < kernelSize.x; x++)
	for (int y = 0; y < kernelSize.y; y++)
	for (int m = 0; m < prevNumMaps; m++) {
		float weight = read_imagef(layerWeights, (int2)(weightIndex, position.z)).x;
		float prevLayerOutput = read_imagef(prevLayerOutputs, normalizedClampedNearestSampler, (float4)(positionNormalized2D.x + (x - kernelSize.x * 0.5f) * prevLayerSizeInv.x, positionNormalized2D.y + (y - kernelSize.y * 0.5f) * prevLayerSizeInv.x, m, 0)).x;
		
		sum += weight * prevLayerOutput;
		
		weightIndex++;
	}
	
	float output = sigmoid(sum);
	
	write_imagef(layerOutputs, (int4)(position.x, position.y, position.z, 0), (float4)(output, output, output, output));
}

void kernel layerActivateBackward(read_only image2d_array_t layerOutputs, read_only image2d_array_t prevLayerOutputs,
	read_only image2d_t layerWeights, read_only image2d_array_t prevLayerBiases, write_only image2d_array_t newPrevLayerBiases, write_only image2d_array_t prevLayerErrors,
	int2 kernelSize, int2 reverseKernelSize, int numMaps, int prevNumMaps, float2 layerSizeInv, float2 prevLayerSizeInv, float alpha)
{
	int3 prevPosition = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	float2 prevPositionNormalized2D = (float2)(prevPosition.x * prevLayerSizeInv.x, prevPosition.y * prevLayerSizeInv.y);
	
	float prevLayerBias = read_imagef(prevLayerBiases, (int4)(prevPosition.x, prevPosition.y, prevPosition.z, 0)).x;
	
	float sum = prevLayerBias;
	
	int2 start = (int2)(prevPosition.x, prevPosition.y);
	
	for (int x = 0; x < reverseKernelSize.x; x++)
	for (int y = 0; y < reverseKernelSize.y; y++)
	for (int m = 0; m < numMaps; m++) {
		float weight = read_imagef(layerWeights, (int2)(prevPosition.z + y * prevNumMaps + x * prevNumMaps * reverseKernelSize.y + 1, m)).x;
		float layerOutput = read_imagef(layerOutputs, normalizedClampedNearestSampler, (float4)(prevPositionNormalized2D.x + (x - reverseKernelSize.x * 0.5f) * layerSizeInv.x, prevPositionNormalized2D.y + (y - reverseKernelSize.y * 0.5f) * layerSizeInv.y, m, 0)).x;
		
		sum += weight * layerOutput;
	}
	
	float output = sigmoid(sum);
	
	float target = read_imagef(prevLayerOutputs, (int4)(prevPosition.x, prevPosition.y, prevPosition.z, 0)).x;
	
	float error = (target - output);// * output * (1.0f - output);
	
	// Update prev layer bias
	float newPrevLayerBias = prevLayerBias + alpha * error;
	
	write_imagef(newPrevLayerBiases, (int4)(prevPosition.x, prevPosition.y, prevPosition.z, 0), (float4)(newPrevLayerBias, newPrevLayerBias, newPrevLayerBias, newPrevLayerBias));
	
	// Store error
	write_imagef(prevLayerErrors, (int4)(prevPosition.x, prevPosition.y, prevPosition.z, 0), (float4)(error, error, error, error));
}

void kernel layerWeightUpdate(read_only image2d_array_t layerOutputs, read_only image2d_array_t prevLayerOutputs, read_only image2d_array_t prevLayerErrors,
	read_only image2d_t layerWeights, write_only image2d_array_t weightDeltaSummationMap,
	int2 kernelSize, int numMaps, int prevNumMaps, float2 layerSizeInv, float2 prevLayerSizeInv, int numWeightsPerMap, float alpha)
{
	int3 position = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	
	float2 positionNormalized2D = (float2)(position.x * layerSizeInv.x, position.y * layerSizeInv.y);
	
	float layerOutput = read_imagef(layerOutputs, (int4)(position.x, position.y, position.z, 0)).x;

	int weightIndex = 1; // 1 since we are skipping the bias for the error backpropagation
	
	// Backpropagate error
	float error = 0.0f;
	
	for (int x = 0; x < kernelSize.x; x++)
	for (int y = 0; y < kernelSize.y; y++)
	for (int m = 0; m < prevNumMaps; m++) {
		float weight = read_imagef(layerWeights, (int2)(weightIndex, position.z)).x;
		float prevLayerError = read_imagef(prevLayerErrors, normalizedClampedNearestSampler, (float4)(positionNormalized2D.x + (x - kernelSize.x * 0.5f) * prevLayerSizeInv.x, positionNormalized2D.y + (y - kernelSize.y * 0.5f) * prevLayerSizeInv.y, m, 0)).x;
		
		error += weight * prevLayerError;
		
		weightIndex++;
	}

	error *= layerOutput * (1.0f - layerOutput);
	
	// Update bias
	float bias = read_imagef(layerWeights, (int2)(0, position.z)).x;
	
	float biasDelta = alpha * error;
	
	int thisWeightsStart = position.z * numWeightsPerMap;
	
	write_imagef(weightDeltaSummationMap, (int4)(position.x, position.y, thisWeightsStart, 0), biasDelta);
	
	weightIndex = 1;
	
	// Update all non-bias weights
	for (int x = 0; x < kernelSize.x; x++)
	for (int y = 0; y < kernelSize.y; y++)
	for (int m = 0; m < prevNumMaps; m++) {
		float weight = read_imagef(layerWeights, (int2)(weightIndex, position.z)).x;
		float prevLayerOutput = read_imagef(prevLayerOutputs, normalizedClampedNearestSampler, (float4)(positionNormalized2D.x + (x - kernelSize.x * 0.5f) * prevLayerSizeInv.x, positionNormalized2D.y + (y - kernelSize.y * 0.5f) * prevLayerSizeInv.y, m, 0)).x;
		float prevLayerError = read_imagef(prevLayerErrors, normalizedClampedNearestSampler, (float4)(positionNormalized2D.x + (x - kernelSize.x * 0.5f) * prevLayerSizeInv.x, positionNormalized2D.y + (y - kernelSize.y * 0.5f) * prevLayerSizeInv.y, m, 0)).x;
		
		float weightDelta = alpha * (error * prevLayerOutput + prevLayerError * layerOutput);
		
		write_imagef(weightDeltaSummationMap, (int4)(position.x, position.y, thisWeightsStart + weightIndex, 0), weightDelta);
		
		weightIndex++;
	}
}

void kernel weightDeltaReduce(read_only image2d_array_t expandedWeightDeltas, write_only image2d_array_t reducedWeightDeltas, int totalNumWeightsPerLayer, int2 reduceStep) {
	int2 positionReduced = (int2)(get_global_id(0), get_global_id(1));
	int2 positionExpanded = (int2)(positionReduced.x * reduceStep.x, positionReduced.y * reduceStep.y);
	
	for (int i = 0; i < totalNumWeightsPerLayer; i++) {
		float sum = 0.0f;
		
		for (int dx = 0; dx < reduceStep.x; dx++)
		for (int dy = 0; dy < reduceStep.y; dy++)
			sum += read_imagef(expandedWeightDeltas, unnormalizedClampedNearestSampler, (int4)(positionExpanded.x + dx, positionExpanded.y + dy, i, 0)).x;
			
		write_imagef(reducedWeightDeltas, (int4)(positionReduced.x, positionReduced.y, i, 0), (float4)(sum, sum, sum, sum));
	}
}

void kernel mapsDeltaUpdate(read_only image2d_array_t reducedWeightDeltas, read_only image2d_t layerWeights, write_only image2d_t newLayerWeights, int numWeightsPerMap) {
	int3 position = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	
	for (int i = 0; i < numWeightsPerMap; i++) {
		float delta = read_imagef(reducedWeightDeltas, (int4)(position.x, position.y, position.z * numWeightsPerMap + i, 0)).x;
	
		float original = read_imagef(layerWeights, (int2)(i, position.z)).x;
	
		float next = original + delta;
	
		write_imagef(newLayerWeights, (int2)(i, position.z), (float4)(next, next, next, next));
	}
}

void kernel layerDownsample(read_only image2d_array_t layerOutputs, write_only image2d_array_t downsampledOutputs, int2 downsampleSize) {
	int3 position = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	
	float sample = 0.0f;
	
	for (int dx = 0; dx < downsampleSize.x; dx++)
	for (int dy = 0; dy < downsampleSize.y; dy++) {
		float layerOutput = read_imagef(layerOutputs, (int4)(position.x * downsampleSize.x + dx, position.y * downsampleSize.y + dy, position.z, 0)).x;
	
		sample = max(sample, layerOutput);
	}
	
	write_imagef(downsampledOutputs, (int4)(position.x, position.y, position.z, 0), (float4)(sample, sample, sample, sample));
}