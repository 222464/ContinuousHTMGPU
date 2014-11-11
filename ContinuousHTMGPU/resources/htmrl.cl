constant sampler_t normalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;
	
constant sampler_t unnormalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;
	
constant float activationIntensity = 4.0f;
constant float columnIntensity = 32.0f;
constant float cellStateIntensity = 4.0f;
constant float cellPredictionIntensity = 2.0f;
constant float minActivation = 0.00001f;
constant float minLearningThreshold = 0.1f;
constant float minDistance = 0.0001f;
constant float widthScalar = 0.0025f;
constant float predictionRangeExtension = 0.1f;
constant float cellQStrength = 0.025f;
constant float columnQStrength = 0.05f;

float randFloat(uint2* state) {
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

void kernel initialize(write_only image2d_t columnActivations, write_only image2d_t columnStates, write_only image3d_t columnWeights,
	write_only image3d_t cellStates, write_only image3d_t cellWeights, write_only image3d_t cellPredictions, write_only image3d_t cellQWeights, write_only image2d_t columnOutputs,
	int cellsInColumn, int layerWidth, int receptiveFieldSize, int lateralConnectionsSize, uint2 seed, float minWeight, float maxWeight, float minWidth, float maxWidth)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 100;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));

	write_imagef(columnActivations, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	write_imagef(columnStates, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	write_imagef(columnOutputs, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));

	for (int wi = 0; wi < receptiveFieldSize; wi++) {
		int4 weightPosition = (int4)(columnPosition.x, columnPosition.y, wi, 0);
	
		float columnWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;
		float columnWidth = randFloat(&seedValue) * (maxWidth - minWidth) + minWidth;

		write_imagef(columnWeights, weightPosition, (float4)(columnWeight, columnWidth, 0.0f, 0.0f));
	}
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		write_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
		write_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
		
		float cellQWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;
	
		write_imagef(cellQWeights, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(cellQWeight, 0.0f, 0.0f, 0.0f));
		
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
	
		for (int wi = 0; wi < lateralConnectionsSize; wi++) {
			int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
			float cellWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;
	
			write_imagef(cellWeights, weightPosition, (float4)(cellWeight, cellWeight, cellWeight, cellWeight));
		}
	}
}

void kernel layerColumnActivate(read_only image2d_t columnStatesPrev, write_only image2d_t columnActivations, read_only image3d_t columnWeights, float2 layerSizeInv, int2 receptiveFieldRadius, float2 inputReceptiveFieldStep) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeInv.x, columnPosition.y * layerSizeInv.y);

	float sum = 0.0f;
	
	int weightIndex = 0;
	
	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		float2 inputPositionNormalized = inputCenterPositionNormalized + (float2)(dx * inputReceptiveFieldStep.x, dy * inputReceptiveFieldStep.y);
		
		float2 weight = read_imagef(columnWeights, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).xy;
		float prevState = read_imagef(columnStatesPrev, normalizedClampedNearestSampler, inputPositionNormalized).x;
		
		float difference = (weight.x - prevState) * weight.y;
		
		sum += difference * difference;
		
		weightIndex++;
	}
	
	float activation = exp(-sum * activationIntensity);
	
	write_imagef(columnActivations, columnPosition, (float4)(activation, activation, activation, activation));
}

void kernel layerColumnInhibit(read_only image2d_t columnActivations, write_only image2d_t columnStates, float2 layerSizeInv, int2 receptiveFieldRadius, float2 layerReceptiveFieldStep, uint2 seed) {
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 50;
	
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 layerCenterPositionNormalized = (float2)(columnPosition.x * layerSizeInv.x, columnPosition.y * layerSizeInv.y);

	float maximum = 0.0f;
	float average = 0.0f;

	int weightIndex = 0;
	
	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		float2 layerPositionNormalized = layerCenterPositionNormalized + (float2)(dx * layerReceptiveFieldStep.x, dy * layerReceptiveFieldStep.y);
		
		float activation = read_imagef(columnActivations, normalizedClampedNearestSampler, layerPositionNormalized).x;
		
		maximum = fmax(maximum, activation);
		average += activation;
		
		weightIndex++;
	}
	
	average /= weightIndex;
	
	float thisActivation = read_imagef(columnActivations, normalizedClampedNearestSampler, layerCenterPositionNormalized).x;
	
	float inhibitedResult;
	
	float difference = maximum - average;

	if (difference == 0.0f)
		inhibitedResult = randFloat(&seedValue) < 1.0f / ((2 * receptiveFieldRadius.x + 1) * (2 * receptiveFieldRadius.y + 1)) ? 1.0f : 0.0f;
	else
		inhibitedResult = exp((thisActivation - maximum) / fmax(minActivation, maximum - average) * columnIntensity);

	write_imagef(columnStates, columnPosition, (float4)(inhibitedResult, inhibitedResult, inhibitedResult, inhibitedResult));
}

void kernel layerColumnWeightUpdate(read_only image2d_t columnStatesPrev, read_only image2d_t columnStates, read_only image3d_t columnWeightsPrev, write_only image3d_t columnWeights, float2 layerSizeInv, int2 receptiveFieldRadius, float2 inputReceptiveFieldStep, float connectionAlpha, float widthAlpha) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeInv.x, columnPosition.y * layerSizeInv.y);

	float state = read_imagef(columnStates, columnPosition).x;
		
	float learnScalar = fmax(0.0f, state - minLearningThreshold);
		
	// Adjust weights by their source activations and error
	int weightIndex = 0;
	
	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		float2 inputPositionNormalized = inputCenterPositionNormalized + (float2)(dx * inputReceptiveFieldStep.x, dy * inputReceptiveFieldStep.y);
		
		float prevState = read_imagef(columnStatesPrev, normalizedClampedNearestSampler, inputPositionNormalized).x;
		
		float2 prevWeight = read_imagef(columnWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).xy;
		
		float difference = prevState - prevWeight.x;
		
		float2 change = (float2)(connectionAlpha * learnScalar * difference, widthAlpha * learnScalar * (widthScalar / fmax(minDistance, fabs(difference)) - prevWeight.y));
		
		float2 newWeight = prevWeight + change;
		
		write_imagef(columnWeights, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0), (float4)(newWeight.x, newWeight.y, 0.0f, 0.0f));
		
		weightIndex++;
	}
}

void kernel layerCellActivate(read_only image2d_t columnStates, read_only image3d_t cellStatesPrev, read_only image3d_t cellPredictionsPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev,
	write_only image3d_t cellStates, int cellsInColumn, int layerWidth, int2 lateralConnectionsRadii)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	
	float minPredictionError = 1.0f;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float prediction = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float predictionError = fabs(columnState - prediction);
		
		minPredictionError = fmin(minPredictionError, predictionError);
	}
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float prediction = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float predictionError = fabs(columnState - prediction);
		
		float newCellState = exp((minPredictionError - predictionError) * cellStateIntensity) * columnState;
	
		write_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(newCellState, newCellState, newCellState, newCellState));
	}
}

void kernel layerCellWeightUpdate(read_only image2d_t columnStates, read_only image3d_t cellStatesPrev, read_only image3d_t cellPredictionsPrev, read_only image2d_t nextLayerContextPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev,
	write_only image3d_t cellWeights, int cellsInColumn, int layerWidth, int2 lateralConnectionsRadii, float2 layerSizeInv, float alpha)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	float columnPredictionPrev = read_imagef(columnPredictionsPrev, columnPosition).x;
	
	float predictionError = (columnState - columnPredictionPrev);
		
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float prediction = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		// Go through all connections and update them
		int wi = 0;
		
		for (int dx = -lateralConnectionsRadii.x; dx <= lateralConnectionsRadii.x; dx++)
		for (int dy = -lateralConnectionsRadii.y; dy <= lateralConnectionsRadii.y; dy++) {
			int2 connectionCoords = (int2)(columnPosition.x + dx, columnPosition.y + dy);
		
			for (int cio = 0; cio < cellsInColumn; cio++) {
				int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
			
				float cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).x;
		
				float connectionState = read_imagef(cellStatesPrev, unnormalizedClampedNearestSampler, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
				
				float newCellWeight = cellWeightPrev + alpha * predictionError * connectionState;
				
				write_imagef(cellWeights, weightPosition, (float4)(newCellWeight, newCellWeight, newCellWeight, newCellWeight));
				
				wi++;
			}
			
			int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
			
			// Additional context from next layer
			float2 normalizedConnectionCoords = (float2)(connectionCoords.x * layerSizeInv.x, connectionCoords.y * layerSizeInv.y);
	
			float nextContext = read_imagef(nextLayerContextPrev, normalizedClampedNearestSampler, normalizedConnectionCoords).x;
			
			float cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).x;

			float newCellWeight = cellWeightPrev + alpha * predictionError * nextContext;
			
			write_imagef(cellWeights, weightPosition, (float4)(newCellWeight, newCellWeight, newCellWeight, newCellWeight));
				
			wi++;
		}
		
		int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float cellBiasPrev = read_imagef(cellWeightsPrev, biasPosition).x;

		float newCellBias = cellBiasPrev + alpha * predictionError;
		
		write_imagef(cellWeights, biasPosition, (float4)(newCellBias, newCellBias, newCellBias, newCellBias));
	}
}

void kernel layerCellWeightUpdateLast(read_only image2d_t columnStates, read_only image3d_t cellStatesPrev, read_only image3d_t cellPredictionsPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev,
	write_only image3d_t cellWeights, int cellsInColumn, int layerWidth, int2 lateralConnectionsRadii, float alpha)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	float columnPredictionPrev = read_imagef(columnPredictionsPrev, columnPosition).x;
	
	float predictionError = (columnState - columnPredictionPrev);
		
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float prediction = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		// Go through all connections and update them
		int wi = 0;
		
		for (int dx = -lateralConnectionsRadii.x; dx <= lateralConnectionsRadii.x; dx++)
		for (int dy = -lateralConnectionsRadii.y; dy <= lateralConnectionsRadii.y; dy++)
		for (int cio = 0; cio < cellsInColumn; cio++) {
			int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
			float cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).x;
	
			float connectionState = read_imagef(cellStatesPrev, unnormalizedClampedNearestSampler, (int4)(columnPosition.x + dx, columnPosition.y + dy, cio, 0)).x;
			
			float newCellWeight = cellWeightPrev + alpha * predictionError * connectionState;
			
			write_imagef(cellWeights, weightPosition, (float4)(newCellWeight, newCellWeight, newCellWeight, newCellWeight));
			
			wi++;
		}
		
		int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float cellBiasPrev = read_imagef(cellWeightsPrev, biasPosition).x;

		float newCellBias = cellBiasPrev + alpha * predictionError;
		
		write_imagef(cellWeights, biasPosition, (float4)(newCellBias, newCellBias, newCellBias, newCellBias));
	}
}

void kernel layerCellPredict(read_only image2d_t columnStates, read_only image3d_t cellStates, read_only image3d_t cellWeights, read_only image2d_t nextLayerContextPrev,
	write_only image3d_t cellPredictions, int cellsInColumn, int layerWidth, int2 lateralConnectionsRadii, float2 layerSizeInv)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float sum = 0.0f;
		
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		// Go through all connections 
		int wi = 0;
		
		for (int dx = -lateralConnectionsRadii.x; dx <= lateralConnectionsRadii.x; dx++)
		for (int dy = -lateralConnectionsRadii.y; dy <= lateralConnectionsRadii.y; dy++) {
			int2 connectionCoords = (int2)(columnPosition.x + dx, columnPosition.y + dy);
		
			for (int cio = 0; cio < cellsInColumn; cio++) {
				int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
			
				float cellWeight = read_imagef(cellWeights, weightPosition).x;
		
				float connectionState = read_imagef(cellStates, unnormalizedClampedNearestSampler, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
				
				sum += cellWeight * connectionState;
				
				wi++;
			}
			
			int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
			
			float cellWeight = read_imagef(cellWeights, weightPosition).x;
			
			float2 normalizedConnectionCoords = (float2)(connectionCoords.x * layerSizeInv.x, connectionCoords.y * layerSizeInv.y);
	
			float nextContext = read_imagef(nextLayerContextPrev, normalizedClampedNearestSampler, normalizedConnectionCoords).x;
			
			sum += cellWeight * nextContext;
			
			wi++;
		}
		
		int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float bias = read_imagef(cellWeights, biasPosition).x;
		
		sum += bias;
		
		float prediction = fmin(1.0f, fmax(0.0f, sigmoid(sum * cellPredictionIntensity) * (1.0f + 2.0f * predictionRangeExtension) - predictionRangeExtension));
		
		write_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0), prediction);
	}
}

void kernel layerCellPredictLast(read_only image2d_t columnStates, read_only image3d_t cellStates, read_only image3d_t cellWeights,
	write_only image3d_t cellPredictions, int cellsInColumn, int layerWidth, int2 lateralConnectionsRadii)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float sum = 0.0f;
		
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		// Go through all connections 
		int wi = 0;
		
		for (int dx = -lateralConnectionsRadii.x; dx <= lateralConnectionsRadii.x; dx++)
		for (int dy = -lateralConnectionsRadii.y; dy <= lateralConnectionsRadii.y; dy++)
		for (int cio = 0; cio < cellsInColumn; cio++) {
			int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
			float cellWeight = read_imagef(cellWeights, weightPosition).x;
	
			float connectionState = read_imagef(cellStates, unnormalizedClampedNearestSampler, (int4)(columnPosition.x + dx, columnPosition.y + dy, cio, 0)).x;
			
			sum += cellWeight * connectionState;
			
			wi++;
		}
		
		int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float bias = read_imagef(cellWeights, biasPosition).x;
		
		sum += bias;
		
		float prediction = fmin(1.0f, fmax(0.0f, sigmoid(sum * cellPredictionIntensity) * (1.0f + 2.0f * predictionRangeExtension) - predictionRangeExtension));
		
		write_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0), prediction);
	}
}

void kernel layerColumnPrediction(read_only image3d_t cellPredictions, read_only image3d_t cellStates, write_only image2d_t columnPredictions, int cellsInColumn) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float maxPrediction = 0.0f;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float prediction = read_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
	
		maxPrediction = fmax(maxPrediction, prediction);
	}
	
	float output = maxPrediction;
	
	write_imagef(columnPredictions, columnPosition, (float4)(output, output, output, output));
}

void kernel layerColumnOutput(read_only image2d_t columnStates, read_only image2d_t columnPredictions, write_only image2d_t columnOutputs) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	float columnPrediction = read_imagef(columnPredictions, columnPosition).x;
	
	float output = fmax(columnState, columnPrediction);
	
	write_imagef(columnOutputs, columnPosition, (float4)(output, output, output, output));
}

void kernel layerRetrievePartialQSums(read_only image3d_t cellStates, read_only image2d_t columnStates, read_only image3d_t cellQWeightsPrev, write_only image2d_t qSummationBuffer, int cellsInColumn) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float sum = 0.0f;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
	
		float cellQWeight = read_imagef(cellQWeightsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		sum += cellQWeight * cellState * cellQStrength;
	}
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	
	float columnQWeight = read_imagef(cellQWeightsPrev, (int4)(columnPosition.x, columnPosition.y, cellsInColumn, 0)).x;
		
	sum += columnState * columnQWeight * columnQStrength;
	
	write_imagef(qSummationBuffer, columnPosition, (float4)(sum, sum, sum, sum));
}

void kernel layerDownsample(read_only image2d_t qSummationBuffer, write_only image2d_t downsampledQSummationBuffer, int2 downsampleSize) {
	int2 position = (int2)(get_global_id(0), get_global_id(1));
	
	float sum = 0.0f;
	
	for (int dx = 0; dx < downsampleSize.x; dx++)
	for (int dy = 0; dy < downsampleSize.y; dy++) {
		float partialSum = read_imagef(qSummationBuffer, (int2)(position.x * downsampleSize.x + dx, position.y * downsampleSize.y + dy)).x;
	
		sum += partialSum;
	}
	
	write_imagef(downsampledQSummationBuffer, position, (float4)(sum, sum, sum, sum));
}

void kernel layerUpdateQWeights(read_only image3d_t cellStates, read_only image2d_t columnStates, read_only image3d_t cellQWeightsPrev, write_only image3d_t cellQWeights, float tdError, float eligibilityDecay, int cellsInColumn) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float2 cellQWeightPrev = read_imagef(cellQWeightsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).xy;
	
		float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
	
		float2 newCellQWeight = cellQWeightPrev + (float2)(tdError * cellQWeightPrev.y, -eligibilityDecay * cellQWeightPrev.y + cellState * cellQStrength);
	
		write_imagef(cellQWeights, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(newCellQWeight.x, newCellQWeight.y, 0.0f, 0.0f));
	}
	
	float2 columnQWeightPrev = read_imagef(cellQWeightsPrev, (int4)(columnPosition.x, columnPosition.y, cellsInColumn, 0)).xy;
	
	float columnState = read_imagef(columnStates, columnPosition).x;

	float2 newCellQWeight = columnQWeightPrev + (float2)(tdError * columnQWeightPrev.y, -eligibilityDecay * columnQWeightPrev.y + columnState * columnQStrength);

	write_imagef(cellQWeights, (int4)(columnPosition.x, columnPosition.y, cellsInColumn, 0), (float4)(newCellQWeight.x, newCellQWeight.y, 0.0f, 0.0f));
}

void kernel reconstructionInit(write_only image3d_t reconstructionWeights, int reconstructionNumWeights, uint2 seed, float minWeight, float maxWeight) {
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 50;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));

	for (int wi = 0; wi < reconstructionNumWeights; wi++) {
		float weight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;
		
		write_imagef(reconstructionWeights, (int4)(columnPosition.x, columnPosition.y, wi, 0), (float4)(weight, weight, weight, weight));
	}
}

void kernel reconstructInput(read_only image3d_t reconstructionWeights, read_only image2d_t sdr, write_only image2d_t inputs, int2 receptiveFieldRadius, float2 inputSizeInv, float2 sdrSizeInv) {
	int2 inputPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputPositionNormalized = (float2)(inputPosition.x * inputSizeInv.x, inputPosition.y * inputSizeInv.y);
	
	float sum = 0.0f;
	
	int wi = 0;
	
	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		float2 sdrPositionNormalized = inputPositionNormalized + (float2)(dx * sdrSizeInv.x, dy + sdrSizeInv.y);
		
		float source = read_imagef(sdr, normalizedClampedNearestSampler, sdrPositionNormalized).x;

		float weight = read_imagef(reconstructionWeights, (int4)(inputPosition.x, inputPosition.y, wi, 0)).x;
		
		sum += source * weight;
		
		wi++;
	}
	
	// Bias
	float bias = read_imagef(reconstructionWeights, (int4)(inputPosition.x, inputPosition.y, wi, 0)).x;
		
	sum += bias;
	
	write_imagef(inputs, inputPosition, (float4)(sum, sum, sum, sum));
}

void kernel updateReconstruction(read_only image2d_t targets, read_only image2d_t inputs, read_only image3d_t reconstructionWeightsPrev, read_only image2d_t sdr, write_only image3d_t reconstructionWeights, int2 receptiveFieldRadius, float2 inputSizeInv, float2 sdrSizeInv, float alpha) {
	int2 inputPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputPositionNormalized = (float2)(inputPosition.x * inputSizeInv.x, inputPosition.y * inputSizeInv.y);
	
	float target = read_imagef(targets, inputPosition).x;
	float input = read_imagef(inputs, inputPosition).x;
	
	float error = alpha * (target - input);
	
	int wi = 0;
	
	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		float2 sdrPositionNormalized = inputPositionNormalized + (float2)(dx * sdrSizeInv.x, dy + sdrSizeInv.y);
		
		float source = read_imagef(sdr, normalizedClampedNearestSampler, sdrPositionNormalized).x;

		float prevWeight = read_imagef(reconstructionWeightsPrev, (int4)(inputPosition.x, inputPosition.y, wi, 0)).x;
		
		float newWeight = prevWeight + error * source;
		
		write_imagef(reconstructionWeights, (int4)(inputPosition.x, inputPosition.y, wi, 0), (float4)(newWeight, newWeight, newWeight, newWeight));
		
		wi++;
	}
	
	// Bias
	float prevBias = read_imagef(reconstructionWeightsPrev, (int4)(inputPosition.x, inputPosition.y, wi, 0)).x;
		
	float newBias = prevBias + error;
	
	write_imagef(reconstructionWeights, (int4)(inputPosition.x, inputPosition.y, wi, 0), (float4)(newBias, newBias, newBias, newBias));
}