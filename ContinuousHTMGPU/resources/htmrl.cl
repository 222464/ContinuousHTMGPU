constant sampler_t normalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;
	
constant sampler_t normalizedClampedToEdgeNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;
	
constant sampler_t unnormalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;
	
constant sampler_t defaultNormalizedSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;
	
constant sampler_t defaultUnnormalizedSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;
	
constant float activationIntensity = 1.0f;
constant float columnIntensity = 4.0f;
constant float cellStateIntensity = 4.0f;
constant float cellPredictionIntensity = 4.0f;
constant float minActivation = 0.00001f;
constant float minLearningThreshold = 0.0f;
constant float predictionRangeExtension = 0.1f;
constant float localActivity = 4.0f;
constant float minDutyCycleRatio = 0.02f;
constant float minDutyCycle = 0.02f;
constant float boostFactor = 20.0f;

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

float boostFunction(float active, float minimum) {
	return fmax(0.0f, minimum - active) * boostFactor;
}

void kernel initializePartOne(write_only image2d_t columnActivations, write_only image2d_t columnDutyCycles, write_only image2d_t columnStates, write_only image3d_t columnWeights,
	int cellsInColumn, int receptiveFieldSize, int lateralConnectionsSize, uint2 seed, float minWeight, float maxWeight, float minWidth, float maxWidth)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 100;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));

	write_imagef(columnActivations, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	write_imagef(columnStates, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));

	write_imagef(columnDutyCycles, columnPosition, (float4)(minDutyCycle, 0.0f, 0.0f, 0.0f));
	
	for (int wi = 0; wi < receptiveFieldSize; wi++) {
		int4 weightPosition = (int4)(columnPosition.x, columnPosition.y, wi, 0);
	
		float columnConnectionWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;

		write_imagef(columnWeights, weightPosition, (float4)(columnConnectionWeight, 0.0f, 0.0f, 0.0f));
	}
}

void kernel initializePartTwo(write_only image3d_t cellStates, write_only image3d_t cellWeights, write_only image3d_t cellPredictions, write_only image3d_t cellQWeights,
	int cellsInColumn, int receptiveFieldSize, int lateralConnectionsSize, uint2 seed, float minWeight, float maxWeight, float minWidth, float maxWidth)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 130;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
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

void kernel layerColumnActivate(read_only image2d_t columnStatesInput, read_only image3d_t columnWeightsPrev, write_only image2d_t columnActivations, float2 layerSizeInv, int2 inputReceptiveFieldRadius, float2 inputReceptiveFieldStep, int2 inputSize) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeInv.x, columnPosition.y * layerSizeInv.y);

	float sum = 0.0f;
	
	int weightIndex = 0;
	
	for (int dx = -inputReceptiveFieldRadius.x; dx <= inputReceptiveFieldRadius.x; dx++)
	for (int dy = -inputReceptiveFieldRadius.y; dy <= inputReceptiveFieldRadius.y; dy++) {
		float2 inputPositionNormalized = inputCenterPositionNormalized + (float2)(dx * inputReceptiveFieldStep.x, dy * inputReceptiveFieldStep.y);
		int2 inputPosition = (int2)(inputPositionNormalized.x * inputSize.x, inputPositionNormalized.y * inputSize.y);
		
		if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
			float weight = read_imagef(columnWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
			float inputState = read_imagef(columnStatesInput, inputPosition).x;
				
			float difference = weight - inputState;
				
			sum += difference * difference;
				
			weightIndex++;
		}
	}

	float activation = exp(-sum * activationIntensity);
	
	write_imagef(columnActivations, columnPosition, (float4)(activation, activation, activation, activation));
}

void kernel layerColumnInhibit(read_only image2d_t columnActivations, read_only image2d_t columnDutyCyclesPrev, write_only image2d_t columnStates, int2 layerSize, float2 layerSizeInv, int2 receptiveFieldRadius, float2 layerReceptiveFieldStep, uint2 seed) {
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 50;
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float thisActivation = read_imagef(columnActivations, columnPosition).x;
	
	float higherSum = 0.0f;
	
	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		int2 layerPosition = columnPosition + (int2)(dx, dy);
	
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			float activation = read_imagef(columnActivations, layerPosition).x;
			
			if (activation > thisActivation)
				higherSum++;
		}
	}
	
	float boost = read_imagef(columnDutyCyclesPrev, columnPosition).y;
	
	float inhibitedResult = fmin(1.0f, boost + sigmoid((localActivity - higherSum) * columnIntensity) * thisActivation);
	
	write_imagef(columnStates, columnPosition, (float4)(inhibitedResult, inhibitedResult, inhibitedResult, inhibitedResult));
}

void kernel layerColumnDutyCycleUpdate(read_only image2d_t columnDutyCyclesPrev, read_only image2d_t columnStates, write_only image2d_t columnDutyCycles, write_only image2d_t columnAttentions, int2 receptiveFieldRadius, float2 layerReceptiveFieldStep, int2 layerSize, float2 layerSizeInv, float dutyCycleDecay) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float maxNeighborhoodDutyCycle = 0.0f;
	
	float lowerSum = 0.0f;

	float thisColumnDutyCyclePrev = read_imagef(columnDutyCyclesPrev, columnPosition).x;
	
	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		int2 layerPosition = columnPosition + (int2)(dx, dy);
	
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			float columnDutyCyclePrev = read_imagef(columnDutyCyclesPrev, layerPosition).x;
			
			if (columnDutyCyclePrev < thisColumnDutyCyclePrev)
				lowerSum++;
			
			maxNeighborhoodDutyCycle = fmax(maxNeighborhoodDutyCycle, columnDutyCyclePrev);
		}
	}
	
	float lowerDutyCycle = maxNeighborhoodDutyCycle * minDutyCycleRatio;
	
	float thisColumnState = read_imagef(columnStates, columnPosition).x;
	
	float newColumnDutyCycle = (1.0f - dutyCycleDecay) * thisColumnDutyCyclePrev + dutyCycleDecay * thisColumnState;
	
	float boost = boostFunction(newColumnDutyCycle, lowerDutyCycle);
	
	write_imagef(columnDutyCycles, columnPosition, (float4)(newColumnDutyCycle, boost, 0.0f, 0.0f));
	
	float inhibitedResult = sigmoid((localActivity - lowerSum) * columnIntensity);
	
	write_imagef(columnAttentions, columnPosition, (float4)(inhibitedResult, inhibitedResult, inhibitedResult, inhibitedResult));
}

void kernel layerColumnWeightUpdate(read_only image2d_t columnStatesInput, read_only image2d_t columnStates, read_only image3d_t columnWeightsPrev, write_only image3d_t columnWeights, int2 layerSize, float2 layerSizeInv, int2 inputReceptiveFieldRadius, float2 inputReceptiveFieldStep, int2 inputSize, float connectionAlpha) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeInv.x, columnPosition.y * layerSizeInv.y);

	float state = read_imagef(columnStates, columnPosition).x;
		
	float learnScalar = fmax(0.0f, (state - minLearningThreshold) / (1.0f - minLearningThreshold));
		
	// Adjust weights by their source activations and error
	int weightIndex = 0;
	
	for (int dx = -inputReceptiveFieldRadius.x; dx <= inputReceptiveFieldRadius.x; dx++)
	for (int dy = -inputReceptiveFieldRadius.y; dy <= inputReceptiveFieldRadius.y; dy++) {
		float2 inputPositionNormalized = inputCenterPositionNormalized + (float2)(dx * inputReceptiveFieldStep.x, dy * inputReceptiveFieldStep.y);
		int2 inputPosition = (int2)(inputPositionNormalized.x * inputSize.x, inputPositionNormalized.y * inputSize.y);
		
		if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
			float inputState = read_imagef(columnStatesInput, inputPosition).x;
				
			float prevWeight = read_imagef(columnWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
				
			float difference = inputState - prevWeight;
				
			float change = connectionAlpha * learnScalar * difference;
				
			float newWeight = prevWeight + change;
				
			write_imagef(columnWeights, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0), (float4)(newWeight, newWeight, newWeight, newWeight));
				
			weightIndex++;
		}
	}
}

void kernel layerCellActivate(read_only image2d_t columnStates, read_only image3d_t cellStatesPrev, read_only image3d_t cellPredictionsPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev,
	write_only image3d_t cellStates, int cellsInColumn, int2 lateralConnectionsRadii)
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

void kernel layerCellWeightUpdate(read_only image2d_t columnStates, read_only image3d_t cellStatesPrev, read_only image2d_t nextLayerContextPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev,
	write_only image3d_t cellWeights, int cellsInColumn, int2 layerSize, int2 lateralConnectionsRadii, float2 layerSizeInv, int2 nextLayerSize, float alpha)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	float columnPredictionPrev = read_imagef(columnPredictionsPrev, columnPosition).x;
	
	float predictionError = (columnState - columnPredictionPrev);
		
	for (int ci = 0; ci < cellsInColumn; ci++) {
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		// Go through all connections and update them
		int wi = 0;
		
		for (int dx = -lateralConnectionsRadii.x; dx <= lateralConnectionsRadii.x; dx++)
		for (int dy = -lateralConnectionsRadii.y; dy <= lateralConnectionsRadii.y; dy++) {
			int2 connectionCoords = (int2)(columnPosition.x + dx, columnPosition.y + dy);
			
			if (connectionCoords.x >= 0 && connectionCoords.x < layerSize.x && connectionCoords.y >= 0 && connectionCoords.y < layerSize.y) {	
				for (int cio = 0; cio < cellsInColumn; cio++) {
					int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
				
					float cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).x;
			
					float connectionState = read_imagef(cellStatesPrev, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
					
					float newCellWeight = cellWeightPrev + alpha * predictionError * connectionState;
					
					write_imagef(cellWeights, weightPosition, (float4)(newCellWeight, newCellWeight, newCellWeight, newCellWeight));
					
					wi++;
				}
				
				int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
				
				// Additional context from next layer
				float2 normalizedConnectionCoords = (float2)(connectionCoords.x * layerSizeInv.x, connectionCoords.y * layerSizeInv.y);
				int2 connectionCoordsNext = (int2)(normalizedConnectionCoords.x * nextLayerSize.x, normalizedConnectionCoords.y * nextLayerSize.y);
		
				float nextContext = read_imagef(nextLayerContextPrev, connectionCoordsNext).x;
				
				float cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).x;

				float newCellWeight = cellWeightPrev + alpha * predictionError * nextContext;
				
				write_imagef(cellWeights, weightPosition, (float4)(newCellWeight, newCellWeight, newCellWeight, newCellWeight));
					
				wi++;
			}
		}
		
		int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float cellBiasPrev = read_imagef(cellWeightsPrev, biasPosition).x;

		float newCellBias = cellBiasPrev + alpha * predictionError;
		
		write_imagef(cellWeights, biasPosition, (float4)(newCellBias, newCellBias, newCellBias, newCellBias));
	}
}

void kernel layerCellWeightUpdateLast(read_only image2d_t columnStates, read_only image3d_t cellStatesPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev,
	write_only image3d_t cellWeights, int cellsInColumn, int2 layerSize, int2 lateralConnectionsRadii, float alpha)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	float columnPredictionPrev = read_imagef(columnPredictionsPrev, columnPosition).x;
	
	float predictionError = (columnState - columnPredictionPrev);
		
	for (int ci = 0; ci < cellsInColumn; ci++) {
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		// Go through all connections and update them
		int wi = 0;
		
		for (int dx = -lateralConnectionsRadii.x; dx <= lateralConnectionsRadii.x; dx++)
		for (int dy = -lateralConnectionsRadii.y; dy <= lateralConnectionsRadii.y; dy++) {
			int2 connectionCoords = (int2)(columnPosition.x + dx, columnPosition.y + dy);
			
			if (connectionCoords.x >= 0 && connectionCoords.x < layerSize.x && connectionCoords.y >= 0 && connectionCoords.y < layerSize.y) {	
				for (int cio = 0; cio < cellsInColumn; cio++) {
					int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
				
					float cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).x;
			
					float connectionState = read_imagef(cellStatesPrev, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
					
					float newCellWeight = cellWeightPrev + alpha * predictionError * connectionState;
					
					write_imagef(cellWeights, weightPosition, (float4)(newCellWeight, newCellWeight, newCellWeight, newCellWeight));
					
					wi++;
				}
			}
		}
		
		int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float cellBiasPrev = read_imagef(cellWeightsPrev, biasPosition).x;

		float newCellBias = cellBiasPrev + alpha * predictionError;
		
		write_imagef(cellWeights, biasPosition, (float4)(newCellBias, newCellBias, newCellBias, newCellBias));
	}
}

void kernel layerCellPredict(read_only image2d_t columnStates, read_only image3d_t cellStates, read_only image3d_t cellWeights, read_only image2d_t nextLayerContextPrev,
	write_only image3d_t cellPredictions, int cellsInColumn, int2 layerSize, int2 lateralConnectionsRadii, float2 layerSizeInv, int2 nextLayerSize)
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
			
			if (connectionCoords.x >= 0 && connectionCoords.x < layerSize.x && connectionCoords.y >= 0 && connectionCoords.y < layerSize.y) {	
				for (int cio = 0; cio < cellsInColumn; cio++) {
					int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
				
					float cellWeight = read_imagef(cellWeights, weightPosition).x;
			
					float connectionState = read_imagef(cellStates, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
					
					sum += cellWeight * connectionState;
					
					wi++;
				}
				
				int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
				
				float cellWeight = read_imagef(cellWeights, weightPosition).x;
				
				float2 normalizedConnectionCoords = (float2)(connectionCoords.x * layerSizeInv.x, connectionCoords.y * layerSizeInv.y);
				int2 connectionCoordsNext = (int2)(normalizedConnectionCoords.x * nextLayerSize.x, normalizedConnectionCoords.y * nextLayerSize.y);
		
				float nextContext = read_imagef(nextLayerContextPrev, connectionCoordsNext).x;
				
				sum += cellWeight * nextContext;
				
				wi++;
			}
		}
		
		int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float bias = read_imagef(cellWeights, biasPosition).x;
		
		sum += bias;
		
		float prediction = fmin(1.0f, fmax(0.0f, sigmoid(sum * cellPredictionIntensity) * (1.0f + 2.0f * predictionRangeExtension) - predictionRangeExtension));
		
		write_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0), prediction);
	}
}

void kernel layerCellPredictLast(read_only image2d_t columnStates, read_only image3d_t cellStates, read_only image3d_t cellWeights,
	write_only image3d_t cellPredictions, int cellsInColumn, int2 layerSize, int2 lateralConnectionsRadii)
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
			
			if (connectionCoords.x >= 0 && connectionCoords.x < layerSize.x && connectionCoords.y >= 0 && connectionCoords.y < layerSize.y) {	
				for (int cio = 0; cio < cellsInColumn; cio++) {
					int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
				
					float cellWeight = read_imagef(cellWeights, weightPosition).x;
			
					float connectionState = read_imagef(cellStates, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
					
					sum += cellWeight * connectionState;
					
					wi++;
				}
			}
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

void kernel layerRetrieveQ(read_only image2d_t columnAttentions, read_only image3d_t cellStates, read_only image3d_t cellQWeightsPrev, write_only image2d_t partialQSums, int cellsInColumn) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));

	float columnAttention = read_imagef(columnAttentions, columnPosition).x;
	
	float sum = 0.0f;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
	
		float cellQWeight = read_imagef(cellQWeightsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		sum += cellQWeight * cellState * columnAttention;
	}

	write_imagef(partialQSums, columnPosition, (float4)(sum, sum, sum, sum));
}

void kernel layerUpdateQWeights(read_only image2d_t columnAttentions, read_only image3d_t cellStates, read_only image3d_t cellQWeightsPrev, write_only image3d_t cellQWeights, int cellsInColumn, float elgibilityDecay, float tdError) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnAttention = read_imagef(columnAttentions, columnPosition).x;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
	
		float eligibility = columnAttention * cellState;
		
		float2 cellQWeightPrev = read_imagef(cellQWeightsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).xy;
		
		float2 newCellQWeight = (float2)(cellQWeightPrev.x + tdError * cellQWeightPrev.y, (1.0f - elgibilityDecay) * cellQWeightPrev.y + eligibility);
		
		write_imagef(cellQWeights, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(newCellQWeight.x, newCellQWeight.y, 0.0f, 0.0f));
	}
}

void kernel reconstructionInit(write_only image3d_t reconstructionWeights, int reconstructionNumWeights, uint2 seed, float minWeight, float maxWeight) {
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 50;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));

	for (int wi = 0; wi < reconstructionNumWeights; wi++) {
		float weight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;
		
		write_imagef(reconstructionWeights, (int4)(columnPosition.x, columnPosition.y, wi, 0), (float4)(weight, weight, weight, weight));
	}
}

void kernel reconstructInput(read_only image3d_t reconstructionWeights, read_only image2d_t sdr, write_only image2d_t inputs, int2 receptiveFieldRadius, float2 inputSizeInv, int2 sdrSize, float2 sdrSizeInv) {
	int2 inputPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputPositionNormalized = (float2)(inputPosition.x * inputSizeInv.x, inputPosition.y * inputSizeInv.y);
	
	float sum = 0.0f;
	
	int wi = 0;
	
	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		float2 sdrPositionNormalized = inputPositionNormalized + (float2)(dx * sdrSizeInv.x, dy + sdrSizeInv.y);
		int2 sdrPosition = (int2)(sdrPositionNormalized.x * sdrSize.x, sdrPositionNormalized.y * sdrSize.y);
		
		if (sdrPosition.x >= 0 && sdrPosition.x < sdrSize.x && sdrPosition.y >= 0 && sdrPosition.y < sdrSize.y) {
			float source = read_imagef(sdr, sdrPosition).x;

			float weight = read_imagef(reconstructionWeights, (int4)(inputPosition.x, inputPosition.y, wi, 0)).x;
			
			sum += source * weight;
			
			wi++;
		}
	}
	
	// Bias
	float bias = read_imagef(reconstructionWeights, (int4)(inputPosition.x, inputPosition.y, wi, 0)).x;
		
	sum += bias;
	
	float output = sum;
	
	write_imagef(inputs, inputPosition, (float4)(output, output, output, output));
}

void kernel updateReconstruction(read_only image2d_t targets, read_only image2d_t inputs, read_only image3d_t reconstructionWeightsPrev, read_only image2d_t sdr, write_only image3d_t reconstructionWeights, int2 receptiveFieldRadius, float2 inputSizeInv, int2 sdrSize, float2 sdrSizeInv, float alpha) {
	int2 inputPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputPositionNormalized = (float2)(inputPosition.x * inputSizeInv.x, inputPosition.y * inputSizeInv.y);
	
	float target = read_imagef(targets, inputPosition).x;
	float input = read_imagef(inputs, inputPosition).x;
	
	float error = alpha * (target - input);
	
	int wi = 0;
	
	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		float2 sdrPositionNormalized = inputPositionNormalized + (float2)(dx * sdrSizeInv.x, dy + sdrSizeInv.y);
		int2 sdrPosition = (int2)(sdrPositionNormalized.x * sdrSize.x, sdrPositionNormalized.y * sdrSize.y);
		
		if (sdrPosition.x >= 0 && sdrPosition.x < sdrSize.x && sdrPosition.y >= 0 && sdrPosition.y < sdrSize.y) {
			float source = read_imagef(sdr, sdrPosition).x;

			float prevWeight = read_imagef(reconstructionWeightsPrev, (int4)(inputPosition.x, inputPosition.y, wi, 0)).x;
			
			float newWeight = prevWeight + error * source;
			
			write_imagef(reconstructionWeights, (int4)(inputPosition.x, inputPosition.y, wi, 0), (float4)(newWeight, newWeight, newWeight, newWeight));
			
			wi++;
		}
	}
	
	// Bias
	float prevBias = read_imagef(reconstructionWeightsPrev, (int4)(inputPosition.x, inputPosition.y, wi, 0)).x;
		
	float newBias = prevBias + error;
	
	write_imagef(reconstructionWeights, (int4)(inputPosition.x, inputPosition.y, wi, 0), (float4)(newBias, newBias, newBias, newBias));
}