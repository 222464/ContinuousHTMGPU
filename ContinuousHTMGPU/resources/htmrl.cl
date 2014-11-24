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
	
constant float activationIntensity = 100.0f;
constant float columnIntensity = 4.0f;
constant float cellStateIntensity = 4.0f;
constant float cellPredictionIntensity = 4.0f;
constant float minLearningThreshold = 0.0f;
constant float predictionRangeExtension = 0.1f;
constant float localActivity = 4.0f;
constant float sensitivityDecay = 0.01f;
constant float sensitivityIntensity = 1.0f;
constant float sensitivityIncrease = 1.0f;
constant float sensitivityThreshold = 0.5f;
constant float dutyCycleDecay = 0.01f;
constant float uniquenessPower = 4.0f;
constant float subActivation = 0.2f;

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

void kernel initializePartOne(write_only image2d_t columnActivations, write_only image2d_t columnStates, write_only image3d_t columnWeights, write_only image2d_t columnSensitivities, write_only image2d_t columnDutyCycles,
	int cellsInColumn, int receptiveFieldSize, int lateralConnectionsSize, uint2 seed, float minWeight, float maxWeight, float minSensitivity, float maxSensitivity)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 100;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));

	write_imagef(columnActivations, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	write_imagef(columnStates, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	write_imagef(columnDutyCycles, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));

	float sensitivity = randFloat(&seedValue) * (maxSensitivity - minSensitivity) + minSensitivity;
	
	write_imagef(columnSensitivities, columnPosition, (float4)(sensitivity, 0.0f, 0.0f, 0.0f));
	
	float columnQUsage = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;
	
	for (int wi = 0; wi < receptiveFieldSize; wi++) {
		int4 weightPosition = (int4)(columnPosition.x, columnPosition.y, wi, 0);
	
		float columnConnectionWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;

		write_imagef(columnWeights, weightPosition, (float4)(columnConnectionWeight, 0.0f, 0.0f, 0.0f));
	}
}

void kernel initializePartTwo(write_only image3d_t cellStates, write_only image3d_t cellWeights, write_only image3d_t cellPredictions, write_only image3d_t cellQWeights,
	int cellsInColumn, int receptiveFieldSize, int lateralConnectionsSize, uint2 seed, float minWeight, float maxWeight, float minSensitivity, float maxSensitivity)
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

void kernel layerColumnActivate(read_only image2d_t columnStatesInput, read_only image2d_t columnSensitivitiesPrev, read_only image3d_t columnWeightsPrev, write_only image2d_t columnActivations, float2 layerSizeInv, int2 inputReceptiveFieldRadius, float2 inputReceptiveFieldStep, int2 inputSize, uint2 seed) {
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 20;
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeInv.x, columnPosition.y * layerSizeInv.y);

	float sum = 0.0f;

	int weightIndex = 0;
	
	float thisSensitivity = read_imagef(columnSensitivitiesPrev, columnPosition).x;
	
	for (int dx = -inputReceptiveFieldRadius.x; dx <= inputReceptiveFieldRadius.x; dx++)
	for (int dy = -inputReceptiveFieldRadius.y; dy <= inputReceptiveFieldRadius.y; dy++) {
		float2 inputPositionNormalized = inputCenterPositionNormalized + (float2)(dx * inputReceptiveFieldStep.x, dy * inputReceptiveFieldStep.y);
		int2 inputPosition = (int2)(inputPositionNormalized.x * (float)(inputSize.x), inputPositionNormalized.y * (float)(inputSize.y));
		
		if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
			float weight = read_imagef(columnWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
			float inputState = read_imagef(columnStatesInput, inputPosition).x;
				
			float difference = weight - inputState;
				
			sum += difference * difference;
		}
		
		weightIndex++;
	}

	float activation = exp(-sum * (sigmoid(thisSensitivity * sensitivityIntensity) * 2.0f - 1.0f) * activationIntensity);//(randFloat(&seedValue) < (randomFireChance * thisDutyCycle.y) ? randomFireStrength : 0.0f));
	
	write_imagef(columnActivations, columnPosition, (float4)(activation, activation, activation, activation));
}

void kernel layerColumnInhibit(read_only image2d_t columnActivations, write_only image2d_t columnStates, int2 layerSize, float2 layerSizeInv, int2 receptiveFieldRadius, float2 layerReceptiveFieldStep, uint2 seed) {
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 50;
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float thisActivation = read_imagef(columnActivations, columnPosition).x;
	
	float higherSum = 0.0f;
	
	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		int2 layerPosition = columnPosition + (int2)(dx, dy);
	
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			float activation = read_imagef(columnActivations, layerPosition).x;
			
			if (activation >= thisActivation)
				higherSum++;
		}
	}
	
	float inhibitedResult = sigmoid((localActivity - higherSum) * columnIntensity) * thisActivation;// * fmax(thisActivation, subActivation);
	
	write_imagef(columnStates, columnPosition, (float4)(inhibitedResult, inhibitedResult, inhibitedResult, inhibitedResult));
}

void kernel layerColumnWeightUpdate(read_only image2d_t columnStatesInput, read_only image2d_t columnActivations, read_only image2d_t columnSensitivitiesPrev, read_only image2d_t columnStates, read_only image3d_t columnWeightsPrev, read_only image2d_t columnDutyCyclesPrev, write_only image2d_t columnSensitivities, write_only image3d_t columnWeights, write_only image2d_t columnDutyCycles, int2 layerSize, float2 layerSizeInv, int2 inputReceptiveFieldRadius, float2 inputReceptiveFieldStep, int2 inputSize, float connectionAlpha, uint2 seed) {
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 130;
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeInv.x, columnPosition.y * layerSizeInv.y);

	float thisState = read_imagef(columnStates, columnPosition).x;
	
	float thisSensitivity = read_imagef(columnSensitivitiesPrev, columnPosition).x;
		
	float newSensitivity = (1.0f - sensitivityDecay) * thisSensitivity + sensitivityIncrease * thisState;
		
	write_imagef(columnSensitivities, columnPosition, (float4)(newSensitivity, newSensitivity, newSensitivity, newSensitivity));
		
	float learnScalar = fmin(1.0f, fmax(0.0f, thisState - minLearningThreshold) + fmax(0.0f, sensitivityThreshold - newSensitivity) / sensitivityThreshold);
		
	// Adjust weights by their source activations and error
	//int weightIndex = 0;
	
	for (int dx = -inputReceptiveFieldRadius.x; dx <= inputReceptiveFieldRadius.x; dx++)
	for (int dy = -inputReceptiveFieldRadius.y; dy <= inputReceptiveFieldRadius.y; dy++) {
		float2 inputPositionNormalized = (float2)(inputCenterPositionNormalized.x + dx * inputReceptiveFieldStep.x, inputCenterPositionNormalized.y + dy * inputReceptiveFieldStep.y);
		int2 inputPosition = (int2)(inputPositionNormalized.x * (float)(inputSize.x), inputPositionNormalized.y * (float)(inputSize.y));
		
		if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
			float inputState = read_imagef(columnStatesInput, inputPosition).x;
				
			int weightIndex = (dy + inputReceptiveFieldRadius.y) + (dx + inputReceptiveFieldRadius.x) * (2 * inputReceptiveFieldRadius.y + 1);
		
			float prevWeight = read_imagef(columnWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
				
			float difference = inputState - prevWeight;
				
			float change = connectionAlpha * learnScalar * difference;
				
			float newWeight = prevWeight + change;
				
			write_imagef(columnWeights, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0), (float4)(newWeight, newWeight, newWeight, newWeight));
		}
		
		//weightIndex++;
	}
	
	// Update duty cycle
	float dutyCyclePrev = read_imagef(columnDutyCyclesPrev, columnPosition).x;
	
	float newDutyCycle = (1.0f - dutyCycleDecay) * dutyCyclePrev + dutyCycleDecay * thisState;
	
	write_imagef(columnDutyCycles, columnPosition, (float4)(newDutyCycle, newDutyCycle, newDutyCycle, newDutyCycle));
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
				int2 connectionCoordsNext = (int2)(normalizedConnectionCoords.x * (float)(nextLayerSize.x), normalizedConnectionCoords.y * (float)(nextLayerSize.y));
		
				float nextContextPrev = read_imagef(nextLayerContextPrev, connectionCoordsNext).x;
				
				float cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).x;

				float newCellWeight = cellWeightPrev + alpha * predictionError * nextContextPrev;
				
				write_imagef(cellWeights, weightPosition, (float4)(newCellWeight, newCellWeight, newCellWeight, newCellWeight));
					
				wi++;
			}
		}
		
		/*int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float cellBiasPrev = read_imagef(cellWeightsPrev, biasPosition).x;

		float newCellBias = cellBiasPrev + alpha * predictionError;
		
		write_imagef(cellWeights, biasPosition, (float4)(newCellBias, newCellBias, newCellBias, newCellBias));*/
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
		
		/*int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float cellBiasPrev = read_imagef(cellWeightsPrev, biasPosition).x;

		float newCellBias = cellBiasPrev + alpha * predictionError;
		
		write_imagef(cellWeights, biasPosition, (float4)(newCellBias, newCellBias, newCellBias, newCellBias));*/
	}
}

void kernel layerCellPredict(read_only image2d_t columnStates, read_only image3d_t cellStates, read_only image3d_t cellWeights, read_only image2d_t nextLayerContext,
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
				int2 connectionCoordsNext = (int2)(normalizedConnectionCoords.x * (float)(nextLayerSize.x), normalizedConnectionCoords.y * (float)(nextLayerSize.y));
		
				float nextContext = read_imagef(nextLayerContext, connectionCoordsNext).x;
				
				sum += cellWeight * nextContext;
				
				wi++;
			}
		}
		
		/*int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float bias = read_imagef(cellWeights, biasPosition).x;
		
		sum += bias;*/
		
		float prediction = fmin(1.0f, fmax(0.0f, (sigmoid(sum * cellPredictionIntensity) * 2.0f - 1.0f) * (1.0f + 2.0f * predictionRangeExtension) - predictionRangeExtension));
		
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
		
		/*int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float bias = read_imagef(cellWeights, biasPosition).x;
		
		sum += bias;*/
		
		float prediction = fmin(1.0f, fmax(0.0f, (sigmoid(sum * cellPredictionIntensity) * 2.0f - 1.0f) * (1.0f + 2.0f * predictionRangeExtension) - predictionRangeExtension));
		
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

void kernel layerRetrieveQ(read_only image2d_t columnStates, read_only image2d_t columnDutyCycles, read_only image3d_t cellStates, read_only image3d_t cellQWeightsPrev, write_only image2d_t partialQSums, int cellsInColumn) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));

	float state = read_imagef(columnStates, columnPosition).x;
	
	//float dutyCycle = read_imagef(columnDutyCycles, columnPosition).x;
	
	//float uniqueness = pow(fabs(state - dutyCycle), uniquenessPower);
	
	float sum = 0.0f;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
	
		float cellQWeight = read_imagef(cellQWeightsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		sum += cellQWeight * cellState;// * uniqueness;
	}
	
	write_imagef(partialQSums, columnPosition, (float4)(sum, sum, sum, sum));
}

void kernel layerUpdateQWeights(read_only image2d_t columnStates, read_only image2d_t columnDutyCycles, read_only image3d_t cellStates, read_only image3d_t cellQWeightsPrev, write_only image3d_t cellQWeights, int cellsInColumn, float eligibilityDecay, float tdError) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float state = read_imagef(columnStates, columnPosition).x;
	
	//float dutyCycle = read_imagef(columnDutyCycles, columnPosition).x;
	
	//float uniqueness = pow(fabs(state - dutyCycle), uniquenessPower);
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
	
		float eligibility = cellState;// * uniqueness
		
		float2 cellQWeightPrev = read_imagef(cellQWeightsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).xy;
		
		float newTrace = (1.0f - eligibility) * (1.0f - eligibilityDecay) * cellQWeightPrev.y + eligibility;
 
		float2 newCellQWeight = (float2)(cellQWeightPrev.x + tdError * cellQWeightPrev.y, newTrace);

		write_imagef(cellQWeights, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(newCellQWeight.x, newCellQWeight.y, 0.0f, 0.0f));
	}
}

void kernel reconstructInput(read_only image2d_t inputs, read_only image2d_t sdr, read_only image3d_t columnWeights, read_only image2d_t columnSensitivities, write_only image2d_t reconstruction, int2 reconstructionReceptiveFieldRadius, int2 sdrReceptiveFieldRadius, float2 inputSizeInv, int2 sdrSize, float2 sdrSizeInv, float2 inputOverSdr) {
	int2 inputPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputPositionNormalized = (float2)(inputPosition.x * inputSizeInv.x, inputPosition.y * inputSizeInv.y);
	
	float input = read_imagef(inputs, inputPosition).x;
	
	float sum = 0.0f;
	float total = 0.0f;
	
	for (int dx = -reconstructionReceptiveFieldRadius.x; dx <= reconstructionReceptiveFieldRadius.x; dx++)
	for (int dy = -reconstructionReceptiveFieldRadius.y; dy <= reconstructionReceptiveFieldRadius.y; dy++) {
		float2 sdrPositionNormalized = (float2)(inputPositionNormalized.x + dx * sdrSizeInv.x, inputPositionNormalized.y + dy * sdrSizeInv.y);
		int2 sdrPosition = (int2)(sdrPositionNormalized.x * (float)(sdrSize.x), sdrPositionNormalized.y * (float)(sdrSize.y));
		
		int scaledDx = (float)(dx) * inputOverSdr.x;
		int scaledDy = (float)(dy) * inputOverSdr.y;
		
		if (sdrPosition.x >= 0 && sdrPosition.x < sdrSize.x && sdrPosition.y >= 0 && sdrPosition.y < sdrSize.y) {
			float source = read_imagef(sdr, sdrPosition).x;
			
			int weightIndex = (sdrReceptiveFieldRadius.y - scaledDy) + (sdrReceptiveFieldRadius.x - scaledDx) * (sdrReceptiveFieldRadius.y * 2 + 1);

			float weight = read_imagef(columnWeights, (int4)(sdrPosition.x, sdrPosition.y, weightIndex, 0)).x;

			float sensitivity = read_imagef(columnSensitivities, sdrPosition).x;
			
			sum += source * weight;
			
			total += source;
		}
	}
	
	float output;
	
	if (total == 0.0f)
		output = 0.0f;
	else
		output = sum / total;
	
	write_imagef(reconstruction, inputPosition, (float4)(output, output, output, output));
}