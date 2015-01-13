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
	
constant float columnIntensity = 8.0f;
constant float autoFireIntensity = 0.1f;
constant float learnIntensity = 0.3f;
constant float modulationPower = 1.0f;
constant float crowdingIntensity = 3.0f;
constant float cellStateIntensity = 32.0f;
constant float cellPredictionIntensity = 4.0f;
constant float minLearningThreshold = 0.0f;
constant float predictionRangeExtension = 0.1f;
constant float localActivity = 3.0f;
constant float crowdingActivity = 5.0f;
constant float uniquenessPower = 4.0f;
constant float minOverlapForActivation = 0.0f;
constant float subOverlapIncrement = 0.0005f;
constant float boostDutyCycleRatio = 0.01f;
constant float maxBoost = 1.0f;
constant float typicalMaxActivationRatio = 0.5f;
constant float widthScalar = 1.0f;
constant float minWidth = 0.0001f;
constant float minBoostThreshold = 0.0001f;
constant float nodeOutputIntensity = 1.0f;
constant float rectifierLeak = 0.001f;

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

float relu(float x) {
	return log(1.0f + exp(x));
}

float rectifierDerivative(float x) {
	return x > rectifierLeak ? 1.0f : rectifierLeak;
}

float scaledSigmoid(float x) {
	return 2.0f / (1.0f + exp(-x)) - 1.0f;
}

float boostFunction(float dutyCycle, float threshold) {
	return fmin(1.0f, fmax(0.0f, threshold - dutyCycle) / threshold);//fmax(threshold, minBoostThreshold) * boostIntensity);
}

void kernel initializePartOne(write_only image2d_t columnActivations, write_only image2d_t columnStates, write_only image3d_t columnWeights, write_only image2d_t columnDutyCycles,
	int cellsInColumn, int receptiveFieldSize, int lateralConnectionsSize, uint2 seed, float minWeight, float maxWeight, float minWidth, float maxWidth)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 - 12, get_global_id(1) * 16 + 23) * 36;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));

	write_imagef(columnActivations, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	write_imagef(columnStates, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	write_imagef(columnDutyCycles, columnPosition, (float4)(1.0f, 0.0f, 0.0f, 0.0f));

	for (int wi = 0; wi < receptiveFieldSize; wi++) {
		int4 weightPosition = (int4)(columnPosition.x, columnPosition.y, wi, 0);
	
		float columnConnectionWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;

		write_imagef(columnWeights, weightPosition, (float4)(columnConnectionWeight, 0.0f, 0.0f, 0.0f));
	}
	
	int4 widthPosition = (int4)(columnPosition.x, columnPosition.y, receptiveFieldSize, 0);
	
	float columnWidth = randFloat(&seedValue) * (maxWidth - minWidth) + minWidth;
	
	write_imagef(columnWeights, widthPosition, (float4)(columnWidth, 0.0f, 0.0f, 0.0f));
}

void kernel initializePartTwo(write_only image3d_t cellStates, write_only image3d_t cellWeights, write_only image3d_t cellPredictions,
	int cellsInColumn, int receptiveFieldSize, int lateralConnectionsSize, uint2 seed, float minWeight, float maxWeight)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 32 + 24, get_global_id(1) * 11 - 66) * 23;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		write_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
		write_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
		
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
	
		for (int wi = 0; wi < lateralConnectionsSize; wi++) {
			int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
			float cellWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;
	
			write_imagef(cellWeights, weightPosition, (float4)(cellWeight, 0.0f, 0.0f, 0.0f));
		}
	}
}

void kernel initializePartThree(write_only image3d_t nodeOutputs, write_only image3d_t nodeErrors, write_only image3d_t nodeBiases, write_only image3d_t nodeWeights,
	int cellsInColumn, int receptiveFieldSize, int lateralConnectionsSize, uint2 seed, float minWeight, float maxWeight)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 67 + 12, get_global_id(1) * 9 - 11) * 12;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
		
	for (int ci = 0; ci < cellsInColumn; ci++) {
		write_imagef(nodeOutputs, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
		write_imagef(nodeErrors, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
		
		float biasWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;

		write_imagef(nodeBiases, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(biasWeight, 0.0f, 0.0f, 0.0f));
		
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
	
		for (int wi = 0; wi < receptiveFieldSize; wi++) {
			int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
			float weight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;
	
			write_imagef(nodeWeights, weightPosition, (float4)(weight, 0.0f, 0.0f, 0.0f));
		}
	}
}

void kernel initializePartFour(write_only image3d_t outputWeights, 
	int cellsInColumn, uint2 seed, float minWeight, float maxWeight)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 77 - 45, get_global_id(1) * 24 + 66) * 61;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float weight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;

		write_imagef(outputWeights, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(weight, 0.0f, 0.0f, 0.0f));
	}
}

void kernel layerColumnActivate(read_only image2d_t columnStatesInput, read_only image3d_t columnWeightsPrev, read_only image2d_t columnDutyCyclesPrev, write_only image2d_t columnActivations,
	float2 layerSizeMinusOneInv, int2 inputReceptiveFieldRadius, int2 inputSize, int2 inputSizeMinusOne, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 20;
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	float2 inputCenterPosition = (float2)(inputCenterPositionNormalized.x * inputSizeMinusOne.x, inputCenterPositionNormalized.y * inputSizeMinusOne.y);

	float sum = 0.0f;

	int weightIndex = 0;
	int usageCount = 0;
	
	for (int dx = -inputReceptiveFieldRadius.x; dx <= inputReceptiveFieldRadius.x; dx++)
	for (int dy = -inputReceptiveFieldRadius.y; dy <= inputReceptiveFieldRadius.y; dy++) {
		int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
		
		if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
			float weight = read_imagef(columnWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
			float inputState = read_imagef(columnStatesInput, inputPosition).x;
				
			//float modulation = pow((float)(abs(dx) + abs(dy)) / (float)(inputReceptiveFieldRadius.x + inputReceptiveFieldRadius.y), modulationPower);
				
			sum += inputState * weight;
			
			usageCount++;
		}
		
		weightIndex++;
	}
	
	//float width = read_imagef(columnWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
	
	//float boost = read_imagef(columnDutyCyclesPrev, columnPosition).y;

	float output = sum * (float)(weightIndex) / (float)(usageCount);// * (1.0f - boost);

	write_imagef(columnActivations, columnPosition, (float4)(output, sum, 0.0f, 0.0f));
}

void kernel layerColumnInhibit(read_only image2d_t columnActivations, read_only image2d_t columnDutyCyclesPrev, write_only image2d_t columnStates, int2 layerSize, float2 layerSizeInv, int2 receptiveFieldRadius) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float thisActivation = read_imagef(columnActivations, columnPosition).x;

	float higherSum = 0.0f;

	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		int2 layerPosition = (int2)(columnPosition.x + dx, columnPosition.y + dy);
	
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			float activation = read_imagef(columnActivations, layerPosition).x;
			
			if (activation >= thisActivation) {
				higherSum++;
			}
		}
	}
	
	float boost = read_imagef(columnDutyCyclesPrev, columnPosition).y;
	
	float inhibitedResult = fmax(0.0f, sigmoid((localActivity - higherSum) * columnIntensity) * 2.0f - 1.0f);//exp((thisActivation - maxActivation) * columnIntensity);//exp(-higherSum * columnIntensity);// //

	write_imagef(columnStates, columnPosition, (float4)(inhibitedResult, 0.0f, 0.0f, 0.0f));
}

void kernel layerColumnDutyCycleUpdate(read_only image2d_t columnActivations, read_only image2d_t columnStates, read_only image2d_t columnDutyCyclesPrev, write_only image2d_t columnDutyCycles,
	int2 layerSize, int2 receptiveFieldRadius, float activationDutyCycleDecay, float stateDutyCycleDecay)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float stateSum = 0.0f;
	
	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		int2 layerPosition = columnPosition + (int2)(dx, dy);
	
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			float state = read_imagef(columnStates, layerPosition).x;
			
			stateSum += state;
		}
	}
	
	//float thisActivation = read_imagef(columnActivations, columnPosition).x;
	float thisState = read_imagef(columnStates, columnPosition).x;
	
	float thisDutyCycle = read_imagef(columnDutyCyclesPrev, columnPosition).x;
		
	//float newDutyCycle = fmax(minDutyCycleForBoost, fmax((1.0f - stateDutyCycleDecay) * thisDutyCycle, thisState));
	
	float newDutyCycle = fmax((1.0f - stateDutyCycleDecay) * thisDutyCycle, thisState);
		
	//float boost = fmax(0.0f, typicalMaxActivationRatio - exp(maxActivation * stability));//boostFunction(newDutyCycle, (1.0f - exp(maxActivation * stability)) * boostDutyCycleRatio);
	float crowding = sigmoid((crowdingActivity - stateSum) * crowdingIntensity);// * maxBoost * boostFunction(newDutyCycle, boostDutyCycleRatio));//
		
	write_imagef(columnDutyCycles, columnPosition, (float4)(newDutyCycle, crowding, 0.0f, 0.0f));
}

void kernel layerColumnWeightUpdate(read_only image2d_t columnStatesInput, read_only image2d_t columnActivations, read_only image2d_t columnStates, read_only image3d_t columnWeightsPrev, read_only image2d_t columnDutyCyclesPrev, write_only image3d_t columnWeights,
	float2 layerSizeMinusOneInv, int2 inputReceptiveFieldRadius, int2 inputSize, int2 inputSizeMinusOne, float connectionAlpha, float widthAlpha, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 130;
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	float2 inputCenterPosition = (float2)(inputCenterPositionNormalized.x * inputSizeMinusOne.x, inputCenterPositionNormalized.y * inputSizeMinusOne.y);
	
	float2 thisState = read_imagef(columnStates, columnPosition).xy;
	
	float2 thisActivation = read_imagef(columnActivations, columnPosition).xy;
	
	float2 dutyCyclePrev = read_imagef(columnDutyCyclesPrev, columnPosition).xy;
	
	//float globalIncrement = fmax(0.0f, minOverlapForActivation - dutyCyclePrev.x) * subOverlapIncrement;
	
	float boost = boostFunction(dutyCyclePrev.x, boostDutyCycleRatio);
	
	float error = (1.0f - dutyCyclePrev.y) * ((1.0f - boost) * thisState.x + boost - thisActivation.x);
	
	// Adjust weights by their source activations and error
	int weightIndex = 0;
	
	for (int dx = -inputReceptiveFieldRadius.x; dx <= inputReceptiveFieldRadius.x; dx++)
	for (int dy = -inputReceptiveFieldRadius.y; dy <= inputReceptiveFieldRadius.y; dy++) {
		int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
		
		if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
			float inputState = read_imagef(columnStatesInput, inputPosition).x;
				
			float prevWeight = read_imagef(columnWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
				
			//float modulation = (float)(abs(dx) + abs(dy)) / (inputReceptiveFieldRadius.x + inputReceptiveFieldRadius.y);
				
			float newWeight = prevWeight + connectionAlpha * error * inputState;
				
			write_imagef(columnWeights, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0), (float4)(newWeight, newWeight, newWeight, newWeight));
		}
		
		weightIndex++;
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

void kernel layerCellWeightUpdate(read_only image2d_t columnStates, read_only image3d_t cellStates, read_only image3d_t cellStatesPrev, read_only image2d_t nextLayerContextPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev,
	write_only image3d_t cellWeights, int cellsInColumn, int2 layerSize, int2 lateralConnectionsRadii, float2 layerSizeMinusOneInv, int2 nextLayerSize, int2 nextLayerSizeMinusOne, float alpha, float beta, float temperature, float eligibilityDecay)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 normalizedColumnCoords = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	int2 connectionCoordsNextCenter = (int2)(normalizedColumnCoords.x * nextLayerSizeMinusOne.x, normalizedColumnCoords.y * nextLayerSizeMinusOne.y);
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	float columnPredictionPrev = read_imagef(columnPredictionsPrev, columnPosition).x;
	
	float predictionError = alpha * (columnState - columnPredictionPrev);// * (columnPredictionPrev * 0.5f + 0.5f) * (1.0f - (columnPredictionPrev * 0.5f + 0.5f));
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float cellError = predictionError;
		
		// Go through all connections and update them
		int wi = 0;
		
		for (int dx = -lateralConnectionsRadii.x; dx <= lateralConnectionsRadii.x; dx++)
		for (int dy = -lateralConnectionsRadii.y; dy <= lateralConnectionsRadii.y; dy++) {
			int2 connectionCoords = (int2)(columnPosition.x + dx, columnPosition.y + dy);
			
			if (connectionCoords.x >= 0 && connectionCoords.x < layerSize.x && connectionCoords.y >= 0 && connectionCoords.y < layerSize.y) {	
				for (int cio = 0; cio < cellsInColumn; cio++) {
					int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
				
					float2 cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).x;
			
					float connectionState = read_imagef(cellStatesPrev, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
					
					float eligibility = cellError * connectionState;
					
					float2 newCellWeight = (float2)(cellWeightPrev.x + alpha * cellWeightPrev.y, (1.0f - eligibilityDecay) * cellWeightPrev.y + beta * exp(-temperature * fabs(eligibility)) * eligibility);
					
					write_imagef(cellWeights, weightPosition, (float4)(newCellWeight.x, newCellWeight.y, 0.0f, 0.0f));
					
					wi++;
				}
				
				int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
				
				// Additional context from next layer
				int2 connectionCoordsNext = (int2)(connectionCoordsNextCenter.x + dx, connectionCoordsNextCenter.y + dy);
			
				if (connectionCoordsNext.x >= 0 && connectionCoordsNext.x < nextLayerSize.x && connectionCoordsNext.y >= 0 && connectionCoordsNext.y < nextLayerSize.y) {
					float nextContextPrev = read_imagef(nextLayerContextPrev, connectionCoordsNext).x;
					
					float2 cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).xy;

					float eligibility = cellError * nextContextPrev;
					
					float2 newCellWeight = (float2)(cellWeightPrev.x + alpha * cellWeightPrev.y, (1.0f - eligibilityDecay) * cellWeightPrev.y + beta * exp(-temperature * fabs(eligibility)) * eligibility);
					
					write_imagef(cellWeights, weightPosition, (float4)(newCellWeight.x, newCellWeight.y, 0.0f, 0.0f));
				}
				
				wi++;
			}
			else
				wi += cellsInColumn + 1;
		}
		
		int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float2 cellBiasPrev = read_imagef(cellWeightsPrev, biasPosition).xy;

		float eligibility = cellError;
		
		float2 newCellBias = (float2)(cellBiasPrev.x + alpha * cellBiasPrev.y, (1.0f - eligibilityDecay) * cellBiasPrev.y + beta * exp(-temperature * fabs(eligibility)) * eligibility);

		write_imagef(cellWeights, biasPosition, (float4)(newCellBias.x, newCellBias.y, 0.0f, 0.0f));
	}
}

void kernel layerCellWeightUpdateLast(read_only image2d_t columnStates, read_only image3d_t cellStates, read_only image3d_t cellStatesPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev,
	write_only image3d_t cellWeights, int cellsInColumn, int2 layerSize, int2 lateralConnectionsRadii, float alpha, float beta, float temperature, float eligibilityDecay)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	float columnPredictionPrev = read_imagef(columnPredictionsPrev, columnPosition).x;
	
	float predictionError = alpha * (columnState - columnPredictionPrev);// * (columnPredictionPrev * 0.5f + 0.5f) * (1.0f - (columnPredictionPrev * 0.5f + 0.5f));
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float cellError = predictionError;
		
		// Go through all connections and update them
		int wi = 0;
		
		for (int dx = -lateralConnectionsRadii.x; dx <= lateralConnectionsRadii.x; dx++)
		for (int dy = -lateralConnectionsRadii.y; dy <= lateralConnectionsRadii.y; dy++) {
			int2 connectionCoords = (int2)(columnPosition.x + dx, columnPosition.y + dy);
			
			if (connectionCoords.x >= 0 && connectionCoords.x < layerSize.x && connectionCoords.y >= 0 && connectionCoords.y < layerSize.y) {	
				for (int cio = 0; cio < cellsInColumn; cio++) {
					int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
				
					float2 cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).xy;
			
					float connectionState = read_imagef(cellStatesPrev, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
						
					float eligibility = cellError * connectionState;
					
					float2 newCellWeight = (float2)(cellWeightPrev.x + alpha * cellWeightPrev.y, (1.0f - eligibilityDecay) * cellWeightPrev.y + beta * exp(-temperature * fabs(eligibility)) * eligibility);
					
					write_imagef(cellWeights, weightPosition, (float4)(newCellWeight.x, newCellWeight.y, 0.0f, 0.0f));
					
					wi++;
				}
			}
			else
				wi += cellsInColumn;
		}
		
		int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float2 cellBiasPrev = read_imagef(cellWeightsPrev, biasPosition).xy;

		float eligibility = cellError;
		
		float2 newCellBias = (float2)(cellBiasPrev.x + alpha * cellBiasPrev.y, (1.0f - eligibilityDecay) * cellBiasPrev.y + beta * exp(-temperature * fabs(eligibility)) * eligibility);
			
		write_imagef(cellWeights, biasPosition, (float4)(newCellBias.x, newCellBias.y, 0.0f, 0.0f));
	}
}

void kernel layerCellPredict(read_only image2d_t columnStates, read_only image3d_t cellStates, read_only image3d_t cellWeights, read_only image2d_t nextLayerContext,
	write_only image3d_t cellPredictions, int cellsInColumn, int2 layerSize, int2 lateralConnectionsRadii, float2 layerSizeMinusOneInv, int2 nextLayerSize, int2 nextLayerSizeMinusOne)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 normalizedColumnCoords = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	int2 connectionCoordsNextCenter = (int2)(normalizedColumnCoords.x * nextLayerSizeMinusOne.x, normalizedColumnCoords.y * nextLayerSizeMinusOne.y);
		
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
				
				int2 connectionCoordsNext = (int2)(connectionCoordsNextCenter.x + dx, connectionCoordsNextCenter.y + dy);
				
				if (connectionCoordsNext.x >= 0 && connectionCoordsNext.x < nextLayerSize.x && connectionCoordsNext.y >= 0 && connectionCoordsNext.y < nextLayerSize.y) {
					float nextContext = read_imagef(nextLayerContext, connectionCoordsNext).x;
					
					sum += cellWeight * nextContext;
				}
				
				wi++;
			}
			else
				wi += cellsInColumn + 1; // + 1 for context from higher layer
		}
		
		int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float bias = read_imagef(cellWeights, biasPosition).x;
		
		sum += bias;
		
		float prediction = sigmoid(sum * cellPredictionIntensity);
		
		write_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(prediction, prediction, prediction, prediction));
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
			else
				wi += cellsInColumn;
		}
		
		int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float bias = read_imagef(cellWeights, biasPosition).x;
		
		sum += bias;
		
		float prediction = sigmoid(sum * cellPredictionIntensity);
		
		write_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(prediction, prediction, prediction, prediction));
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

void kernel layerNodeActivate(read_only image3d_t nodeStatesInput, read_only image3d_t cellStates, read_only image3d_t nodeBiases, read_only image3d_t nodeWeights, write_only image3d_t nodeOutputs,
	int cellsInColumn, int inputCellsPerColumn, int2 nodeFieldRadius, float2 layerSizeMinusOneInv, int2 inputSize, int2 inputSizeMinusOne)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	float2 inputCenterPosition = (float2)(inputCenterPositionNormalized.x * inputSizeMinusOne.x, inputCenterPositionNormalized.y * inputSizeMinusOne.y);
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float sum = 0.0f;
		
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		// Go through all connections 
		int wi = 0;
		
		for (int dx = -nodeFieldRadius.x; dx <= nodeFieldRadius.x; dx++)
		for (int dy = -nodeFieldRadius.y; dy <= nodeFieldRadius.y; dy++) {
			int2 connectionCoords = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
			
			if (connectionCoords.x >= 0 && connectionCoords.x < inputSize.x && connectionCoords.y >= 0 && connectionCoords.y < inputSize.y) {	
				for (int cio = 0; cio < inputCellsPerColumn; cio++) {
					int weightCoord = cio + (nodeFieldRadius.y + dy) * inputCellsPerColumn + (nodeFieldRadius.x + dx) * inputCellsPerColumn * (nodeFieldRadius.y * 2 + 1);
					int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, weightCoord, 0);
				
					float nodeWeight = read_imagef(nodeWeights, weightPosition).x;
			
					float connectionState = read_imagef(nodeStatesInput, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
					
					sum += nodeWeight * connectionState;
					
					wi++;
				}
			}
			else
				wi += inputCellsPerColumn;
		}
		
		float bias = read_imagef(nodeBiases, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		sum += bias;
		
		float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float nodeOutput = relu(sum * nodeOutputIntensity) * cellState;
		
		write_imagef(nodeOutputs, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(nodeOutput, fmax(rectifierLeak, sigmoid(sum * nodeOutputIntensity)) * cellState, 0.0f, 0.0f));
	}
}

void kernel layerNodeActivateFirst(read_only image2d_t statesInput, read_only image3d_t cellStates, read_only image3d_t nodeBiases, read_only image3d_t nodeWeights, write_only image3d_t nodeOutputs,
	int cellsInColumn, int2 nodeFieldRadius, float2 layerSizeMinusOneInv, int2 inputSize, int2 inputSizeMinusOne)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	float2 inputCenterPosition = (float2)(inputCenterPositionNormalized.x * inputSizeMinusOne.x, inputCenterPositionNormalized.y * inputSizeMinusOne.y);
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float sum = 0.0f;
		
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		// Go through all connections 
		int wi = 0;
		
		for (int dx = -nodeFieldRadius.x; dx <= nodeFieldRadius.x; dx++)
		for (int dy = -nodeFieldRadius.y; dy <= nodeFieldRadius.y; dy++) {
			int2 connectionCoords = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
			
			if (connectionCoords.x >= 0 && connectionCoords.x < inputSize.x && connectionCoords.y >= 0 && connectionCoords.y < inputSize.y) {
				int weightCoord = (nodeFieldRadius.y + dy) + (nodeFieldRadius.x + dx) * (nodeFieldRadius.y * 2 + 1);
				int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, weightCoord, 0);
			
				float nodeWeight = read_imagef(nodeWeights, weightPosition).x;
		
				float connectionState = read_imagef(statesInput, connectionCoords).x;
				
				sum += nodeWeight * connectionState;
			}
			
			wi++;
		}
		
		float bias = read_imagef(nodeBiases, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		sum += bias;
		
		float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float nodeOutput = relu(sum * nodeOutputIntensity) * cellState;
		
		write_imagef(nodeOutputs, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(nodeOutput, fmax(rectifierLeak, sigmoid(sum * nodeOutputIntensity)) * cellState, 0.0f, 0.0f));
	}
}

void kernel weighOutput(read_only image3d_t statesInput, read_only image3d_t weights, write_only image2d_t partialSums, int inputCellsPerColumn) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float partialSum = 0.0f;
	
	for (int ci = 0; ci < inputCellsPerColumn; ci++) {
		float nodeState = read_imagef(statesInput, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float weight = read_imagef(weights, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		partialSum += nodeState * weight;
	}
	
	write_imagef(partialSums, columnPosition, (float4)(partialSum, partialSum, partialSum, partialSum));
}

void kernel layerNodeBackpropagateLast(read_only image3d_t weights, read_only image3d_t lastLayerNodeOutputs, read_only image3d_t lastLayerCellStates, write_only image3d_t lastLayerNodeErrors, int cellsPerColumn, float qError) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	for (int ci = 0; ci < cellsPerColumn; ci++) {
		float weight = read_imagef(weights, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		//float cellState = read_imagef(lastLayerCellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		float deriv = read_imagef(lastLayerNodeOutputs, (int4)(columnPosition.x, columnPosition.y, ci, 0)).y;
		float error = qError * weight * deriv;
		
		write_imagef(lastLayerNodeErrors, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(error, error, error, error));
	}
}

void kernel layerNodeBackpropagate(read_only image3d_t nextLayerNodeErrors, read_only image3d_t nextLayerNodeWeights, read_only image3d_t nodeOutputs, read_only image3d_t cellStates, write_only image3d_t nodeErrors,
	int cellsPerColumn, float2 layerSizeMinusOneInv, int2 nextLayerSize, int2 nextLayerSizeMinusReverseReceptiveRadius, int nextLayerCellsPerColumn, int2 reverseNodeFieldSize, int2 nextLayerReceptiveSize, float2 nextOverReverseNodeFieldSize)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 centerPositionNormalized = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	float2 nextCenterPosition = (float2)(centerPositionNormalized.x * nextLayerSizeMinusReverseReceptiveRadius.x, centerPositionNormalized.y * nextLayerSizeMinusReverseReceptiveRadius.y);
	
	for (int ci = 0; ci < cellsPerColumn; ci++) {
		float sum = 0.0f;
		
		for (int dx = -reverseNodeFieldSize.x; dx <= reverseNodeFieldSize.x; dx++)
		for (int dy = -reverseNodeFieldSize.y; dy <= reverseNodeFieldSize.y; dy++) {
			int2 nextColumnPosition = (int2)(nextCenterPosition.x + dx, nextCenterPosition.y + dy);
			
			if (nextColumnPosition.x >= 0 && nextColumnPosition.x < nextLayerSize.x && nextColumnPosition.y >= 0 && nextColumnPosition.y < nextLayerSize.y) {
				for (int cio = 0; cio < nextLayerCellsPerColumn; cio++) {
					float nodeError = read_imagef(nextLayerNodeErrors, (int4)(nextColumnPosition.x, nextColumnPosition.y, cio, 0)).x;
				
					int weightSecondCoordinate = cio + nextColumnPosition.y * nextLayerCellsPerColumn;
					int weightThirdCoordinate = ci + round((reverseNodeFieldSize.y - dy) * nextOverReverseNodeFieldSize.y) * cellsPerColumn + round((reverseNodeFieldSize.x - dx) * nextOverReverseNodeFieldSize.x) * cellsPerColumn * (nextLayerReceptiveSize.y * 2 + 1);
				
					int4 weightCoord = (int4)(nextColumnPosition.x, weightSecondCoordinate, weightThirdCoordinate, 0);
					
					float weight = read_imagef(nextLayerNodeWeights, weightCoord).x;
					
					sum += nodeError * weight;
				}
			}
		}
	
		//float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		float deriv = read_imagef(nodeOutputs, (int4)(columnPosition.x, columnPosition.y, ci, 0)).y;
		float error = sum * deriv;
		
		write_imagef(nodeErrors, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(error, error, error, error));
	}
}

void kernel layerNodeBackpropagateToInput(read_only image3d_t nextLayerNodeErrors, read_only image3d_t nextLayerNodeWeights, read_only image2d_t inputs, write_only image2d_t inputErrors,
	float2 layerSizeMinusOneInv, int2 nextLayerSize, int2 nextLayerSizeMinusReverseReceptiveRadius, int nextLayerCellsPerColumn, int2 reverseNodeFieldSize, int2 nextLayerReceptiveSize, float2 nextOverReverseNodeFieldSize)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 centerPositionNormalized = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	float2 nextCenterPosition = (float2)(centerPositionNormalized.x * nextLayerSizeMinusReverseReceptiveRadius.x, centerPositionNormalized.y * nextLayerSizeMinusReverseReceptiveRadius.y);
	
	float sum = 0.0f;
	
	for (int dx = -reverseNodeFieldSize.x; dx <= reverseNodeFieldSize.x; dx++)
	for (int dy = -reverseNodeFieldSize.y; dy <= reverseNodeFieldSize.y; dy++) {
		int2 nextColumnPosition = (int2)(nextCenterPosition.x + dx, nextCenterPosition.y + dy);
		
		if (nextColumnPosition.x >= 0 && nextColumnPosition.x < nextLayerSize.x && nextColumnPosition.y >= 0 && nextColumnPosition.y < nextLayerSize.y) {
			for (int cio = 0; cio < nextLayerCellsPerColumn; cio++) {
				float nodeError = read_imagef(nextLayerNodeErrors, (int4)(nextColumnPosition.x, nextColumnPosition.y, cio, 0)).x;
			
				int weightSecondCoordinate = cio + nextColumnPosition.y * nextLayerCellsPerColumn;
				int weightThirdCoordinate = round((reverseNodeFieldSize.y - dy) * nextOverReverseNodeFieldSize.y) + round((reverseNodeFieldSize.x - dx) * nextOverReverseNodeFieldSize.x) * (nextLayerReceptiveSize.y * 2 + 1);
				
				int4 weightCoord = (int4)(nextColumnPosition.x, weightSecondCoordinate, weightThirdCoordinate, 0);
				
				float weight = read_imagef(nextLayerNodeWeights, weightCoord).x;
				
				sum += nodeError * weight;
			}
		}
	}

	//float input = read_imagef(inputs, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
	//float sigmoid = read_imagef(nodeOutputs, (int4)(columnPosition.x, columnPosition.y, ci, 0)).y;
	float error = sum;
	
	write_imagef(inputErrors, columnPosition, (float4)(error, error, error, error));
}

void kernel layerNodeWeightUpdate(read_only image3d_t layerNodeErrors, read_only image3d_t statesInput, read_only image3d_t nodeBiasesPrev, read_only image3d_t nodeWeightsPrev, write_only image3d_t nodeBiases, write_only image3d_t nodeWeights,
	int cellsInColumn, int inputCellsPerColumn, int2 nodeFieldRadius, float2 layerSizeMinusOneInv, int2 inputSize, int2 inputSizeMinusOne, float alpha, float beta, float temperature, float eligibilityDecay)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	float2 inputCenterPosition = (float2)(inputCenterPositionNormalized.x * inputSizeMinusOne.x, inputCenterPositionNormalized.y * inputSizeMinusOne.y);
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		float error = read_imagef(layerNodeErrors, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;

		// Go through all connections 
		int wi = 0;
		
		for (int dx = -nodeFieldRadius.x; dx <= nodeFieldRadius.x; dx++)
		for (int dy = -nodeFieldRadius.y; dy <= nodeFieldRadius.y; dy++) {
			int2 connectionCoords = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
			
			if (connectionCoords.x >= 0 && connectionCoords.x < inputSize.x && connectionCoords.y >= 0 && connectionCoords.y < inputSize.y) {	
				for (int cio = 0; cio < inputCellsPerColumn; cio++) {
					int weightCoord = cio + (nodeFieldRadius.y + dy) * inputCellsPerColumn + (nodeFieldRadius.x + dx) * inputCellsPerColumn * (nodeFieldRadius.y * 2 + 1);
					int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, weightCoord, 0);
				
					float2 nodeWeightPrev = read_imagef(nodeWeightsPrev, weightPosition).xy;
			
					float connectionState = read_imagef(statesInput, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
					
					float eligibility = error * connectionState;
				
					float2 newNodeWeight = (float2)(nodeWeightPrev.x + alpha * nodeWeightPrev.y, (1.0f - eligibilityDecay) * nodeWeightPrev.y + beta * exp(-temperature * fabs(eligibility)) * eligibility);
				
					write_imagef(nodeWeights, weightPosition, (float4)(newNodeWeight.x, newNodeWeight.y, 0.0f, 0.0f));
					
					wi++;
				}
			}
			else
				wi += inputCellsPerColumn;
		}
		
		float2 biasPrev = read_imagef(nodeBiasesPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).xy;
		
		float2 newBias = (float2)(biasPrev.x + alpha * biasPrev.y, (1.0f - eligibilityDecay) * biasPrev.y + beta * exp(-temperature * fabs(error)) * error);
		
		write_imagef(nodeBiases, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(newBias.x, newBias.y, 0.0f, 0.0f));
	}
}

void kernel layerNodeWeightUpdateFirst(read_only image3d_t layerNodeErrors, read_only image2d_t statesInput, read_only image3d_t nodeBiasesPrev, read_only image3d_t nodeWeightsPrev, write_only image3d_t nodeBiases, write_only image3d_t nodeWeights,
	int cellsInColumn, int2 nodeFieldRadius, float2 layerSizeMinusOneInv, int2 inputSize, int2 inputSizeMinusOne, float alpha, float beta, float temperature, float eligibilityDecay)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	float2 inputCenterPosition = (float2)(inputCenterPositionNormalized.x * inputSizeMinusOne.x, inputCenterPositionNormalized.y * inputSizeMinusOne.y);
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		float error = read_imagef(layerNodeErrors, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;

		// Go through all connections 
		int wi = 0;
		
		for (int dx = -nodeFieldRadius.x; dx <= nodeFieldRadius.x; dx++)
		for (int dy = -nodeFieldRadius.y; dy <= nodeFieldRadius.y; dy++) {
			int2 connectionCoords = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
			
			if (connectionCoords.x >= 0 && connectionCoords.x < inputSize.x && connectionCoords.y >= 0 && connectionCoords.y < inputSize.y) {	
				int weightCoord = (nodeFieldRadius.y + dy) + (nodeFieldRadius.x + dx) * (nodeFieldRadius.y * 2 + 1);
				int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, weightCoord, 0);
			
				float2 nodeWeightPrev = read_imagef(nodeWeightsPrev, weightPosition).xy;
		
				float connectionState = read_imagef(statesInput, connectionCoords).x;
				
				float eligibility = error * connectionState;
				
				float2 newNodeWeight = (float2)(nodeWeightPrev.x + alpha * nodeWeightPrev.y, (1.0f - eligibilityDecay) * nodeWeightPrev.y + beta * exp(-temperature * fabs(eligibility)) * eligibility);
				
				write_imagef(nodeWeights, weightPosition, (float4)(newNodeWeight.x, newNodeWeight.y, 0.0f, 0.0f));
			}
			
			wi++;
		}
		
		float2 biasPrev = read_imagef(nodeBiasesPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).xy;
		
		float2 newBias = (float2)(biasPrev.x + alpha * biasPrev.y, (1.0f - eligibilityDecay) * biasPrev.y + beta * exp(-temperature * fabs(error)) * error);
		
		write_imagef(nodeBiases, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(newBias.x, newBias.y, 0.0f, 0.0f));
	}
}

void kernel layerNodeWeightUpdateLast(read_only image3d_t statesInput, read_only image3d_t weightsPrev, write_only image3d_t weights, int inputCellsPerColumn, float alpha, float beta, float temperature, float eligibilityDecay) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	for (int ci = 0; ci < inputCellsPerColumn; ci++) {
		float nodeState = read_imagef(statesInput, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float2 weightPrev = read_imagef(weightsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).xy;
		
		float2 newWeight = (float2)(weightPrev.x + alpha * weightPrev.y, (1.0f - eligibilityDecay) * weightPrev.y + beta * exp(-temperature * fabs(nodeState)) * nodeState);
		
		write_imagef(weights, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(newWeight.x, newWeight.y, 0.0f, 0.0f));
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

void kernel reconstructInput(read_only image3d_t reconstructionWeights, read_only image2d_t sdr, write_only image2d_t inputs,
	int2 receptiveFieldRadius, float2 inputSizeMinusOneInv, int2 layerSize, int2 layerSizeMinusOne)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * inputSizeMinusOneInv.x, columnPosition.y * inputSizeMinusOneInv.y);
	float2 inputCenterPosition = (float2)(inputCenterPositionNormalized.x * layerSizeMinusOne.x, inputCenterPositionNormalized.y * layerSizeMinusOne.y);

	float sum = 0.0f;

	int wi = 0;

	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
		
		if (inputPosition.x >= 0 && inputPosition.x < layerSize.x && inputPosition.y >= 0 && inputPosition.y < layerSize.y) {
			float source = read_imagef(sdr, inputPosition).x;

			float weight = read_imagef(reconstructionWeights, (int4)(columnPosition.x, columnPosition.y, wi, 0)).x;
			
			sum += source * weight;
		}
		
		wi++;
	}

	// Bias
	float bias = read_imagef(reconstructionWeights, (int4)(columnPosition.x, columnPosition.y, wi, 0)).x;
		
	sum += bias;

	float output = sigmoid(sum);
	
	write_imagef(inputs, columnPosition, (float4)(output, output, output, output));
}

void kernel learnReconstruction(read_only image2d_t targets, read_only image2d_t inputs, read_only image3d_t reconstructionWeightsPrev, read_only image2d_t sdr, write_only image3d_t reconstructionWeights,
	int2 receptiveFieldRadius, float2 inputSizeMinusOneInv, int2 layerSize, int2 layerSizeMinusOne, float alpha)
{
 	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * inputSizeMinusOneInv.x, columnPosition.y * inputSizeMinusOneInv.y);
	float2 inputCenterPosition = (float2)(inputCenterPositionNormalized.x * layerSizeMinusOne.x, inputCenterPositionNormalized.y * layerSizeMinusOne.y);
 		
 	float target = read_imagef(targets, columnPosition).x;
 	float input = read_imagef(inputs, columnPosition).x;
 	
 	float error = alpha * (target - input);
 	
 	int wi = 0;
 	
 	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
 	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
 		int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
 		
 		if (inputPosition.x >= 0 && inputPosition.x < layerSize.x && inputPosition.y >= 0 && inputPosition.y < layerSize.y) {
 			float source = read_imagef(sdr, inputPosition).x;
 
 			float prevWeight = read_imagef(reconstructionWeightsPrev, (int4)(columnPosition.x, columnPosition.y, wi, 0)).x;
 			
 			float newWeight = prevWeight + error * source;
 			
 			write_imagef(reconstructionWeights, (int4)(columnPosition.x, columnPosition.y, wi, 0), (float4)(newWeight, newWeight, newWeight, newWeight));
 		}
		
		wi++;
 	}
 	
 	// Bias
	float prevBias = read_imagef(reconstructionWeightsPrev, (int4)(columnPosition.x, columnPosition.y, wi, 0)).x;
 		
 	float newBias = prevBias + error;
 	
 	write_imagef(reconstructionWeights, (int4)(columnPosition.x, columnPosition.y, wi, 0), (float4)(newBias, newBias, newBias, newBias));
 }