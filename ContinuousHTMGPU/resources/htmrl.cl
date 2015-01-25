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
constant float activationModulationPower = 1.5f;
constant float qModulationPower = 1.5f;
constant float crowdingIntensity = 4.0f;
constant float cellStateIntensity = 64.0f;
constant float cellPredictionIntensity = 4.0f;
constant float minLearningThreshold = 0.0f;
constant float predictionRangeExtension = 0.1f;
constant float localActivity = 2.0f;
constant float crowdingActivity = 3.0f;
constant float uniquenessPower = 4.0f;
constant float minOverlapForActivation = 0.0f;
constant float subOverlapIncrement = 0.0005f;
constant float boostDutyCycleRatio = 0.01f;
constant float maxDutyCycleForLearnRatio = 0.3f;
constant float maxBoost = 1.0f;
constant float rectifierLeak = 0.03f;
constant float cellNoise = 0.01f;
constant float contributionSensitivity = 100.0f;
constant float minDivisor = 0.001f;

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

float rectifier(float x) {
	return fmax(0.0f, x);
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

void kernel initializePartOne(write_only image2d_t columnActivations, write_only image2d_t columnStates, write_only image3d_t columnWeights, write_only image2d_t columnDutyCycles, write_only image2d_t columnPrevValues,
	int cellsInColumn, int receptiveFieldSize, int lateralConnectionsSize, uint2 seed, float minWeight, float maxWeight, float minWidth, float maxWidth)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 - 12, get_global_id(1) * 16 + 23) * 36;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));

	write_imagef(columnActivations, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	write_imagef(columnStates, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	write_imagef(columnDutyCycles, columnPosition, (float4)(1.0f, 0.0f, 0.0f, 0.0f));
	write_imagef(columnPrevValues, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));

	for (int wi = 0; wi < receptiveFieldSize; wi++) {
		int4 weightPosition = (int4)(columnPosition.x, columnPosition.y, wi, 0);
	
		float columnConnectionWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;

		write_imagef(columnWeights, weightPosition, (float4)(columnConnectionWeight, 0.0f, 0.0f, 0.0f));
	}
	
	int4 widthPosition = (int4)(columnPosition.x, columnPosition.y, receptiveFieldSize, 0);
	
	float columnWidth = randFloat(&seedValue) * (maxWidth - minWidth) + minWidth;
	
	write_imagef(columnWeights, widthPosition, (float4)(columnWidth, 0.0f, 0.0f, 0.0f));
}

void kernel initializePartTwo(write_only image3d_t cellStates, write_only image3d_t cellWeights, write_only image3d_t cellPredictions, write_only image3d_t cellQValues,
	int cellsInColumn, int receptiveFieldSize, int lateralConnectionsSize, uint2 seed, float minWeight, float maxWeight)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 32 + 24, get_global_id(1) * 11 - 66) * 23;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		write_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
		write_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
		write_imagef(cellQValues, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
		
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
	
		for (int wi = 0; wi < lateralConnectionsSize; wi++) {
			int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
			float cellWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;
	
			write_imagef(cellWeights, weightPosition, (float4)(cellWeight, 0.0f, 0.0f, 0.0f));
		}
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
				
			float modulation = pow(1.0f - (float)(abs(dx) + abs(dy)) / (float)(inputReceptiveFieldRadius.x + inputReceptiveFieldRadius.y), activationModulationPower);
				
			float delta = inputState - weight;
				
			sum += modulation * delta * delta;
			
			usageCount++;
		}
		
		weightIndex++;
	}
	
	//float bias = read_imagef(columnWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
	
	//sum += bias;
	
	//weightIndex++;
	//usageCount++;
	
	float output = -sum * (float)(weightIndex) / (float)(usageCount);// * (1.0f - boost);

	write_imagef(columnActivations, columnPosition, (float4)(output, 0.0f, 0.0f, 0.0f));
}

void kernel layerColumnInhibit(read_only image2d_t columnActivations, read_only image2d_t columnDutyCyclesPrev, write_only image2d_t columnStates, int2 layerSize, float2 layerSizeInv, int2 receptiveFieldRadius, float noMatchIntensity) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float thisActivation = read_imagef(columnActivations, columnPosition).x;

	float higherSum = 0.0f;
	
	float highest = thisActivation;
	
	float minDutyCycle = 1.0f;

	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		int2 layerPosition = (int2)(columnPosition.x + dx, columnPosition.y + dy);
	
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			float activation = read_imagef(columnActivations, layerPosition).x;
			
			float dutyCycle = read_imagef(columnDutyCyclesPrev, layerPosition).x;
	
			if (activation > thisActivation) {
				higherSum++;
			}
			
			highest = fmax(highest, activation);
			
			if (dx != 0 && dy != 0)
				minDutyCycle = fmin(minDutyCycle, dutyCycle);
		}
	}
	
	float thisDutyCycle = read_imagef(columnDutyCyclesPrev, columnPosition).x;
	
	float noMatch = highest < -noMatchIntensity ? 1.0f : 0.0f;
	
	float learnFactor = noMatch * (thisDutyCycle <= fmin(maxDutyCycleForLearnRatio, minDutyCycle) ? 1.0f : 0.0f);
	
	float inhibitedResult = fmax(0.0f, sigmoid((localActivity - higherSum) * columnIntensity) * 2.0f - 1.0f);//exp((thisActivation - maxActivation) * columnIntensity);//exp(-higherSum * columnIntensity);// //
	//higherSum < localActivity ? 1.0f : 0.0f;//
	write_imagef(columnStates, columnPosition, (float4)(inhibitedResult, learnFactor, 0.0f, 0.0f));
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

void kernel layerColumnWeightUpdate(read_only image2d_t columnStatesInput, read_only image2d_t columnActivations, read_only image2d_t columnStates, read_only image2d_t columnStatesPrev, read_only image3d_t columnWeightsPrev, read_only image2d_t columnDutyCyclesPrev, write_only image3d_t columnWeights,
	float2 layerSizeMinusOneInv, int2 inputReceptiveFieldRadius, int2 inputSize, int2 inputSizeMinusOne, float connectionAlpha, float widthAlpha, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 130;
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	float2 inputCenterPosition = (float2)(inputCenterPositionNormalized.x * inputSizeMinusOne.x, inputCenterPositionNormalized.y * inputSizeMinusOne.y);
	
	float2 thisState = read_imagef(columnStates, columnPosition).xy;
	
	float2 thisStatePrev = read_imagef(columnStatesPrev, columnPosition).xy;
	
	float2 thisActivation = read_imagef(columnActivations, columnPosition).xy;
	
	float2 dutyCyclePrev = read_imagef(columnDutyCyclesPrev, columnPosition).xy;
	
	//float globalIncrement = fmax(0.0f, minOverlapForActivation - dutyCyclePrev.x) * subOverlapIncrement;
	
	//float boost = boostFunction(dutyCyclePrev.x, boostDutyCycleRatio);
	
	float boost = thisState.y * dutyCyclePrev.y;
	
	float learnScalar = (1.0f - boost) * fmax(0.0f, thisState.x - thisStatePrev.x) + boost;// * ((1.0f - boost) * thisState.x + boost - thisActivation.x);
	
	// Adjust weights by their source activations and error
	int weightIndex = 0;
	
	for (int dx = -inputReceptiveFieldRadius.x; dx <= inputReceptiveFieldRadius.x; dx++)
	for (int dy = -inputReceptiveFieldRadius.y; dy <= inputReceptiveFieldRadius.y; dy++) {
		int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
		
		if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
			float inputState = read_imagef(columnStatesInput, inputPosition).x;
				
			float prevWeight = read_imagef(columnWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
				
			//float modulation = (float)(abs(dx) + abs(dy)) / (inputReceptiveFieldRadius.x + inputReceptiveFieldRadius.y);
				
			float newWeight = prevWeight + connectionAlpha * learnScalar * (inputState - prevWeight);
				
			write_imagef(columnWeights, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0), (float4)(newWeight, newWeight, newWeight, newWeight));
		}
		
		weightIndex++;
	}

	//float prevBias = read_imagef(columnWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
	
	//float newBias = prevBias + connectionAlpha * error;
	
	//write_imagef(columnWeights, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0), (float4)(newBias, newBias, newBias, newBias));
}

void kernel layerCellActivate(read_only image2d_t columnStates, read_only image3d_t cellStatesPrev, read_only image3d_t cellPredictionsPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev,
	write_only image3d_t cellStates, int cellsInColumn, int2 lateralConnectionsRadii, float cellTraceDecay, uint2 seed)
{
	/*uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 84;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	
	float highestPrediction = 0.0f;
	
	int highestPredictionCell = 0;

	for (int ci = 0; ci < cellsInColumn; ci++) {
		float prediction = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		if (prediction > highestPrediction) {
			highestPrediction = prediction;
			
			highestPredictionCell = ci;
		}
	}
	
	float allPossibilitiesIncrease = 1.0f - highestPrediction;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float newCellState = (highestPrediction * (ci == highestPredictionCell ? 1.0f : 0.0f) + allPossibilitiesIncrease) * columnState;
	
		float prevTrace = read_imagef(cellStatesPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).y;
	
		float newTrace = fmax((1.0f - cellTraceDecay) * prevTrace, newCellState);
	
		write_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(newCellState, newTrace, 0.0f, 0.0f));
	}*/
	
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	
	float maxCellPrediction = 0.0f;
	int maxCellPredictionIndex = 0;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float prediction = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		if (prediction > maxCellPrediction) {
			maxCellPredictionIndex = ci;
			
			maxCellPrediction = prediction;
		}
	}
	
	float allCellsIncrease = 1.0f - maxCellPrediction;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float newCellState = ((1.0f - allCellsIncrease) * (ci == maxCellPredictionIndex ? 1.0f : 0.0f) + allCellsIncrease) * columnState;
	
		float prevTrace = read_imagef(cellStatesPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).y;
	
		float newTrace = fmax((1.0f - cellTraceDecay) * prevTrace, newCellState);
	
		write_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(newCellState, newTrace, 0.0f, 0.0f));
	}
}

void kernel layerCellWeightUpdate(read_only image2d_t columnStatesPrev, read_only image2d_t columnStates, read_only image3d_t cellPredictionsPrev, read_only image3d_t cellStates, read_only image3d_t cellStatesPrev, read_only image2d_t nextLayerContextPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev, read_only image2d_t columnTdErrors,
	write_only image3d_t cellWeights, int cellsInColumn, int2 layerSize, int2 lateralConnectionsRadii, float2 layerSizeMinusOneInv, int2 nextLayerSize, int2 nextLayerSizeMinusOne, float alpha, float beta, float temperature, float eligibilityDecay)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 normalizedColumnCoords = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	int2 connectionCoordsNextCenter = (int2)(normalizedColumnCoords.x * nextLayerSizeMinusOne.x, normalizedColumnCoords.y * nextLayerSizeMinusOne.y);
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	float columnPredictionPrev = read_imagef(columnPredictionsPrev, columnPosition).x;
	
	float tdError = read_imagef(columnTdErrors, columnPosition).x;
	
	float learn = tdError > 0.0f ? 1.0f : 0.0f;
	
	//float predictionError = columnState - columnPredictionPrev;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		float2 cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).xy;
		float cellPredictionPrev = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;

		float cellError = cellState.x - cellPredictionPrev;
		
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
					
					float newTrace = (1.0f - eligibilityDecay) * cellWeightPrev.y + beta * (sign(cellWeightPrev.y) == sign(eligibility) ? exp(-temperature * fabs(cellWeightPrev.y)) : 1.0f) * eligibility;
					
					float2 newCellWeight = (float2)(cellWeightPrev.x + alpha * learn * newTrace, newTrace);
					
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
					
					float newTrace = (1.0f - eligibilityDecay) * cellWeightPrev.y + beta * (sign(cellWeightPrev.y) == sign(eligibility) ? exp(-temperature * fabs(cellWeightPrev.y)) : 1.0f) * eligibility;
					
					float2 newCellWeight = (float2)(cellWeightPrev.x + alpha * learn * newTrace, newTrace);
					
					write_imagef(cellWeights, weightPosition, (float4)(newCellWeight.x, newCellWeight.y, 0.0f, 0.0f));
				}
				
				wi++;
			}
			else
				wi += cellsInColumn + 1;
		}
		
		/*int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float2 cellBiasPrev = read_imagef(cellWeightsPrev, biasPosition).xy;

		float eligibility = cellError;
		
		float decayedTrace = (1.0f - eligibilityDecay) * cellBiasPrev.y;
		
		float newTrace = decayedTrace + beta * exp(-temperature * fabs(decayedTrace)) * eligibility;
		
		float2 newCellBias = (float2)(cellBiasPrev.x + alpha * newTrace, newTrace);

		write_imagef(cellWeights, biasPosition, (float4)(newCellBias.x, newCellBias.y, 0.0f, 0.0f));*/
	}
}

void kernel layerCellWeightUpdateLast(read_only image2d_t columnStatesPrev, read_only image2d_t columnStates, read_only image3d_t cellPredictionsPrev, read_only image3d_t cellStates, read_only image3d_t cellStatesPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev, read_only image2d_t columnTdErrors,
	write_only image3d_t cellWeights, int cellsInColumn, int2 layerSize, int2 lateralConnectionsRadii, float alpha, float beta, float temperature, float eligibilityDecay)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	float columnPredictionPrev = read_imagef(columnPredictionsPrev, columnPosition).x;
	
	float tdError = read_imagef(columnTdErrors, columnPosition).x;
	
	float learn = tdError > 0.0f ? 1.0f : 0.0f;
	
	//float predictionError = columnState - columnPredictionPrev;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		float2 cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).xy;
		float cellPredictionPrev = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float cellError = cellState.x - cellPredictionPrev;
		
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
					
					float newTrace = (1.0f - eligibilityDecay) * cellWeightPrev.y + beta * (sign(cellWeightPrev.y) == sign(eligibility) ? exp(-temperature * fabs(cellWeightPrev.y)) : 1.0f) * eligibility;
					
					float2 newCellWeight = (float2)(cellWeightPrev.x + alpha * learn * newTrace, newTrace);
					
					write_imagef(cellWeights, weightPosition, (float4)(newCellWeight.x, newCellWeight.y, 0.0f, 0.0f));
					
					wi++;
				}
			}
			else
				wi += cellsInColumn;
		}
		
		/*int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float2 cellBiasPrev = read_imagef(cellWeightsPrev, biasPosition).xy;

		float eligibility = cellError;
		
		float decayedTrace = (1.0f - eligibilityDecay) * cellBiasPrev.y;
		
		float newTrace = decayedTrace + beta * exp(-temperature * fabs(decayedTrace)) * eligibility;
		
		float2 newCellBias = (float2)(cellBiasPrev.x + alpha * newTrace, newTrace);
		
		write_imagef(cellWeights, biasPosition, (float4)(newCellBias.x, newCellBias.y, 0.0f, 0.0f));*/
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
		
		/*int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float bias = read_imagef(cellWeights, biasPosition).x;
		
		sum += bias;*/
		
		float prediction = fmax(0.0f, sigmoid(sum * cellPredictionIntensity) * 2.0f - 1.0f);
		
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
		
		/*int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float bias = read_imagef(cellWeights, biasPosition).x;
		
		sum += bias;*/
		
		float prediction = fmax(0.0f, sigmoid(sum * cellPredictionIntensity) * 2.0f - 1.0f);
		
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

void kernel layerTdError(read_only image3d_t cellStatesPrev, read_only image2d_t columnStatesPrev, read_only image2d_t columnStates, read_only image2d_t columnDutyCyclesPrev, read_only image2d_t columnQValues, read_only image2d_t columnPrevValuesPrev, write_only image2d_t columnTdErrors, write_only image2d_t columnPrevValues,
	int cellsInColumn, int2 layerSize, int2 qConnectionsRadii, float reward, float alpha, float gamma)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float qSum = 0.0f;
	float divisor = 0.0f;
	
	// Go through all connections 
	for (int dx = -qConnectionsRadii.x; dx <= qConnectionsRadii.x; dx++)
	for (int dy = -qConnectionsRadii.y; dy <= qConnectionsRadii.y; dy++) {
		int2 connectionCoords = (int2)(columnPosition.x + dx, columnPosition.y + dy);
			
		if (connectionCoords.x >= 0 && connectionCoords.x < layerSize.x && connectionCoords.y >= 0 && connectionCoords.y < layerSize.y) {	
			float q = read_imagef(columnQValues, connectionCoords).x;
		
			float modulation = pow(1.0f - (float)(abs(dx) + abs(dy)) / (float)(qConnectionsRadii.x + qConnectionsRadii.y), qModulationPower);
			
			// Ignore columns that will get reassigned
			float2 columnState = read_imagef(columnStates, connectionCoords).xy;
		
			float contribution = modulation * (1.0f - columnState.y) * columnState.x;
			
			qSum += contribution * q;
			divisor += contribution;
		}
	}
		
	float tdError;
	float keepQ;

	//float columnLearn = read_imagef(columnStatesPrev, columnPosition).y;
	
	if (divisor < minDivisor) {
		tdError = 0.0f;
		keepQ = 0.0f;
	}
	else {
		keepQ = qSum / divisor;
		
		float prevValue = read_imagef(columnPrevValuesPrev, columnPosition).x;
	
		float2 columnState = read_imagef(columnStates, columnPosition).xy;
			
		float prevModulatedValue = (1.0f - columnState.y) * prevValue + columnState.y * keepQ;
		
		tdError = columnState.x * (alpha * (reward + gamma * keepQ - prevModulatedValue));
	}
		
	write_imagef(columnTdErrors, columnPosition, (float4)(tdError, 0.0f, 0.0f, 0.0f));
	write_imagef(columnPrevValues, columnPosition, (float4)(keepQ, 0.0f, 0.0f, 0.0f));
}

void kernel layerAssignQ(read_only image2d_t blurredColumnTdErrors, read_only image3d_t cellQValuesPrev, read_only image3d_t cellStatesPrev, write_only image3d_t cellQValues,
	int cellsInColumn)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float tdError = read_imagef(blurredColumnTdErrors, columnPosition).x;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float qPrev = read_imagef(cellQValuesPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float cellEligibility = read_imagef(cellStatesPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).y;
		
		float storeQ = qPrev + cellEligibility * tdError;
		
		write_imagef(cellQValues, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(storeQ, 0.0f, 0.0f, 0.0f));
	}
}

void kernel layerColumnQ(read_only image3d_t cellQValuesPrev, read_only image3d_t cellStatesPrev, read_only image3d_t cellStates, write_only image2d_t columnQValues, int cellsInColumn) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float sum = 0.0f;
	float divisor = 0.0f;
	float unweightedAverage = 0.0f;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float state = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		float cellQ = read_imagef(cellQValuesPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		sum += state * cellQ;
		divisor += state;
		
		unweightedAverage += cellQ;
	}
	
	unweightedAverage /= cellsInColumn;
	
	float output;
	
	if (divisor < minDivisor)
		output = unweightedAverage;
	else
		output = sum / divisor;
		
	write_imagef(columnQValues, columnPosition, (float4)(output, 0.0f, 0.0f, 0.0f));
}

void kernel reconstructInput(read_only image3d_t sdrCenters, read_only image2d_t sdr, write_only image2d_t inputs,
	int2 reverseReceptiveFieldRadius, int2 sdrReceptiveFieldRadius, int2 inputSizeMinusOne, float2 inputSizeMinusOneInv, int2 sdrSize, int2 sdrSizeMinusOne, float2 sdrSizeMinusOneInv)
{
	int2 inputPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputPositionNormalized = (float2)(inputPosition.x * inputSizeMinusOneInv.x, inputPosition.y * inputSizeMinusOneInv.y);
	float2 sdrPositionCenter = (float2)(inputPositionNormalized.x * sdrSizeMinusOne.x, inputPositionNormalized.y * sdrSizeMinusOne.y);
	
	float sum = 0.0f;
	float divisor = 0.0f;

	for (int dx = -reverseReceptiveFieldRadius.x; dx <= reverseReceptiveFieldRadius.x; dx++)
	for (int dy = -reverseReceptiveFieldRadius.y; dy <= reverseReceptiveFieldRadius.y; dy++) {
		int2 sdrPosition = (int2)(sdrPositionCenter.x + dx, sdrPositionCenter.y + dy);
		
		if (sdrPosition.x >= 0 && sdrPosition.x < sdrSize.x && sdrPosition.y >= 0 && sdrPosition.y < sdrSize.y) {
			// Next layer node's receptive field
			int2 fieldCenter = (int2)(sdrPosition.x * sdrSizeMinusOneInv.x * inputSizeMinusOne.x, sdrPosition.y * sdrSizeMinusOneInv.y * inputSizeMinusOne.y);

			int2 fieldLowerBounds = fieldCenter - sdrReceptiveFieldRadius;
			int2 fieldUpperBounds = fieldCenter + sdrReceptiveFieldRadius;
		
			// Check for containment
			if (inputPosition.x >= fieldLowerBounds.x && inputPosition.x <= fieldUpperBounds.x && inputPosition.y >= fieldLowerBounds.y && inputPosition.y <= fieldUpperBounds.y) {	
				int rdx = inputPosition.x - fieldCenter.x;
				int rdy = inputPosition.y - fieldCenter.y;
				
				float source = read_imagef(sdr, sdrPosition).x;
				
				int weightIndex = (sdrReceptiveFieldRadius.y + rdy) + (sdrReceptiveFieldRadius.x + rdx) * (sdrReceptiveFieldRadius.y * 2 + 1);

				float weight = read_imagef(sdrCenters, (int4)(sdrPosition.x, sdrPosition.y, weightIndex, 0)).x;
				
				float modulation = pow((float)(abs(rdx) + abs(rdy)) / (float)(sdrReceptiveFieldRadius.x + sdrReceptiveFieldRadius.y), activationModulationPower);
				
				sum += modulation * source * weight;
				divisor += modulation * source;
			}
		}
	}
	
	float recon;
	
	if (divisor == 0.0f)
		recon = 0.5f;
	else
		recon = sum / divisor;
	
	write_imagef(inputs, inputPosition, (float4)(recon, recon, recon, recon));
}

void kernel gaussianBlurX(read_only image2d_t source, write_only image2d_t destination, float2 sizeInv, float kernelWidth) {
	int2 destinationPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 destinationPositionNormalized = (float2)(destinationPosition.x * sizeInv.x, destinationPosition.y * sizeInv.y);
	
	float sum = 0.0f;
	
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x - 4.0f * kernelWidth, destinationPositionNormalized.y)).x * 0.05f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x - 3.0f * kernelWidth, destinationPositionNormalized.y)).x * 0.09f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x - 2.0f * kernelWidth, destinationPositionNormalized.y)).x * 0.12f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x - kernelWidth, destinationPositionNormalized.y)).x * 0.15f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y)).x * 0.16f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x + kernelWidth, destinationPositionNormalized.y)).x * 0.15f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x + 2.0f * kernelWidth, destinationPositionNormalized.y)).x * 0.12f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x + 3.0f * kernelWidth, destinationPositionNormalized.y)).x * 0.09f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x + 4.0f * kernelWidth, destinationPositionNormalized.y)).x * 0.05f;
 
	write_imagef(destination, destinationPosition, (float4)(sum, sum, sum, sum));
}

void kernel gaussianBlurY(read_only image2d_t source, write_only image2d_t destination, float2 sizeInv, float kernelWidth) {
	int2 destinationPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 destinationPositionNormalized = (float2)(destinationPosition.x * sizeInv.x, destinationPosition.y * sizeInv.y);
	
	float sum = 0.0f;
	
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y - 4.0f * kernelWidth)).x * 0.05f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y - 3.0f * kernelWidth)).x * 0.09f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y - 2.0f * kernelWidth)).x * 0.12f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y - kernelWidth)).x * 0.15f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y)).x * 0.16f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y + kernelWidth)).x * 0.15f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y + 2.0f * kernelWidth)).x * 0.12f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y + 3.0f * kernelWidth)).x * 0.09f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y + 4.0f * kernelWidth)).x * 0.05f;
 
	write_imagef(destination, destinationPosition, (float4)(sum, sum, sum, sum));
}