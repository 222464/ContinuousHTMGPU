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
	
#define MAX_RECEPTIVE_SIZE 81
#define MAX_SEGMENTS_PER_CELL 4
	
constant float columnIntensity = 1.0f;
constant float learnTolerance = 0.01f;
constant float sparsityMultiplier = 10.0f;
constant float sparsityThreshold = 0.04f;
constant float sparsity = 0.06f;
constant float segmentSparsity = 0.3f;
constant float columnTraceDecay = 0.002f;
constant float columnMomentum = 0.1f;
constant float columnRandomness = 0.1f;
constant float minDerivative = 0.1f;
constant float minSimilarity = 0.0001f;
constant float minLearn = 0.0f;
constant float learnFalloff = 0.1f;
constant float noMatchTolerance = 0.0001f;
constant float falloffIntensity = 0.5f;
constant float activationModulationPower = 4.0f;
constant float qModulationPower = 1.0f;
constant float crowdingIntensity = 8.0f;
constant float cellStateIntensity = 32.0f;
constant float cellPredictionIntensity = 4.0f;
constant float minLearningThreshold = 0.0f;
constant float predictionRangeExtension = 0.1f;
constant float localActivity = 1.0f;
constant float reconstructionErrorActivity = 2.0f;
constant float boostThreshold = 0.01f;
constant float rectifierLeak = 0.03f;
constant float minDivisor = 0.0001f;
constant float higherLayerQPower = 16.0f;
constant float dutyCycleDecay = 0.005f;
constant float minReconstructionError = 0.1f;

// LCA
constant float lcaTauInv = 0.01f;
constant float lcaAlpha = 0.01f;
constant float lcaLambda = 0.01f;
constant float lcaGamma = 100.0f;

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

float lcaThreshold(float potential) {
	return (potential - lcaAlpha * lcaLambda) / (1.0f + exp(-lcaGamma * (potential - lcaLambda)));
}

float boostFunction(float dutyCycle, float threshold) {
	return fmin(1.0f, fmax(0.0f, threshold - dutyCycle) / threshold);
}

void kernel initializePartOne(write_only image2d_t columnActivations, write_only image2d_t columnStates, write_only image3d_t columnFeedForwardWeights, write_only image2d_t columnPrevValues,
	int cellsInColumn, int receptiveFieldSize, int lateralConnectionsSize, uint2 seed, float minWeight, float maxWeight)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 - 12, get_global_id(1) * 16 + 23) * 36;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));

	write_imagef(columnActivations, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	write_imagef(columnStates, columnPosition, (float4)(0.0f, localActivity / receptiveFieldSize, 0.0f, 0.0f));
	write_imagef(columnPrevValues, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));

	for (int wi = 0; wi < receptiveFieldSize; wi++) {
		int4 weightPosition = (int4)(columnPosition.x, columnPosition.y, wi, 0);
	
		float columnConnectionWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;

		write_imagef(columnFeedForwardWeights, weightPosition, (float4)(columnConnectionWeight, 0.0f, 0.0f, 0.0f));
	}
}

void kernel initializePartTwo(write_only image3d_t cellStates, write_only image3d_t segmentStates, write_only image3d_t cellWeights, write_only image3d_t cellPredictions, write_only image3d_t cellQValues,
	int cellsInColumn, int receptiveFieldSize, int lateralConnectionsSize, int segmentsPerCell, uint2 seed, float minWeight, float maxWeight)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 32 + 24, get_global_id(1) * 11 - 66) * 23;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		write_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
		
		for (int i = 0; i < segmentsPerCell; i++) {
			write_imagef(segmentStates, (int4)(columnPosition.x, columnPosition.y, ci * segmentsPerCell + i, 0), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
		}
		
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

void kernel layerColumnActivate(read_only image2d_t columnStatesInput, read_only image3d_t columnFeedForwardWeightsPrev, read_only image2d_t columnStatesPrev, write_only image2d_t columnActivations,
	float2 layerSizeMinusOneInv, int2 inputReceptiveFieldRadius, int2 inputSize, int2 inputSizeMinusOne, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 20;
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	float2 inputCenterPosition = (float2)(inputCenterPositionNormalized.x * inputSizeMinusOne.x, inputCenterPositionNormalized.y * inputSizeMinusOne.y);

	float sum = 0.0f;

	int weightIndex = 0;

	for (int dx = -inputReceptiveFieldRadius.x; dx <= inputReceptiveFieldRadius.x; dx++)
	for (int dy = -inputReceptiveFieldRadius.y; dy <= inputReceptiveFieldRadius.y; dy++) {
		int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
		
		if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
			float input = read_imagef(columnStatesInput, inputPosition).x;
	
			float weight = read_imagef(columnFeedForwardWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
				
			sum += weight * input;
		}
		
		weightIndex++;
	}
	
	// Bias
	float bias = read_imagef(columnFeedForwardWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
	
	sum += bias;
	
	write_imagef(columnActivations, columnPosition, (float4)(sigmoid(sum), 0.0f, 0.0f, 0.0f));
}

void kernel layerColumnInhibit(read_only image2d_t columnActivations, read_only image2d_t columnStatesPrev, read_only image3d_t columnFeedForwardWeightsPrev, write_only image2d_t columnStates,
	int2 layerSize, float2 layerSizeInv, int2 inhibitionRadii, int receptiveFieldSize)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float thisActivation = read_imagef(columnActivations, columnPosition).x;
	
	float numHigher = 0.0f;
	
	for (int dx = -inhibitionRadii.x; dx <= inhibitionRadii.x; dx++)
	for (int dy = -inhibitionRadii.y; dy <= inhibitionRadii.y; dy++) {
		int2 layerPosition = (int2)(columnPosition.x + dx, columnPosition.y + dy);
		
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			float activation = read_imagef(columnActivations, layerPosition).x;
	
			if (activation > thisActivation)
				numHigher++;
		}
	}
	
	float prevTrace = read_imagef(columnStatesPrev, columnPosition).y;
	
	float newState = numHigher < localActivity ? 1.0f : 0.0f;//exp(-numHigher * columnIntensity) * sigmoid(thisActivation); //&& thisActivation > 0.0f 
	
	float newTrace = (1.0f - columnTraceDecay) * prevTrace + columnTraceDecay * newState;
	
	write_imagef(columnStates, columnPosition, (float4)(newState, newTrace, 0.0f, 0.0f));
}

void kernel layerColumnWeightUpdate(read_only image2d_t reconstruction, read_only image2d_t inputs, read_only image2d_t columnActivations, read_only image2d_t columnStates, read_only image2d_t columnPredictions, read_only image3d_t columnFeedForwardWeightsPrev, write_only image3d_t columnFeedForwardWeights,
	int2 layerSize, float2 layerSizeMinusOneInv, int2 inputReceptiveFieldRadius, int2 inhibitionRadii, int2 inputSize, int2 inputSizeMinusOne, int receptiveFieldSize, float alpha, float beta, float gamma, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 130;
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	float2 inputCenterPosition = (float2)(inputCenterPositionNormalized.x * inputSizeMinusOne.x, inputCenterPositionNormalized.y * inputSizeMinusOne.y);

	float2 thisState = read_imagef(columnStates, columnPosition).xy;
	float thisActivation = read_imagef(columnActivations, columnPosition).x;
	
	// Inhibition
	/*float averageState = 0.0f;
	
	int count = 0;
	
	for (int dx = -inhibitionRadii.x; dx <= inhibitionRadii.x; dx++)
	for (int dy = -inhibitionRadii.y; dy <= inhibitionRadii.y; dy++) {
		int2 layerPosition = (int2)(columnPosition.x + dx, columnPosition.y + dy);
		
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			float state = read_imagef(columnStates, layerPosition).x;
	
			averageState += state;

			count++;
		}
	}
	
	averageState /= count;
	
	float sparsityPenalty = beta * (sparsity - averageState);*/

	float sum = 0.0f;
	
	int weightIndex = 0;

	for (int dx = -inputReceptiveFieldRadius.x; dx <= inputReceptiveFieldRadius.x; dx++)
	for (int dy = -inputReceptiveFieldRadius.y; dy <= inputReceptiveFieldRadius.y; dy++) {
		int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
		
		if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
			float input = read_imagef(inputs, inputPosition).x;
	
			float recon = read_imagef(reconstruction, inputPosition).x;
				
			int4 weightPosition = (int4)(columnPosition.x, columnPosition.y, weightIndex, 0);
	
			float2 prevWeight = read_imagef(columnFeedForwardWeightsPrev, weightPosition).xy;
			
			sum += (input - recon) * prevWeight.x;
		}
		
		weightIndex++;
	}
	
	float hiddenError = sum / weightIndex * thisActivation * (1.0f - thisActivation);
	float sparsity = localActivity / weightIndex;
	
	weightIndex = 0;

	for (int dx = -inputReceptiveFieldRadius.x; dx <= inputReceptiveFieldRadius.x; dx++)
	for (int dy = -inputReceptiveFieldRadius.y; dy <= inputReceptiveFieldRadius.y; dy++) {
		int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
		
		if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
			float input = read_imagef(inputs, inputPosition).x;
	
			float recon = read_imagef(reconstruction, inputPosition).x;
				
			int4 weightPosition = (int4)(columnPosition.x, columnPosition.y, weightIndex, 0);
	
			float2 prevWeight = read_imagef(columnFeedForwardWeightsPrev, weightPosition).xy;
			
			float delta = prevWeight.y * columnMomentum + alpha * 0.5f * ((input - recon) * thisState.x + hiddenError * input);// + beta * (sparsity - thisState.y) * input;
			
			float newWeight = prevWeight.x + delta;
			
			write_imagef(columnFeedForwardWeights, weightPosition, (float4)(newWeight, delta, 0.0f, 0.0f));
		}
		
		weightIndex++;
	}
	
	// Bias
	int4 weightPosition = (int4)(columnPosition.x, columnPosition.y, weightIndex, 0);

	float2 prevWeight = read_imagef(columnFeedForwardWeightsPrev, weightPosition).xy;
	
	float delta = prevWeight.y * columnMomentum + alpha * hiddenError;
	
	float newWeight = prevWeight.x + delta;// + beta * (sparsity - thisState.y);
	
	write_imagef(columnFeedForwardWeights, weightPosition, (float4)(newWeight, delta, 0.0f, 0.0f));
}

void kernel layerCellActivate(read_only image2d_t columnStates, read_only image3d_t cellStatesPrev, read_only image3d_t cellPredictionsPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev,
	write_only image3d_t cellStates, int cellsInColumn, int2 lateralConnectionsRadii, float cellTraceDecay, uint2 seed)
{
	/*int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
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
		//float prediction = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float umodulatedCellState = (1.0f - allCellsIncrease) * (ci == maxCellPredictionIndex ? 1.0f : 0.0f) + allCellsIncrease;
	
		//float umodulatedCellState = (1.0f - maximum) * prediction + maximum;
		
		float newCellState = umodulatedCellState * columnState;
	
		float prevTrace = read_imagef(cellStatesPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).y;
	
		float newTrace = fmax((1.0f - cellTraceDecay) * prevTrace, newCellState);
	
		write_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(newCellState, newTrace, 0.0f, 0.0f));
	}
}

void kernel layerCellWeightUpdate(read_only image2d_t columnStates, read_only image2d_t columnPredictionsPrev, read_only image3d_t cellPredictionsPrev, read_only image3d_t cellStates, read_only image3d_t cellStatesPrev, read_only image2d_t nextLayerContextPrev, read_only image3d_t segmentStatesPrev, read_only image3d_t cellWeightsPrev,
	write_only image3d_t cellWeights, int cellsInColumn, int2 layerSize, int2 lateralConnectionsRadii, int numSegmentsPerCell, float2 layerSizeMinusOneInv, int2 nextLayerSize, int2 nextLayerSizeMinusOne, float tdError, float alpha, float beta, float gamma, float temperature, float eligibilityDecay)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 normalizedColumnCoords = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	int2 connectionCoordsNextCenter = (int2)(normalizedColumnCoords.x * nextLayerSizeMinusOne.x, normalizedColumnCoords.y * nextLayerSizeMinusOne.y);
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	float columnPredictionPrev = read_imagef(columnPredictionsPrev, columnPosition).x;
	
	//float tdError = read_imagef(columnTdErrors, columnPosition).x;
	
	//float learn = tdError > 0.0f ? 1.0f : 0.0f;
	
	//float predictionError = columnState - columnPredictionPrev;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		float2 cellPredictionPrev = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).xy;
		
		float cellError = cellState - cellPredictionPrev.y;
		
		float errors[MAX_SEGMENTS_PER_CELL];
		
		int wi = 0;
		
		if (cellState > 0.5f) {
			for (int i = 0; i < numSegmentsPerCell; i++) {
				float value = read_imagef(segmentStatesPrev, (int4)(columnPosition.x, columnPosition.y, ci * numSegmentsPerCell + i, 0)).x;
		
				if (value == cellPredictionPrev.y)
					errors[i] = 1.0f - value;
				else
					errors[i] = 0.0f - value;
			}
		}
		else {
			for (int i = 0; i < numSegmentsPerCell; i++) {
				float value = read_imagef(segmentStatesPrev, (int4)(columnPosition.x, columnPosition.y, ci * numSegmentsPerCell + i, 0)).x;
		
				errors[i] = 0.0f - value;
			}
		}
		
		// Go through all connections and update them
		for (int dx = -lateralConnectionsRadii.x; dx <= lateralConnectionsRadii.x; dx++)
		for (int dy = -lateralConnectionsRadii.y; dy <= lateralConnectionsRadii.y; dy++) {
			int2 connectionCoords = (int2)(columnPosition.x + dx, columnPosition.y + dy);
				
			if (connectionCoords.x >= 0 && connectionCoords.x < layerSize.x && connectionCoords.y >= 0 && connectionCoords.y < layerSize.y) {	
				for (int cio = 0; cio < cellsInColumn; cio++) {
					float connection = read_imagef(cellStatesPrev, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
	
					for (int i = 0; i < numSegmentsPerCell; i++) {
						int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
					
						float2 cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).xy;
						
						float eligibility = errors[i] * connection;
						
						float newTrace = (1.0f - eligibilityDecay) * cellWeightPrev.y + beta * exp(-fabs(cellWeightPrev.y) * temperature) * eligibility;
						
						float2 newCellWeight = (float2)(cellWeightPrev.x + alpha * newTrace, newTrace);
						
						write_imagef(cellWeights, weightPosition, (float4)(newCellWeight.x, newCellWeight.y, 0.0f, 0.0f));
						
						wi++;
					}
				}
				
				// Additional context from next layer
				int2 connectionCoordsNext = (int2)(connectionCoordsNextCenter.x + dx, connectionCoordsNextCenter.y + dy);
			
				if (connectionCoordsNext.x >= 0 && connectionCoordsNext.x < nextLayerSize.x && connectionCoordsNext.y >= 0 && connectionCoordsNext.y < nextLayerSize.y) {
					float nextContextPrev = read_imagef(nextLayerContextPrev, connectionCoordsNext).x;
	
					for (int i = 0; i < numSegmentsPerCell; i++) {
						int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
					
						float2 cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).xy;
				
						float eligibility = errors[i] * nextContextPrev;
						
						float newTrace = (1.0f - eligibilityDecay) * cellWeightPrev.y + beta * exp(-fabs(cellWeightPrev.y) * temperature) * eligibility;
						
						float2 newCellWeight = (float2)(cellWeightPrev.x + alpha * newTrace, newTrace);
						
						write_imagef(cellWeights, weightPosition, (float4)(newCellWeight.x, newCellWeight.y, 0.0f, 0.0f));
						
						wi++;
					}
				}
				else
					wi += numSegmentsPerCell;
			}
			else
				wi += numSegmentsPerCell * (cellsInColumn + 1);
		}
	}
}

void kernel layerCellWeightUpdateLast(read_only image2d_t columnStates, read_only image2d_t columnPredictionsPrev, read_only image3d_t cellPredictionsPrev, read_only image3d_t cellStates, read_only image3d_t cellStatesPrev, read_only image3d_t segmentStatesPrev, read_only image3d_t cellWeightsPrev,
	write_only image3d_t cellWeights, int cellsInColumn, int2 layerSize, int2 lateralConnectionsRadii, int numSegmentsPerCell, float tdError, float alpha, float beta, float gamma, float temperature, float eligibilityDecay)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	float columnPredictionPrev = read_imagef(columnPredictionsPrev, columnPosition).x;
	
	//float tdError = read_imagef(columnTdErrors, columnPosition).x;
	
	//float predictionError = columnState - columnPredictionPrev;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		float2 cellPredictionPrev = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).xy;
		
		float cellError = cellState - cellPredictionPrev.y;//(cellState - cellPredictionPrev);//((1.0f - columnState) * columnPredictionPrev + columnState) * 
		
		float errors[MAX_SEGMENTS_PER_CELL];
		
		int wi = 0;
		
		if (cellState > 0.5f) {
			for (int i = 0; i < numSegmentsPerCell; i++) {
				float value = read_imagef(segmentStatesPrev, (int4)(columnPosition.x, columnPosition.y, ci * numSegmentsPerCell + i, 0)).x;
		
				if (value == cellPredictionPrev.y)
					errors[i] = 1.0f - value;
				else
					errors[i] = 0.0f - value;
			}
		}
		else {
			for (int i = 0; i < numSegmentsPerCell; i++) {
				float value = read_imagef(segmentStatesPrev, (int4)(columnPosition.x, columnPosition.y, ci * numSegmentsPerCell + i, 0)).x;
		
				errors[i] = 0.0f - value;
			}
		}
		
		// Go through all connections and update them
		for (int dx = -lateralConnectionsRadii.x; dx <= lateralConnectionsRadii.x; dx++)
		for (int dy = -lateralConnectionsRadii.y; dy <= lateralConnectionsRadii.y; dy++) {
			int2 connectionCoords = (int2)(columnPosition.x + dx, columnPosition.y + dy);
			
			if (connectionCoords.x >= 0 && connectionCoords.x < layerSize.x && connectionCoords.y >= 0 && connectionCoords.y < layerSize.y) {	
				for (int cio = 0; cio < cellsInColumn; cio++) {
					float connection = read_imagef(cellStatesPrev, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
	
					for (int i = 0; i < numSegmentsPerCell; i++) {
						int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
					
						float2 cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).xy;
						
						float eligibility = errors[i] * connection;
						
						float newTrace = (1.0f - eligibilityDecay) * cellWeightPrev.y + beta * exp(-fabs(cellWeightPrev.y) * temperature) * eligibility;
						
						float2 newCellWeight = (float2)(cellWeightPrev.x + alpha * newTrace, newTrace);
						
						write_imagef(cellWeights, weightPosition, (float4)(newCellWeight.x, newCellWeight.y, 0.0f, 0.0f));
						
						wi++;
					}
				}
			}
			else
				wi += numSegmentsPerCell * cellsInColumn;
		}
	}
}

void kernel layerCellPredict(read_only image3d_t cellStates, read_only image3d_t cellStatesPrev, read_only image3d_t cellWeights, read_only image2d_t nextLayerContext, read_only image2d_t nextLayerContextPrev,
	write_only image3d_t cellPredictions, write_only image3d_t segmentStates, int cellsInColumn, int2 layerSize, int2 lateralConnectionsRadii, int numSegmentsPerCell, float2 layerSizeMinusOneInv, int2 nextLayerSize, int2 nextLayerSizeMinusOne)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 normalizedColumnCoords = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	int2 connectionCoordsNextCenter = (int2)(normalizedColumnCoords.x * nextLayerSizeMinusOne.x, normalizedColumnCoords.y * nextLayerSizeMinusOne.y);
		
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float sums[MAX_SEGMENTS_PER_CELL];
		
		for (int i = 0; i < numSegmentsPerCell; i++)
			sums[i] = 0.0f;
		
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		int wi = 0;
		
		// Go through all connections 
		for (int dx = -lateralConnectionsRadii.x; dx <= lateralConnectionsRadii.x; dx++)
		for (int dy = -lateralConnectionsRadii.y; dy <= lateralConnectionsRadii.y; dy++) {
			int2 connectionCoords = (int2)(columnPosition.x + dx, columnPosition.y + dy);
			
			if (connectionCoords.x >= 0 && connectionCoords.x < layerSize.x && connectionCoords.y >= 0 && connectionCoords.y < layerSize.y) {	
				for (int cio = 0; cio < cellsInColumn; cio++) {
					float connectionState = read_imagef(cellStates, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
					//float connectionStatePrev = read_imagef(cellStatesPrev, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
					
					for (int i = 0; i < numSegmentsPerCell; i++) {
						int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
					
						float cellWeight = read_imagef(cellWeights, weightPosition).x;
			
						sums[i] += cellWeight * connectionState;
						
						wi++;
					}
				}
				
				int2 connectionCoordsNext = (int2)(connectionCoordsNextCenter.x + dx, connectionCoordsNextCenter.y + dy);
				
				if (connectionCoordsNext.x >= 0 && connectionCoordsNext.x < nextLayerSize.x && connectionCoordsNext.y >= 0 && connectionCoordsNext.y < nextLayerSize.y) {
					float nextContext = read_imagef(nextLayerContext, connectionCoordsNext).x;
					//float nextContextPrev = read_imagef(nextLayerContextPrev, connectionCoordsNext).x;
					
					for (int i = 0; i < numSegmentsPerCell; i++) {
						int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
				
						float cellWeight = read_imagef(cellWeights, weightPosition).x;
					
						sums[i] += cellWeight * nextContext;
						
						wi++;
					}
				}
				else
					wi += numSegmentsPerCell;
			}
			else
				wi += numSegmentsPerCell * (cellsInColumn + 1); // + 1 for context from higher layer
		}
		
		float maximum = 0.0f;
		
		for (int i = 0; i < numSegmentsPerCell; i++) {
			float s = sigmoid(sums[i]);
			
			maximum = fmax(maximum, s);
			
			write_imagef(segmentStates, (int4)(columnPosition.x, columnPosition.y, ci * numSegmentsPerCell + i, 0), (float4)(s, 0.0f, 0.0f, 0.0f));
		}
		
		write_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(maximum > 0.5f ? 1.0f : 0.0f, maximum, 0.0f, 0.0f));
	}
}

void kernel layerCellPredictLast(read_only image3d_t cellStates, read_only image3d_t cellStatesPrev, read_only image3d_t cellWeights,
	write_only image3d_t cellPredictions, write_only image3d_t segmentStates, int cellsInColumn, int2 layerSize, int2 lateralConnectionsRadii, int numSegmentsPerCell)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float sums[MAX_SEGMENTS_PER_CELL];
		
		for (int i = 0; i < numSegmentsPerCell; i++)
			sums[i] = 0.0f;
			
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		int wi = 0;
		
		// Go through all connections 
		for (int dx = -lateralConnectionsRadii.x; dx <= lateralConnectionsRadii.x; dx++)
		for (int dy = -lateralConnectionsRadii.y; dy <= lateralConnectionsRadii.y; dy++) {
			int2 connectionCoords = (int2)(columnPosition.x + dx, columnPosition.y + dy);

			if (connectionCoords.x >= 0 && connectionCoords.x < layerSize.x && connectionCoords.y >= 0 && connectionCoords.y < layerSize.y) {	
				for (int cio = 0; cio < cellsInColumn; cio++) {
					float connectionState = read_imagef(cellStates, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
					//float connectionStatePrev = read_imagef(cellStatesPrev, (int4)(connectionCoords.x, connectionCoords.y, cio, 0)).x;
					
					for (int i = 0; i < numSegmentsPerCell; i++) {
						int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
					
						float cellWeight = read_imagef(cellWeights, weightPosition).x;
			
						sums[i] += cellWeight * connectionState;
						
						wi++;
					}
				}
			}
			else
				wi += cellsInColumn * numSegmentsPerCell;
		}
		
		float maximum = 0.0f;
		
		for (int i = 0; i < numSegmentsPerCell; i++) {
			float s = sigmoid(sums[i]);
			
			maximum = fmax(maximum, s);
			
			write_imagef(segmentStates, (int4)(columnPosition.x, columnPosition.y, ci * numSegmentsPerCell + i, 0), (float4)(s, 0.0f, 0.0f, 0.0f));
		}
		
		write_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(maximum > 0.5f ? 1.0f : 0.0f, maximum, 0.0f, 0.0f));
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
	
	write_imagef(columnPredictions, columnPosition, (float4)(output, 0.0f, 0.0f, 0.0f));
}

void kernel layerAssignQ(read_only image3d_t cellQValuesPrev, read_only image3d_t cellStatesPrev, write_only image3d_t cellQValues,
	int cellsInColumn, float alpha)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float qPrev = read_imagef(cellQValuesPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float cellEligibility = read_imagef(cellStatesPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).y;
		
		float storeQ = qPrev + cellEligibility * alpha;
		
		write_imagef(cellQValues, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(storeQ, 0.0f, 0.0f, 0.0f));
	}
}

void kernel layerColumnQ(read_only image3d_t cellQValuesPrev, read_only image3d_t cellStatesPrev, read_only image3d_t cellStates, read_only image2d_t columnStates, read_only image2d_t columnStatesNext, read_only image2d_t columnQValuesNext, write_only image2d_t columnQValues,
	int cellsInColumn, float2 layerSizeMinusOneInv, int2 nextLayerSize, int2 nextLayerSizeMinusOne)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 columnPositionNormalized = (float2)(columnPosition.x * layerSizeMinusOneInv.x, columnPosition.y * layerSizeMinusOneInv.y);
	int2 nextLayerPositionCenter = (int2)(columnPositionNormalized.x * nextLayerSizeMinusOne.x, columnPositionNormalized.y * nextLayerSizeMinusOne.y);
	
	float sum = 0.0f;
	float divisor = 0.0f;

	for (int ci = 0; ci < cellsInColumn; ci++) {
		float state = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		float cellQ = read_imagef(cellQValuesPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		sum += state * cellQ;
		divisor += state;
	}
	
	float thisQ = sum / fmax(minDivisor, divisor);
	
	float output = thisQ;
		
	write_imagef(columnQValues, columnPosition, (float4)(output, 0.0f, 0.0f, 0.0f));
}

void kernel layerColumnQLast(read_only image3d_t cellQValuesPrev, read_only image3d_t cellStatesPrev, read_only image3d_t cellStates, write_only image2d_t columnQValues,
	int cellsInColumn)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float sum = 0.0f;
	float divisor = 0.0f;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float state = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		float cellQ = read_imagef(cellQValuesPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		sum += state * cellQ;
		divisor += state;
	}
	
	float thisQ = sum / fmax(minDivisor, divisor);
	
	write_imagef(columnQValues, columnPosition, (float4)(thisQ, 0.0f, 0.0f, 0.0f));
}

void kernel initializePartThree(write_only image2d_t inputBiases, uint2 seed, float minBias, float maxBias) {
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 130;
	int2 inputPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float bias = randFloat(&seedValue) * (maxBias - minBias) + minBias;
	
	write_imagef(inputBiases, inputPosition, (float4)(bias, 0.0f, 0.0f, 0.0f));
}

void kernel reconstructInput(read_only image3d_t columnFeedForwardWeights, read_only image2d_t inputBiases, read_only image2d_t columnStates, write_only image2d_t reconstruction,
	int2 reverseReceptiveFieldRadius, int2 sdrReceptiveFieldRadius, int2 inputSizeMinusOne, float2 inputSizeMinusOneInv, int2 sdrSize, int2 sdrSizeMinusOne, float2 sdrSizeMinusOneInv)
{
	int2 inputPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputPositionNormalized = (float2)(inputPosition.x * inputSizeMinusOneInv.x, inputPosition.y * inputSizeMinusOneInv.y);
	float2 sdrPositionCenter = (float2)(inputPositionNormalized.x * sdrSizeMinusOne.x, inputPositionNormalized.y * sdrSizeMinusOne.y);
	
	float sum = 0.0f;

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
				
				float source = read_imagef(columnStates, sdrPosition).x;

				int weightIndex = (sdrReceptiveFieldRadius.y + rdy) + (sdrReceptiveFieldRadius.x + rdx) * (sdrReceptiveFieldRadius.y * 2 + 1);

				float weight = read_imagef(columnFeedForwardWeights, (int4)(sdrPosition.x, sdrPosition.y, weightIndex, 0)).x;
				
				sum += source * weight;
			}
		}
	}

	float bias = read_imagef(inputBiases, inputPosition).x;
				
	sum += bias;
	
	write_imagef(reconstruction, inputPosition, (float4)(sum, 0.0f, 0.0f, 0.0f));
}

void kernel inputBiasUpdate(read_only image2d_t inputs, read_only image2d_t reconstruction, read_only image2d_t inputBiasesPrev, write_only image2d_t inputBiases, 
	float gamma)
{
	int2 inputPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 prevBias = read_imagef(inputBiasesPrev, inputPosition).xy;
	
	float recon = read_imagef(reconstruction, inputPosition).x;
	float input = read_imagef(inputs, inputPosition).x;
	
	float delta = prevBias.y * columnMomentum + gamma * (input - recon);
	
	float newBias = prevBias.x + delta;
	
	write_imagef(inputBiases, inputPosition, (float4)(newBias, delta, 0.0f, 0.0f));
}

void kernel gaussianBlurX(read_only image2d_t source, write_only image2d_t destination, float2 sizeInv, float kernelWidth) {
	int2 destinationPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 destinationPositionNormalized = (float2)(destinationPosition.x * sizeInv.x, destinationPosition.y * sizeInv.y);
	
	float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x - 4.0f * kernelWidth, destinationPositionNormalized.y)) * 0.05f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x - 3.0f * kernelWidth, destinationPositionNormalized.y)) * 0.09f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x - 2.0f * kernelWidth, destinationPositionNormalized.y)) * 0.12f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x - kernelWidth, destinationPositionNormalized.y)) * 0.15f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y)) * 0.16f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x + kernelWidth, destinationPositionNormalized.y)) * 0.15f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x + 2.0f * kernelWidth, destinationPositionNormalized.y)) * 0.12f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x + 3.0f * kernelWidth, destinationPositionNormalized.y)) * 0.09f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x + 4.0f * kernelWidth, destinationPositionNormalized.y)) * 0.05f;
 
	write_imagef(destination, destinationPosition, sum);
}

void kernel gaussianBlurY(read_only image2d_t source, write_only image2d_t destination, float2 sizeInv, float kernelWidth) {
	int2 destinationPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 destinationPositionNormalized = (float2)(destinationPosition.x * sizeInv.x, destinationPosition.y * sizeInv.y);
	
	float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y - 4.0f * kernelWidth)) * 0.05f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y - 3.0f * kernelWidth)) * 0.09f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y - 2.0f * kernelWidth)) * 0.12f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y - kernelWidth)) * 0.15f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y)) * 0.16f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y + kernelWidth)) * 0.15f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y + 2.0f * kernelWidth)) * 0.12f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y + 3.0f * kernelWidth)) * 0.09f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y + 4.0f * kernelWidth)) * 0.05f;
 
	write_imagef(destination, destinationPosition, sum);
}