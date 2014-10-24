constant sampler_t normalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;
	
constant sampler_t unnormalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;
	
constant float activationIntensity = 16.0f;
constant float columnIntensity = 16.0f;
constant float cellStateIntensity = 16.0f;
constant float cellPredictionIntensity = 4.0f;

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
	write_only image3d_t interlayerWeights,
	write_only image3d_t cellStates, write_only image3d_t cellWeights, write_only image3d_t cellPredictions, int cellsInColumn,
	int layerWidth, int receptiveFieldSize, int lateralConnectionsSize, uint2 seed, float fminWeight, float fmaxWeight)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1)) * 100;

	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));

	write_imagef(columnActivations, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	write_imagef(columnStates, columnPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));

	for (int wi = 0; wi < receptiveFieldSize; wi++) {
		int4 weightPosition = (int4)(columnPosition.x, columnPosition.y, wi, 0);
	
		float columnWeight = randFloat(&seedValue) * (fmaxWeight - fminWeight) + fminWeight;
		float interlayerWeight = randFloat(&seedValue) * (fmaxWeight - fminWeight) + fminWeight;
	
		write_imagef(columnWeights, weightPosition, (float4)(columnWeight, columnWeight, columnWeight, columnWeight));
		write_imagef(interlayerWeights, weightPosition, (float4)(interlayerWeight, interlayerWeight, interlayerWeight, interlayerWeight));
	}
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		write_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
		write_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	
		int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
	
		for (int wi = 0; wi < lateralConnectionsSize; wi++) {
			int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
			float cellWeight = randFloat(&seedValue) * (fmaxWeight - fminWeight) + fminWeight;
	
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
		
		float weight = read_imagef(columnWeights, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
		float prevState = read_imagef(columnStatesPrev, normalizedClampedNearestSampler, inputPositionNormalized).x;
		
		float difference = weight - prevState;
		
		sum += difference * difference;
		
		weightIndex++;
	}
	
	float activation = exp(-sum * activationIntensity);
	
	write_imagef(columnActivations, columnPosition, (float4)(activation, activation, activation, activation));
}

void kernel layerColumnInhibit(read_only image2d_t columnActivations, write_only image2d_t columnStates, float2 layerSizeInv, int2 receptiveFieldRadius, float2 layerReceptiveFieldStep) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 layerCenterPositionNormalized = (float2)(columnPosition.x * layerSizeInv.x, columnPosition.y * layerSizeInv.y);

	float maximum = 0.0f;

	int weightIndex = 0;
	
	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		float2 layerPositionNormalized = layerCenterPositionNormalized + (float2)(dx * layerReceptiveFieldStep.x, dy * layerReceptiveFieldStep.y);
		
		float activation = read_imagef(columnActivations, normalizedClampedNearestSampler, layerPositionNormalized).x;
		
		maximum = fmax(maximum, activation);
		
		weightIndex++;
	}
	
	float thisActivation = read_imagef(columnActivations, normalizedClampedNearestSampler, layerCenterPositionNormalized).x;
	
	float inhibitedResult = exp((thisActivation - maximum) * columnIntensity);

	write_imagef(columnStates, columnPosition, (float4)(inhibitedResult, inhibitedResult, inhibitedResult, inhibitedResult));
}

void kernel layerColumnWeightUpdate(read_only image2d_t columnStatesPrev, read_only image2d_t columnStates, read_only image3d_t columnWeightsPrev, write_only image3d_t columnWeights, float2 layerSizeInv, int2 receptiveFieldRadius, float2 inputReceptiveFieldStep, float alpha) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 inputCenterPositionNormalized = (float2)(columnPosition.x * layerSizeInv.x, columnPosition.y * layerSizeInv.y);

	float state = read_imagef(columnStates, columnPosition).x;
		
	// Adjust weights by their source activations and error
	int weightIndex = 0;
	
	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		float2 inputPositionNormalized = inputCenterPositionNormalized + (float2)(dx * inputReceptiveFieldStep.x, dy * inputReceptiveFieldStep.y);
		
		float prevState = read_imagef(columnStatesPrev, normalizedClampedNearestSampler, inputPositionNormalized).x;
		
		float prevWeight = read_imagef(columnWeightsPrev, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0)).x;
		
		float change = alpha * (state * (prevState - prevWeight * state));
		
		float newWeight = prevWeight + change;
		
		write_imagef(columnWeights, (int4)(columnPosition.x, columnPosition.y, weightIndex, 0), (float4)(newWeight, newWeight, newWeight, newWeight));
		
		weightIndex++;
	}
}

void kernel layerCellActivate(read_only image2d_t columnStates, read_only image3d_t cellStatesPrev, read_only image3d_t cellPredictionsPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev,
	write_only image3d_t cellStates, int cellsInColumn, int layerWidth, int2 lateralConnectionsRadii)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	
	//float fmaxVectorMagnitude = 0.0f;
	
	//float cellVectorMagnitudes[4];
	
	float minPredictionError = 1.0f;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float prediction = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float predictionError = fabs(columnState - prediction);
		
		minPredictionError = min(minPredictionError, predictionError);
		
		/*int weightSecondCoordinate = ci + columnPosition.y * cellsInColumn;
		
		float3 cellVector = (float3)(0.0f, 0.0f, 0.0f);
		
		// Go through all connections and update them
		int wi = 0;
		
		for (int dx = -lateralConnectionsRadii.x; dx <= lateralConnectionsRadii.x; dx++)
		for (int dy = -lateralConnectionsRadii.y; dy <= lateralConnectionsRadii.y; dy++)
		for (int cio = 0; cio < cellsInColumn; cio++) {
			int4 weightPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
			float cellWeightPrev = read_imagef(cellWeightsPrev, weightPosition).x;
		
			float connectionState = read_imagef(cellStatesPrev, unnormalizedClampedNearestSampler, (int4)(columnPosition.x + dx, columnPosition.y + dy, cio, 0)).x;
			
			cellVector += cellWeightPrev * connectionState * ((float3)(dx, dy, cio - ci));
			
			wi++;
		}
		
		cellVectorMagnitudes[ci] = length(cellVector);
		
		fmaxVectorMagnitude = fmax(fmaxVectorMagnitude, cellVectorMagnitudes[ci]);*/
	}
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float prediction = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float predictionError = fabs(columnState - prediction);
		
		float newCellState = exp((minPredictionError - predictionError) * cellStateIntensity) * columnState;
	
		write_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0), (float4)(newCellState, newCellState, newCellState, newCellState));
	}
}

void kernel layerCellWeightUpdate(read_only image2d_t columnStates, read_only image3d_t cellStates, read_only image3d_t cellStatesPrev, read_only image3d_t cellPredictionsPrev, read_only image3d_t cellWeightsPrev, read_only image2d_t columnPredictionsPrev,
	write_only image3d_t cellWeights, int cellsInColumn, int layerWidth, int2 lateralConnectionsRadii, float alpha)
{
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float columnState = read_imagef(columnStates, columnPosition).x;
	float columnPredictionPrev = read_imagef(columnPredictionsPrev, columnPosition).x;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float cellState = read_imagef(cellStates, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float prediction = read_imagef(cellPredictionsPrev, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
		
		float predictionError = (columnState - columnPredictionPrev);
		
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
		
		/*int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float cellBiasPrev = read_imagef(cellWeightsPrev, biasPosition).x;

		float newCellBias = cellBiasPrev + alpha * predictionError;
		
		write_imagef(cellWeights, biasPosition, (float4)(newCellBias, newCellBias, newCellBias, newCellBias));*/
	}
}

void kernel layerCellPredict(read_only image2d_t columnStates, read_only image3d_t cellStates, read_only image3d_t cellWeights,
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
		
		/*int4 biasPosition = (int4)(columnPosition.x, weightSecondCoordinate, wi, 0);
		
		float bias = read_imagef(cellWeights, biasPosition).x;
		
		sum += bias;*/
		
		float prediction = 1.0f - exp(-fmax(0.0f, sum) * cellPredictionIntensity);
		
		write_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0), prediction);
	}
}

void kernel layerColumnPrediction(read_only image3d_t cellPredictions, read_only image3d_t cellStates, write_only image2d_t columnPredictions, int cellsInColumn) {
	int2 columnPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float maxPrediction = 0.0f;
	
	for (int ci = 0; ci < cellsInColumn; ci++) {
		float prediction = read_imagef(cellPredictions, (int4)(columnPosition.x, columnPosition.y, ci, 0)).x;
	
		maxPrediction = max(maxPrediction, prediction);
	}
	
	float output = maxPrediction;
	
	write_imagef(columnPredictions, columnPosition, (float4)(output, output, output, output));
}

void kernel layerReconstructPrediction(read_only image2d_t columnPredictions, read_only image3d_t columnWeights, write_only image2d_t reconstruction, float2 reconstructionSizeInv, int2 receptiveFieldRadius, float2 inputReceptiveFieldStep) {
	int2 reconstructionPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 reconstructionCenterPositionNormalized = (float2)(reconstructionPosition.x * reconstructionSizeInv.x, reconstructionPosition.y * reconstructionSizeInv.y);
	
	float sum = 0.0f;

	for (int dx = -receptiveFieldRadius.x; dx <= receptiveFieldRadius.x; dx++)
	for (int dy = -receptiveFieldRadius.y; dy <= receptiveFieldRadius.y; dy++) {
		float2 reconstructionPositionNormalized = reconstructionCenterPositionNormalized + (float2)(dx * inputReceptiveFieldStep.x, dy * inputReceptiveFieldStep.y);
		int2 reconstructionPosition = (int2)(reconstructionPositionNormalized.x / reconstructionSizeInv.x, reconstructionPositionNormalized.y / reconstructionSizeInv.y);
		
		int myWeightIndex = (-dy + receptiveFieldRadius.y) + (-dx + receptiveFieldRadius.x) * (receptiveFieldRadius.y * 2 + 1);
		
		float weight = read_imagef(columnWeights, unnormalizedClampedNearestSampler, (int4)(reconstructionPosition.x, reconstructionPosition.y, myWeightIndex, 0)).x;
		float prediction = read_imagef(columnPredictions, normalizedClampedNearestSampler, reconstructionPositionNormalized).x;
		
		sum += weight * (prediction * 2.0f - 1.0f);
	}
	
	float recon = sigmoid((sum - ((2 * receptiveFieldRadius.x + 1) * (2 * receptiveFieldRadius.y + 1))) * columnIntensity) * 2.0f - 1.0f;
	
	write_imagef(reconstruction, reconstructionPosition, (float4)(recon, recon, recon, recon));
}  