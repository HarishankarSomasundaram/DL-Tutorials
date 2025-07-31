using CSV, DataFrames, Flux, MLUtils, JLD2, Statistics, Dates, Random, StatsBase
using Flux: Adam, AdamW, mse, crossentropy, sigmoid, DataLoader
using MLUtils: splitobs
using CUDA

# Set random seed for reproducibility
Random.seed!(42)

# Check for GPU availability and use it
if CUDA.functional()
    device = gpu
    println("Using device: GPU")
    CUDA.allowscalar(false)  # Disable scalar indexing to catch issues early
else
    device = cpu
    println("Using device: CPU (GPU not available)")
end

# Hyperparameters
data_folder = "./data"
window_sizes = [30, 90, 270]
max_params = 1000
epochs = 10
batch_size = 128

function preprocess_data(df::DataFrame)
    """Preprocess the data for LSTM training"""
    # Remove datetime and changepoint columns
    feature_cols = [col for col in names(df) if !(col in ["datetime", "changepoint", "anomaly", "dataset_id"])]
    
    # Extract features and labels
    X = Matrix{Float32}(df[:, feature_cols])  # Ensure Float32
    y = Float32.(df.anomaly)
    dataset_ids = df.dataset_id
    
    # Normalize features (Z-score normalization)
    X_mean = mean(X, dims=1)
    X_std = std(X, dims=1)
    X_std[X_std .== 0] .= 1f0  # Avoid division by zero
    X_normalized = (X .- X_mean) ./ X_std
    
    return X_normalized, y, dataset_ids, X_mean, X_std, feature_cols
end

function create_sequences(X, y, dataset_ids, window_size::Int)
    """Create sequences for LSTM training"""
    sequences_X = []
    sequences_y = []
    
    # Group by dataset to maintain temporal order within each dataset
    unique_datasets = unique(dataset_ids)
    
    for dataset_id in unique_datasets
        dataset_mask = dataset_ids .== dataset_id
        dataset_X = X[dataset_mask, :]
        dataset_y = y[dataset_mask]
        
        # Create sequences within this dataset
        for i in 1:(size(dataset_X, 1) - window_size + 1)
            seq_X = dataset_X[i:(i + window_size - 1), :]'  # (features, time_steps)
            seq_y = dataset_y[i + window_size - 1]  # Use the last label in the sequence
            
            push!(sequences_X, Float32.(seq_X))
            push!(sequences_y, Float32(seq_y))
        end
    end
    
    return sequences_X, sequences_y
end

function create_lstm_model(input_size::Int, window_size::Int; max_params::Int=1000)
    """Create LSTM model with parameter constraint"""
    
    # Calculate hidden size based on parameter constraint
    hidden_size = 16  # Start with a reasonable size
    
    # Estimate parameters and adjust if needed
    while hidden_size > 4
        lstm_params = 4 * hidden_size * (input_size + hidden_size + 1)
        dense_params = hidden_size + 1
        total_params = lstm_params + dense_params
        
        if total_params <= max_params
            break
        end
        hidden_size -= 2
    end
    
    println("Model architecture: LSTM($input_size, $hidden_size) + Dense($hidden_size, 1)")
    
    # Estimate final parameters
    lstm_params = 4 * hidden_size * (input_size + hidden_size + 1)
    dense_params = hidden_size + 1
    total_params = lstm_params + dense_params
    println("Estimated parameters: $total_params")
    
    model = Chain(
        LSTM(input_size => hidden_size),
        Dense(hidden_size => 1)  # Remove sigmoid - we'll apply it in loss calculation
    )
    
    return model |> device  # Move model to GPU
end

function safe_scalar_extract(x)
    """Safely extract scalar from GPU array"""
    if isa(x, AbstractArray)
        if device == gpu && isa(x, CuArray)
            return Float32(Array(x)[1])  # Move to CPU first, then extract and convert
        elseif length(x) == 1
            return Float32(x[1])
        else
            error("Cannot extract scalar from array with length > 1")
        end
    else
        return Float32(x)
    end
end

function evaluate_model(model, test_data)
    """Evaluate model and return accuracy and balanced accuracy - GPU compatible"""
    predictions = Float32[]
    true_labels = Float32[]
    
    for (x, y) in test_data
        # Move data to device
        x_gpu = x |> device
        
        # Reset LSTM state
        Flux.reset!(model[1])
        
        # Get prediction
        output = model(x_gpu)
        ŷ_logit = safe_scalar_extract(output)  # Get logit
        ŷ = sigmoid(ŷ_logit)  # Apply sigmoid to get probability
        
        push!(predictions, ŷ)
        push!(true_labels, Float32(y))
    end
    
    # Convert predictions to binary (threshold = 0.5)
    pred_binary = predictions .>= 0.5f0
    true_binary = true_labels .>= 0.5f0
    
    # Calculate accuracy
    accuracy = mean(pred_binary .== true_binary)
    
    # Calculate balanced accuracy
    tp = sum((pred_binary .== true) .& (true_binary .== true))
    fp = sum((pred_binary .== true) .& (true_binary .== false))
    tn = sum((pred_binary .== false) .& (true_binary .== false))
    fn = sum((pred_binary .== false) .& (true_binary .== true))
    
    # Sensitivity (recall) and specificity
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    # Balanced accuracy
    balanced_accuracy = (sensitivity + specificity) / 2
    
    return Float64(accuracy), Float64(balanced_accuracy)
end

function create_batch_tensors(batch_x, batch_y)
    """Create proper batch tensors for GPU processing"""
    # Stack sequences into a single tensor: (features, time_steps, batch_size)
    batch_tensor = cat(batch_x..., dims=3)
    
    # Create labels tensor
    labels_tensor = reshape(Float32.(batch_y), 1, :)  # (1, batch_size)
    
    return batch_tensor, labels_tensor
end

function train_model_batch_processing(model, train_data, val_data; epochs::Int=100, batch_size::Int=128)
    """Train the LSTM model with proper batch processing - GPU compatible"""
    
    # Setup optimizer with higher learning rate and lower weight decay for better learning
    opt = AdamW(0.003f0, (0.9f0, 0.999f0), 0.01f0)  # lr=0.01, weight_decay=0.0001
    
    # Initialize optimizer state using Flux.setup
    opt_state = Flux.setup(opt, model)
    
    best_val_acc = 0.0
    best_model_state = nothing
    patience = 10
    patience_counter = 0
    
    # Convert to arrays for easier batching
    train_x = [item[1] for item in train_data]
    train_y = [item[2] for item in train_data]
    
    n_batches = ceil(Int, length(train_data) / batch_size)
    
    for epoch in 1:epochs
        # Shuffle data
        indices = randperm(length(train_data))
        train_x_shuffled = train_x[indices]
        train_y_shuffled = train_y[indices]
        
        # Training
        train_loss = 0.0f0
        
        for batch_idx in 1:n_batches
            start_idx = (batch_idx - 1) * batch_size + 1
            end_idx = min(batch_idx * batch_size, length(train_data))
            
            batch_x = train_x_shuffled[start_idx:end_idx]
            batch_y = train_y_shuffled[start_idx:end_idx]
            
            # Create batch tensors and move to device
            batch_tensor, labels_tensor = create_batch_tensors(batch_x, batch_y)
            batch_tensor_gpu = batch_tensor |> device
            labels_tensor_gpu = labels_tensor |> device
            
            # Forward pass and loss computation
            loss, grads = Flux.withgradient(model) do m
                # Reset LSTM state
                Flux.reset!(m[1])
                
                # Process batch through LSTM
                # batch_tensor_gpu is (features, time_steps, batch_size)
                lstm_out = m[1](batch_tensor_gpu)  # LSTM processes all sequences in batch
                
                # Get final outputs from LSTM (last time step)
                final_outputs = lstm_out[:, end, :]  # (hidden_size, batch_size)
                
                # Pass through dense layer
                predictions = m[2](final_outputs)  # (1, batch_size)
                
                # Reshape predictions to match labels
                predictions_flat = reshape(predictions, :)  # (batch_size,)
                labels_flat = reshape(labels_tensor_gpu, :)  # (batch_size,)
                
                # Compute binary cross entropy loss
                mean(Flux.binarycrossentropy.(sigmoid.(predictions_flat), labels_flat))
            end
            
            # Update parameters using the initialized state
            Flux.update!(opt_state, model, grads[1])
            
            # Move loss to CPU for accumulation (safe extraction)
            train_loss += safe_scalar_extract(loss)
        end
        
        avg_train_loss = train_loss / n_batches
        
        # Validation
        if epoch % 3 == 0 || epoch == epochs
            val_acc, val_bal_acc = evaluate_model(model, val_data)
            println("Epoch $epoch: Train Loss = $(round(avg_train_loss, digits=4)), Val Acc = $(round(val_acc*100, digits=2))%, Val Bal Acc = $(round(val_bal_acc*100, digits=2))%")
            
            # Early stopping based on balanced accuracy
            if val_bal_acc > best_val_acc
                best_val_acc = val_bal_acc
                # Move model to CPU for saving
                best_model_state = deepcopy(model |> cpu)
                patience_counter = 0
            else
                patience_counter += 1
                if patience_counter >= patience
                    println("Early stopping at epoch $epoch")
                    break
                end
            end
        end
    end
    
    return best_model_state, best_val_acc
end

# Main execution
println("Loading data...")

all_data = DataFrame[]

# Load anomaly-free data
anomaly_free_file = joinpath(data_folder, "anomaly-free.csv")
if isfile(anomaly_free_file)
    df_normal = CSV.read(anomaly_free_file, DataFrame, delim=';')
    df_normal.anomaly = zeros(nrow(df_normal))  # Add anomaly column
    df_normal.dataset_id = ones(Int, nrow(df_normal))  # Dataset identifier
    if hasproperty(df_normal, :changepoint)
        select!(df_normal, Not(:changepoint))
    end
    push!(all_data, df_normal)
    println("Loaded anomaly-free data: $(nrow(df_normal)) samples")
end

# Load anomaly files
anomaly_files = filter(f -> occursin("anom", lowercase(f)) && endswith(f, ".csv"), readdir(data_folder))

for (i, file) in enumerate(anomaly_files)
    filepath = joinpath(data_folder, file)
    if isfile(filepath)
        df_anom = CSV.read(filepath, DataFrame, delim=';')
        # Ensure anomaly column exists, if not create it
        if !hasproperty(df_anom, :anomaly)
            df_anom.anomaly = ones(nrow(df_anom))  # All anomalies
        end
        df_anom.dataset_id = fill(i + 1, nrow(df_anom))  # Dataset identifier
        if hasproperty(df_anom, :changepoint)
            select!(df_anom, Not(:changepoint))
        end
        push!(all_data, df_anom)
        println("Loaded anomaly data $(file): $(nrow(df_anom)) samples")
    end
end

combined_data = vcat(all_data...)
println("Total combined data: $(nrow(combined_data)) samples")

# Preprocess data
X, y, dataset_ids, X_mean, X_std, feature_cols = preprocess_data(combined_data)

println("Features: ", feature_cols)
println("Feature matrix shape: ", size(X))
println("Label distribution: ", countmap(y))

# Train models for different window sizes
best_models = Dict()

for window_size in window_sizes
    println("\n" * "="^50)
    println("Training model with window size: $window_size")
    println("="^50)
    
    # Create sequences
    sequences_x, sequences_y = create_sequences(X, y, dataset_ids, window_size)
    println("Created $(length(sequences_x)) sequences")
    
    # Create data pairs
    data_pairs = [(seq_x, seq_y) for (seq_x, seq_y) in zip(sequences_x, sequences_y)]
    
    # Time-series aware split (use first 70% for train, next 15% for val, last 15% for test)
    n_total = length(data_pairs)
    n_train = Int(floor(0.7 * n_total))
    n_val = Int(floor(0.15 * n_total))
    
    train_data = data_pairs[1:n_train]
    val_data = data_pairs[(n_train+1):(n_train+n_val)]
    test_data = data_pairs[(n_train+n_val+1):end]
    
    println("Train: $(length(train_data)), Val: $(length(val_data)), Test: $(length(test_data))")
    
    # Create and train model
    input_size = size(sequences_x[1], 1)
    model = create_lstm_model(input_size, window_size, max_params=max_params)
    
    trained_model, best_val_acc = train_model_batch_processing(model, train_data, val_data, epochs=epochs, batch_size=batch_size)
    
    # Final evaluation on test set
    test_acc, test_bal_acc = evaluate_model(trained_model |> device, test_data)
    println("Final Test Accuracy: $(round(test_acc*100, digits=2))%")
    println("Final Test Balanced Accuracy: $(round(test_bal_acc*100, digits=2))%")
    
    # Save model (trained_model is already on CPU)
    model_filename = "lstm_model_window_$(window_size).jld2"
    trained_params = Flux.state(trained_model)
    trained_st = nothing  # Not using explicit state in this implementation
    
    # Save all necessary information for testing
    JLD2.@save model_filename trained_params trained_st X_mean X_std feature_cols window_size input_size
    println("Model saved as: $model_filename")
    
    best_models[window_size] = (
        model=trained_model,
        test_bal_acc=test_bal_acc,
        filename=model_filename
    )
end

# Find best model
best_window = argmax([best_models[w].test_bal_acc for w in window_sizes])
best_window_size = window_sizes[best_window]
best_bal_acc = best_models[best_window_size].test_bal_acc

println("\n" * "="^50)
println("FINAL RESULTS")
println("="^50)
println("Best window size: $best_window_size")
println("Best balanced accuracy: $(round(best_bal_acc*100, digits=2))%")
println("Best model file: $(best_models[best_window_size].filename)")