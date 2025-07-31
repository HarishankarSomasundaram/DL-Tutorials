using CSV, DataFrames, Flux, JLD2, Statistics, StatsBase
using Flux: sigmoid

function preprocess_test_data(test_x, X_mean, X_std, feature_cols)
    """Preprocess test data using saved normalization parameters"""
    # Ensure test_x has the right columns in the right order
    if isa(test_x, DataFrame)
        test_matrix = Matrix{Float32}(test_x[:, feature_cols])
    else
        test_matrix = Float32.(test_x)
    end
    
    # Apply same normalization as training
    test_normalized = (test_matrix .- X_mean) ./ X_std
    return test_normalized
end

function create_test_sequences(X, y, window_size::Int)
    """Create sequences from test data"""
    sequences_X = []
    sequences_y = []
    
    # Create sequences from the entire test set
    for i in 1:(size(X, 1) - window_size + 1)
        seq_X = X[i:(i + window_size - 1), :]'  # (features, time_steps)
        seq_y = y[i + window_size - 1]  # Use the last label in the sequence
        
        push!(sequences_X, Float32.(seq_X))
        push!(sequences_y, Float32(seq_y))
    end
    
    return sequences_X, sequences_y
end

function reconstruct_model(trained_params, input_size, window_size, max_params=1000)
    """Reconstruct the model architecture and load parameters"""
    
    # Calculate hidden size (same logic as in training)
    hidden_size = 16
    while hidden_size > 4
        lstm_params = 4 * hidden_size * (input_size + hidden_size + 1)
        dense_params = hidden_size + 1
        total_params = lstm_params + dense_params
        
        if total_params <= max_params
            break
        end
        hidden_size -= 2
    end
    
    # Create model with same architecture
    model = Chain(
        LSTM(input_size => hidden_size),
        Dense(hidden_size => 1, sigmoid)
    )
    
    # Load trained parameters
    Flux.loadmodel!(model, trained_params)
    
    return model
end

function bal_acc(trained_params, trained_st, test_x, test_y, X_mean, X_std, feature_cols, window_size, input_size)
    """
    Calculate balanced accuracy on test data
    This function performs all necessary transformations and evaluation
    """
    
    # Preprocess test data
    test_x_normalized = preprocess_test_data(test_x, X_mean, X_std, feature_cols)
    
    # Create sequences if needed
    if window_size > 1
        sequences_x, sequences_y = create_test_sequences(test_x_normalized, Float32.(test_y), window_size)
        test_data = [(seq_x, seq_y) for (seq_x, seq_y) in zip(sequences_x, sequences_y)]
    else
        # For window size 1, just use individual samples
        test_data = [(test_x_normalized[i, :], Float32(test_y[i])) for i in 1:size(test_x_normalized, 1)]
    end
    
    # Reconstruct model
    model = reconstruct_model(trained_params, input_size, window_size)
    
    # Make predictions
    predictions = Float32[]
    true_labels = Float32[]
    
    for (x, y) in test_data
        # Reset LSTM state
        Flux.reset!(model[1])
        
        # Get prediction
        if window_size > 1
            ŷ = model(x)[1]  # For sequences
        else
            ŷ = model(reshape(x, :, 1))[1]  # For individual samples
        end
        
        push!(predictions, ŷ)
        push!(true_labels, y)
    end
    
    # Convert predictions to binary (threshold = 0.5)
    pred_binary = predictions .>= 0.5f0
    true_binary = true_labels .>= 0.5f0
    
    # Calculate balanced accuracy
    tp = sum((pred_binary .== 1) .& (true_binary .== 1))
    fp = sum((pred_binary .== 1) .& (true_binary .== 0))
    tn = sum((pred_binary .== 0) .& (true_binary .== 0))
    fn = sum((pred_binary .== 0) .& (true_binary .== 1))
    
    # Sensitivity (recall) and specificity
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    # Balanced accuracy
    balanced_accuracy = (sensitivity + specificity) / 2
    
    return Float64(balanced_accuracy)
end

# Example usage for testing a saved model:
function test_saved_model(model_path, test_x, test_y)
    """
    Test a saved model on holdout test data
    """
    # Load saved model and parameters
    JLD2.@load model_path trained_params trained_st X_mean X_std feature_cols window_size input_size
    
    # Calculate balanced accuracy
    balanced_acc = bal_acc(trained_params, trained_st, test_x, test_y, X_mean, X_std, feature_cols, window_size, input_size)
    
    println("Balanced Accuracy on test set: $(round(balanced_acc, digits=4))")
    return balanced_acc
end

# Example of how to use this for testing:
# Assuming you have test_x (DataFrame or Matrix) and test_y (Vector) ready:

# For window size 30:
# balanced_acc_30 = test_saved_model("lstm_model_window_30.jld2", test_x, test_y)

# For window size 90:
# balanced_acc_90 = test_saved_model("lstm_model_window_90.jld2", test_x, test_y)

# For window size 270:
# balanced_acc_270 = test_saved_model("lstm_model_window_270.jld2", test_x, test_y)