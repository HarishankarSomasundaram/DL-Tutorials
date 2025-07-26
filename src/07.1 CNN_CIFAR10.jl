# CIFAR-10 CNN Analysis with LeNet Variants 

using Flux, MLDatasets, JLD2, Plots, StatsPlots
using Statistics, Random 
 

folder = "cifar10"  # sub-directory in which to save
isdir(folder) || mkdir(folder)

println("Loading CIFAR-10 dataset...")

# Load CIFAR-10 dataset
train_data = MLDatasets.CIFAR10(split=:train)
test_data = MLDatasets.CIFAR10(split=:test)

# CIFAR-10 has 32x32x3 images (RGB) with 10 classes
# Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

function prepare_data(dataset; batchsize::Int = 512)
    # Convert images to Float32 and normalize to [0,1]
    x = Float32.(dataset.features) ./ 255.0f0
    
    # Check the dimensions and adjust accordingly
    println("Original data shape: ", size(x)) 
    # Permute dimensions to match Flux convention: (H, W, C, N)
    # If it's in (C, H, W, N) format, we need to permute
    if size(x, 1) == 3  # If first dimension is channels
        x = permutedims(x, (2, 3, 1, 4))  # From (C, H, W, N) to (H, W, C, N)
        println("Permuted to: ", size(x))
    end
    
    # One-hot encode labels (CIFAR-10 classes are 0-9)
    y = Flux.onehotbatch(dataset.targets, 0:9) 
    return Flux.DataLoader((x, y); batchsize, shuffle=true)
end
train_loader_full = prepare_data(train_data, batchsize=512)
test_loader = prepare_data(test_data, batchsize=1000)

println("Data loaded successfully!")
println("Training samples: ", length(train_data.targets))
println("Test samples: ", length(test_data.targets))


# Models

# LeNet5 Original (32x32x3 → 10 classes)
function LeNet5_original()
    return Chain(
        Conv((5, 5), 3 => 6, relu),   # 32x32x3 → 28x28x6
        MaxPool((2, 2)),              # 28x28x6 → 14x14x6
        Conv((5, 5), 6 => 16, relu),  # 14x14x6 → 10x10x16
        MaxPool((2, 2)),              # 10x10x16 → 5x5x16
        Flux.flatten,                 # 5x5x16 → 400
        Dense(400 => 120, relu),
        Dense(120 => 84, relu),
        Dense(84 => 10)               # 10 classes for CIFAR-10
    )
end

# LeNet3 with (3,3) filters
function LeNet3()
    return Chain(
        Conv((3, 3), 3 => 6, relu),   # 32x32x3 → 30x30x6
        MaxPool((2, 2)),              # 30x30x6 → 15x15x6
        Conv((3, 3), 6 => 16, relu),  # 15x15x6 → 13x13x16
        MaxPool((2, 2)),              # 13x13x16 → 6x6x16 (floor division)
        Flux.flatten,                 # 6x6x16 → 576
        Dense(576 => 120, relu),
        Dense(120 => 84, relu),
        Dense(84 => 10)
    )
end

# LeNet7 with (7,7) filters
function LeNet7()
    return Chain(
        Conv((7, 7), 3 => 6, relu),   # 32x32x3 → 26x26x6
        MaxPool((2, 2)),              # 26x26x6 → 13x13x6
        Conv((7, 7), 6 => 16, relu),  # 13x13x6 → 7x7x16
        MaxPool((2, 2)),              # 7x7x16 → 3x3x16 (floor division)
        Flux.flatten,                 # 3x3x16 → 144
        Dense(144 => 120, relu),
        Dense(120 => 84, relu),
        Dense(84 => 10)
    )
end




# Metrics

function loss_and_accuracy(model, data_loader)
    total_loss = 0.0f0
    total_correct = 0
    total_samples = 0
    
    for (x, y) in data_loader
        ŷ = model(x)
        total_loss += Flux.logitcrossentropy(ŷ, y) * size(x, 4)
        total_correct += sum(Flux.onecold(ŷ, 0:9) .== Flux.onecold(y, 0:9))
        total_samples += size(x, 4)
    end
    
    avg_loss = total_loss / total_samples
    accuracy = round(100 * total_correct / total_samples; digits=2)
    
    return avg_loss, accuracy
end

# Training

function train_model(model, train_loader, test_loader, epochs; lr=0.001, lambda=1e-4)
    println("Training model for $epochs epochs...")
    
    # Setup optimizer
    opt_rule = AdamW(lr, (0.9, 0.999), lambda)
    opt_state = Flux.setup(opt_rule, model)
    
    train_log = []
    
    for epoch in 1:epochs
        epoch_start = time()
        
        # Training step
        for (x, y) in train_loader
            grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), model)
            Flux.update!(opt_state, model, grads[1])
        end
        
        # Evaluate
        train_loss, train_acc = loss_and_accuracy(model, train_loader)
        test_loss, test_acc = loss_and_accuracy(model, test_loader)
        
        epoch_time = time() - epoch_start
        
        println("Epoch $epoch: Train Acc: $(train_acc)%, Test Acc: $(test_acc)%, Time: $(round(epoch_time, digits=2))s")
        
        push!(train_log, (epoch=epoch, train_loss=train_loss, train_acc=train_acc, 
                         test_loss=test_loss, test_acc=test_acc))
    end
    return train_log
end

# Exp 1: Effect of Dataset Size vs Training Steps 

# Creating subsets of training data
function create_subset_loader(n_samples, batchsize=512)
    # Get indices for subset
    indices = randperm(length(train_data.targets))[1:n_samples]
    
    # Extract subset
    x_subset = train_data.features[:, :, :, indices]
    y_subset = train_data.targets[indices]
    
    # Convert to proper format
    x_subset = Float32.(x_subset) ./ 255.0f0
    if size(x_subset, 1) == 3  # If first dimension is channels
        x_subset = permutedims(x_subset, (2, 3, 1, 4))  # From (C, H, W, N) to (H, W, C, N)
    end 
    y_subset = Flux.onehotbatch(y_subset, 0:9)
    
    return Flux.DataLoader((x_subset, y_subset); batchsize, shuffle=true)
end

# Experiment configurations
configs = [
    (samples=10000, epochs=6),
    (samples=20000, epochs=3),
    (samples=30000, epochs=2)
]

dataset_results = []

for config in configs
    println("\nTraining on $(config.samples) samples for $(config.epochs) epochs...")
    
    # Creating model and subset loader
    model = LeNet5_original()
    subset_loader = create_subset_loader(config.samples)
    
    # Train
    log = train_model(model, subset_loader, test_loader, config.epochs)
    final_test_acc = log[end].test_acc
    
    push!(dataset_results, (samples=config.samples, epochs=config.epochs, 
                           test_acc=final_test_acc, steps=config.samples÷512 * config.epochs))
    
    println("Final test accuracy: $(final_test_acc)%")
end

# Exp 2: Effect of Filter Size

# Training models with different filter sizes
filter_results = []
models_dict = Dict("LeNet3" => LeNet3(), "LeNet5" => LeNet5_original(), "LeNet7" => LeNet7())

for (name, model_fn) in [("LeNet3", LeNet3), ("LeNet5", LeNet5_original), ("LeNet7", LeNet7)]
    println("\nTraining $name...")
    
    model = model_fn()
    
    # Training for 5 epochs with full dataset
    log = train_model(model, train_loader_full, test_loader, 5)
    final_test_acc = log[end].test_acc
    
    push!(filter_results, (model=name, test_acc=final_test_acc))
    models_dict[name] = model  # Store trained model
    
    println("$name final test accuracy: $(final_test_acc)%")
end

# plotting Results 

# Plot 1: Dataset size effect
p1 = plot(
    [r.samples for r in dataset_results],
    [r.test_acc for r in dataset_results],
    marker=:circle,
    markersize=8,
    linewidth=2,
    title="Effect of Dataset Size on Test Accuracy",
    xlabel="Number of Training Samples",
    ylabel="Test Accuracy (%)",
    legend=false,
    grid=true
)

# Adding annotations for epochs
for r in dataset_results
    annotate!(p1, r.samples, r.test_acc + 1, text("$(r.epochs) epochs", 8))
end

# Plot 2: Filter size effect
filter_names = [r.model for r in filter_results]
filter_accs = [r.test_acc for r in filter_results]

p2 = bar(
    filter_names,
    filter_accs,
    title="Effect of Filter Size on Test Accuracy",
    xlabel="Model Architecture",
    ylabel="Test Accuracy (%)",
    legend=false,
    color=[:blue, :green, :red],
    grid=true
)

# Adding value labels on bars
for (i, acc) in enumerate(filter_accs)
    annotate!(p2, i, acc + 0.5, text("$(acc)%", 8))
end

# Display plots
display(p1)
display(p2)

# Exp 3: Feature Visualization

# Function to extract and visualize features
function visualize_conv_features(model, sample_idx=1)
    # Getting a sample from test data
    test_x, test_y = first(test_loader)
    sample = test_x[:, :, :, sample_idx:sample_idx]
    
    println("Visualizing features for sample $sample_idx")
    println("True label: ", Flux.onecold(test_y[:, sample_idx:sample_idx], 0:9)[1])
    
    # Original image
    orig_img = sample[:, :, :, 1]
    
    # Apply layers progressively
    after_conv1 = model[1](sample)  # First conv layer
    after_pool1 = model[2](after_conv1)  # First pooling
    after_conv2 = model[3](after_pool1)  # Second conv layer
    after_pool2 = model[4](after_conv2)  # Second pooling
    
    return orig_img, after_conv1, after_pool1, after_conv2, after_pool2
end

# Visualize features for LeNet3
if haskey(models_dict, "LeNet3")
    model_lenet3 = models_dict["LeNet3"]
    
    # Create visualization for 3 samples
    for sample_idx in 1:3
        println("\n--- Sample $sample_idx ---")
        orig, conv1, pool1, conv2, pool2 = visualize_conv_features(model_lenet3, sample_idx)
        
        # Creating subplots for different stages
        p_stages = plot(layout=(2, 3), size=(1200, 800))
        
        # Original image
        heatmap!(p_stages[1], orig[:, :, 1], c=:grays, title="Original (R)", subplot=1)
        heatmap!(p_stages[2], orig[:, :, 2], c=:grays, title="Original (G)", subplot=2)
        heatmap!(p_stages[3], orig[:, :, 3], c=:grays, title="Original (B)", subplot=3)
        
        # Showing first few feature maps from conv1
        heatmap!(p_stages[4], conv1[:, :, 1, 1], c=:viridis, title="Conv1 - Filter 1", subplot=4)
        heatmap!(p_stages[5], conv1[:, :, 2, 1], c=:viridis, title="Conv1 - Filter 2", subplot=5)
        heatmap!(p_stages[6], conv1[:, :, 3, 1], c=:viridis, title="Conv1 - Filter 3", subplot=6)
        
        plot!(p_stages, suptitle="Sample $sample_idx - Feature Maps")
        display(p_stages)
        
        # Additional plot for conv2 features
        p_conv2 = plot(layout=(2, 3), size=(1200, 600))
        for i in 1:6
            heatmap!(p_conv2[i], conv2[:, :, i, 1], c=:plasma, 
                    title="Conv2 - Filter $i", subplot=i)
        end
        plot!(p_conv2, suptitle="Sample $sample_idx - Conv2 Feature Maps")
        display(p_conv2)
    end
end

# Results Summary 

println("\n1. Dataset Size Effect:")
for r in dataset_results
    println("  $(r.samples) samples, $(r.epochs) epochs: $(r.test_acc)% test accuracy")
end

println("\n2. Filter Size Effect:")
for r in filter_results
    println("  $(r.model): $(r.test_acc)% test accuracy")
end