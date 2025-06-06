# Why Learn Julia or High Performance computing, Parallel computing for Deep Learning?

# List of few key softwares we will learn to use:
# 1. Julia
# 2. MLJ.jl
# 3. Flux.jl and Lux.jl

# ## Basic commands

# This is a very brief and rough primer if you're new to Julia and wondering how to do simple things that are relevant for data analysis.
#
# Defining a vector

x = [1, 3.1, 2, 5]
@show x
println(x)
@show lastindex(x)
typeof(x)
eltype(x)

# Operations between vectors

y = 5
z = x .+ y # elementwise operation

# Defining a matrix
x = [1 2 3 4 5]
@show x
X = [1 2; 3 4]
Y = [[1, 2, 3, 5] [6, 7, 8, 9]]

# You can also do that from a vector

X = reshape([1, 2, 3, 4], 2, 2)

# But you have to be careful that it fills the matrix by column; so if you want to get the same result as before, you will need to permute the dimensions

X_transposed = permutedims(reshape([1, 2, 3, 4], 2, 2))
println(X)
println(X_transposed)
# Function calls can be split with the `|>` operator so that the above can also be written

x_ = reshape([1, 2, 3, 4], 2, 2) |> permutedims

# You don't have to do that of course but we will sometimes use it in these tutorials.
#
# There's a wealth of functions available for simple math operations

x = 4
@show x^6
+(4, 5)
4 + 5
@show sqrt(x)
x^(1 / 2)
@show √4
α² = 4

# Element wise operations on a collection can be done with the dot syntax:
.+([4, 6, 9], 5)
[4, 6, 9] .+ 5

function add(x, y)
    z = x + y
    g = z^2

end

# The packages `Statistics` (from the standard library) and [`StatsBase`](https://github.com/JuliaStats/StatsBase.jl) offer a number of useful function for stats:

using Statistics: mean, std
import StatsBase as SB
using Statistics
using Pkg

# Note that if you don't have `StatsBase`, you can add it using `using Pkg; Pkg.add("StatsBase")`.
# Right, let's now compute some simple statistics:
using Random
using Distributions
rng = Xoshiro(1234) #Xoshiro is state of the art Random Number Generator
rand() #seed for reporducibility
rand(rng, 10, 2, 4) # 1 point from a N(0, 1)
x = rand(rng, Gumbel(0, 1), 1_000_000) # 1_000 points iid from a N(0, 1)
# x = randn(rng, 1_000_000) # 1_000 points iid from a N(0, 1)
μ = mean(x)
σ = std(x)
using Plots
histogram(x)
@show (μ, σ)

# Indexing data starts at 1, use `:` to indicate the full range

X = [1 2; 3 4; 5 6]
@show X[1, 2]
@show X[:, 2]
@show X[1, :]
@show X[[1, 2], [1, 2]]
@show X[1:2, 1:2]
@show X[1:2, 1:end]
@show X[1:2, :]

# `size` gives dimensions (nrows, ncolumns)

size(X)

# ## Loading data

#
# There are many ways to load data in Julia, one convenient one is via the [`CSV`](https://github.com/JuliaData/CSV.jl) package.

using CSV

# Many datasets are available via the [`RDatasets`](https://github.com/JuliaStats/RDatasets.jl) package
# Let's load some data from RDatasets (the full list of datasets is available [here](http://vincentarelbundock.github.io/Rdatasets/datasets.html)).

using RDatasets

# And finally the [`DataFrames`](https://github.com/JuliaData/DataFrames.jl) package allows to manipulate data easily

using DataFrames

auto = dataset("ISLR", "Auto")
auto = Matrix(auto_df)
# To get dimensions you can use `size` and `nrow` and `ncol`

@show size(auto)
@show nrow(auto)
@show ncol(auto)

@show first(auto_df, 3)

# The `describe` function allows to get an idea for the data:

auto_df |> describe |> show

# To retrieve column names, you can use `names`:

@show auto_df |> names

# Accesssing columns can be done in different ways:

mpg = auto.MPG
mpg = auto[:, 1]
mpg = auto[:, :MPG]

mean(mpg)
std(mpg)
@show StatsBase.summarystats(mpg)

first_100_sampled_mpg = mpg[1:100]

mean(first_100_sampled_mpg)
std(first_100_sampled_mpg)

random_sampled_mpg = sample(mpg, 50, replace=false)
mean(random_sampled_mpg)
std(random_sampled_mpg)

50 / 392


# For more detailed tutorials on basic data wrangling in Julia, consider
#
# * the [learn x in y](https://learnxinyminutes.com/docs/julia/) julia tutorial
# * the [`DataFrames.jl` docs](http://juliadata.github.io/DataFrames.jl/latest/)
# * the [`StatsBases.jl` docs](https://juliastats.org/StatsBase.jl/latest/)



# ## Plotting data

# There are multiple libraries that can be used to  plot things in Julia:
#
# * [Plots.jl](https://github.com/JuliaPlots/Plots.jl) which supports multiple plotting backends,
# * [Gadfly.jl](https://github.com/GiovineItalia/Gadfly.jl) influenced by the grammar of graphics and `ggplot2`
# * [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl) basically matplotlib
# * [PGFPlotsX.jl](https://github.com/KristofferC/PGFPlotsX.jl) and [PGFPlots](https://github.com/JuliaTeX/PGFPlots.jl) using the LaTeX package  pgfplots,
# * [Makie](https://github.com/JuliaPlots/Makie.jl), [Gaston](https://github.com/mbaz/Gaston.jl), [Vega](https://github.com/queryverse/VegaLite.jl), ...
#
# In these tutorials we use `Plots.jl` but you could use another package of course.

using Plots

plt_hist = histogram(mpg, size=(800, 600), linewidth=1, legend=true)
plt_hist
scatter(mpg, size=(800, 600), linewidth=2, legend=false)


### Other Resources to learn
# 1. https://www.youtube.com/@statquest - For deep learning and Machine learning
# 2. https://www.youtube.com/@doggodotjl - For Julia and ML
# 3. https://www.youtube.com/@3blue1brown - For Intuitive explanations of topics on ML and other similar topics