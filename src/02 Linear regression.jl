
# ## Simple linear regression

# `MLJ` essentially serves as a unified path to many existing Julia packages each of which provides their own functionalities and models, with their own conventions.
#
# The simple linear regression demonstrates this.
# Several packages offer it (beyond just using the backslash operator): here we will use `MLJLinearModels` but we could also have used `GLM`, `ScikitLearn` etc.
#
# To load the model from a given package use `@load ModelName pkg=PackageName`

using MLJ
models()

filter(model) = model.is_pure_julia && model.is_supervised && model.prediction_type == :probabilistic
models(filter)
models("XGB")
measures("F1")

mdls = models(matching(X, y))

y
# Linear regression
Pkg.add("MLJLinearModels")
LR = @load LinearRegressor pkg = MLJLinearModels

# Note: in order to be able to load this, you **must** have the relevant package in your environment, if you don't, you can always add it (``using Pkg; Pkg.add("MLJLinearModels")``).
#
# Let's load the _boston_ data set

import RDatasets: dataset
import DataFrames: describe, select, Not, rename!
data = dataset("MASS", "Boston")
println(first(data, 3))

# Let's get a feel for the data

@show describe(data)

# So there's no missing value and most variables are encoded as floating point numbers.
# In MLJ it's important to specify the interpretation of the features (should it be considered as a Continuous feature, as a Count, ...?), see also [this tutorial section](/getting-started/choosing-a-model/#data_and_its_interpretation) on scientific types.
#
# Here we will just interpret the integer features as continuous as we will just use a basic linear regression:

data = coerce(data, autotype(data, :discrete_to_continuous))
data
# Let's also extract the target variable (`MedV`):

y = data.MedV
X = select(data, Not(:MedV))

# Let's declare a simple multivariate linear regression model:

model = LR()

# First let's do a very simple univariate regression, in order to fit it on the data, we need to wrap it in a _machine_ which, in MLJ, is the composition of a model and data to apply the model on:

X_uni = select(X, :LStat) # only a single feature
mach_uni = machine(model, X_uni, y)
fit!(mach_uni)

# You can then retrieve the  fitted parameters using `fitted_params`:

fp = fitted_params(mach_uni)
@show fp.coefs
@show fp.intercept

# You can also visualise this

using Plots

plot(X.LStat, y, seriestype=:scatter, markershape=:circle, legend=false, size=(800, 600))

#  MLJ.predict(mach_uni, Xnew) to predict from a fitted model
Xnew = (LStat=collect(range(extrema(X.LStat)..., length=100)),)
plot!(Xnew.LStat, MLJ.predict(mach_uni, Xnew), linewidth=3, color=:orange)


# The  multivariate linear regression case is very similar

mach = machine(model, X, y)
fit!(mach)

fp = fitted_params(mach)
coefs = fp.coefs
intercept = fp.intercept
for (name, val) in coefs
    println("$(rpad(name, 8)):  $(round(val, sigdigits=3))")
end
println("Intercept: $(round(intercept, sigdigits=3))")

# You can use the `machine` in order to _predict_ values as well and, for instance, compute the root mean squared error:

ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)

# Let's see what the residuals look like

res = ŷ .- y
plot(res, line=:stem, linewidth=1, marker=:circle, legend=false, size=((800, 600)))
hline!([0], linewidth=2, color=:red)    # add a horizontal line at x=0
mean(y)

# Maybe that a histogram is more appropriate here

histogram(res, normalize=true, size=(800, 600), label="residual")


# ## Interaction and transformation
#
# Let's say we want to also consider an interaction term of `lstat` and `age` taken together.
# To do this, just create a new dataframe with an additional column corresponding to the interaction term:

X2 = hcat(X, X.LStat .* X.Age)

# So here we have a DataFrame with one extra column corresponding to the elementwise products between `:LStat` and `Age`.
# DataFrame gives this a default name (`:x1`) which we can change:

rename!(X2, :x1 => :interaction)

# Ok cool, now let's try the linear regression again

mach = machine(model, X2, y)
fit!(mach)
ŷ = MLJ.predict(mach, X2)
round(rsquared(ŷ, y), sigdigits=4)

# We get slightly better results but nothing spectacular.
#
# Let's get back to the lab where they consider regressing the target variable on `lstat` and `lstat^2`; again, it's essentially a case of defining the right DataFrame:

X3 = DataFrame(hcat(X.LStat, X.LStat .^ 2), [:LStat, :LStat2])
mach = machine(model, X3, y)
fit!(mach)
ŷ = MLJ.predict(mach, X3)
round(rsquared(ŷ, y), sigdigits=4)

# fitting y=mx+c to X3 is the same as fitting y=mx2+c to X3.LStat => Polynomial regression

# which again, we can visualise:

Xnew = (LStat=Xnew.LStat, LStat2=Xnew.LStat .^ 2)

plot(X.LStat, y, seriestype=:scatter, markershape=:circle, legend=false, size=(800, 600))
plot!(Xnew.LStat, MLJ.predict(mach, Xnew), linewidth=3, color=:orange)



# TODO HW : Find the best model by feature selection; best model means highest R²

import Pkg
Pkg.add("Combinatorics")
Pkg.activate(".") 
Pkg.status() 
using MLJ, MLJLinearModels, RDatasets, DataFrames, Statistics, Combinatorics


y = data.MedV
X = select(data, Not(:MedV))
model = LR()
names(X)
feature_names = names(X)
best_r2 = 0
best_features = []
best_model = nothing

combinations(feature_names,1)

for k in 1:length(feature_names)
    println(k)
    for subset in combinations(feature_names, k)
        # println(subset)
        X_sub = select(X, subset)
        mach = machine(model, X_sub, y)
        fit!(mach)
        ŷ = MLJ.predict(mach, X_sub)
        r2 = rsquared(ŷ, y)
        if r2 > best_r2
            best_r2 = r2
            best_features = subset
            best_model = mach
        end
    end
end

println("Best R²: ", best_r2)
println("Best feature subset: ", best_features)

# Best R²: 0.7406426641094094
# Best feature subset: ["Crim", "Zn", "Indus", "Chas", "NOx", "Rm", "Age", "Dis", "Rad", "Tax", "PTRatio", "Black", "LStat"]
 
