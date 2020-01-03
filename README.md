# Fit linear regression using PYMC3

Fit simple model: 𝑌 = 𝑋𝛽+𝜖, where 𝑌 is the output we want to predict (or dependent variable), 𝑋 is our predictor (or independent variable), and 𝛽 are the coefficients (or parameters) of the model we want to estimate. 𝜖 is an error term which is assumed to be normally distributed.

In baysian lingo this is rewritten as: 𝑌 ∼ *N*(𝑋𝛽,𝜎^2^), where 𝑌 as a random variable (or random vector) of which each element (data point) is distributed according to a Normal distribution. The mean of this normal distribution is provided by our linear predictor with variance 𝜎^2^.
