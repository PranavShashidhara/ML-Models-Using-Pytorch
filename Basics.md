Some terms which might be helpful to know beforehand:

- **Prior**
    - Your initial belief about the model parameters (weights) before seeing the data.  
    - Think of it as what you assume the weights should be before looking at the data.  
    - **Simple formula:** w ~ Normal(0, σ²)  
      (Weights are around 0 with some uncertainty.)

- **Likelihood**
    - Measures how likely your observed data is, given a set of weights.  
    - Data acts as evidence that tells you which weights make sense.  
    - **Simple formula:** y ~ Normal(X * w, σ_y²)  
      (If we use weights w, how likely is the observed data?)

- **Posterior**
    - Your updated belief about the weights after combining the prior and likelihood.  
    - It balances your initial assumption with what the data tells you.  
    - **Simple formula:** Posterior ∝ Likelihood × Prior

- **Prediction**
    - Use the posterior distribution of weights to predict outputs for new inputs.  
    - Gives both a **mean prediction** and a **measure of uncertainty**.  
    - **Simple formulas:**  
      - Mean prediction: y_hat = x_*^T * μ_w  
      - Uncertainty: Var(y_*) = x_*^T * Σ_w * x_* + σ_y²  
      (Use all plausible weights from the posterior to predict and know how confident the model is.)
