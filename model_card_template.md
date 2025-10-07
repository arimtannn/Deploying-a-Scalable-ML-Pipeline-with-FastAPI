# Model Card

---

## Model Details
**Model type:** Logistic Regression (Binary Classification)
**Library:** scikit-learn (`LogisticRegression`, solver="lbfgs")
- **Version:** 1.0
- **Author:** Ariana Tan
- **Date:** October 2025
- **Repository:** [Deploying-a-Scalable-ML-Pipeline-with-FastAPI](https://github.com/arimtannn/Deploying-a-Scalable-ML-Pipeline-with-FastAPI)

---

## Intended Use
Predict whether a person’s income exceeds $50K/year based on demographic and employment attributes from the U.S. Census Bureau dataset.
The model is designed for educational and research purposes, specifically to demonstrate the development, training, evaluation, and deployment of a machine learning pipeline using FastAPI.  
It is **not intended for production or decision-making about real individuals**.

The model can be used to:
- Experiment with MLOps principles (training, testing, CI/CD, model serving)
- Explore the impact of different categorical slices (e.g., education, occupation) on model performance
- Demonstrate bias analysis and data slicing metrics

---

## Training Data
- Trained with `LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs")`
- Loss: Log-loss optimized via LBFGS solver
- **Hyperparameters:** Default (no tuning applied)
- **Convergence warning:** The optimizer reached the iteration limit but still produced a valid solution.

---

## Evaluation Metrics
Model performance was evaluated using the following metrics on the **test set**:

| Metric | Value      |
|---------|------------|
| Precision | **0.7491** |
| Recall | **0.5599** |
| F1 Score | **0.6409** |

These results indicate that the model achieves a good balance between precision and recall, with modest overall accuracy for a baseline logistic regression.

---

## Sliced Performance
Performance was also computed on **categorical feature slices** (e.g., by `workclass`, `education`, `sex`, etc.) to identify potential bias or underperformance across subgroups.  
Metrics for each slice are stored in the file `slice_output.txt`.

---

## Ethical Considerations
- The model was trained on a dataset that reflects socioeconomic and demographic information from the 1990s.  
  These patterns may encode biases related to gender, race, and occupation.
- The dataset is not representative of current global populations.
- Predictions **should not be used for employment, financial, or social decisions.**
- The project’s purpose is **educational**, emphasizing model deployment and pipeline design.

---

## Caveats and Recommendations
- Increasing the number of iterations (`max_iter`) or scaling numerical features could improve convergence and possibly performance.
- Using a tree-based model (e.g., RandomForest or XGBoost) could better capture nonlinear relationships.
- Further bias evaluation (e.g., Fairlearn or Aequitas) is recommended if extending this project.
- Regular retraining and re-evaluation would be needed for any real-world application.

