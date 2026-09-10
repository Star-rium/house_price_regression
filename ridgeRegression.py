import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

X_train = train_df.drop(columns=['price'])
y_train = train_df['price']
X_test = test_df.drop(columns=['price'])
y_test = test_df['price']

alphas_to_test = np.logspace(-2, 4, 100)

ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=alphas_to_test, cv=5))
])

ridge_pipeline.fit(X_train, y_train)

best_alpha = ridge_pipeline.named_steps['ridge'].alpha_
y_pred_ridge = ridge_pipeline.predict(X_test)

print(f"Optimal Ridge Alpha: {best_alpha:.2f}")
print(f"Test R² Score:       {r2_score(y_test, y_pred_ridge):.4f}")
print(f"Test MAE:            {mean_absolute_error(y_test, y_pred_ridge):.2f}")
print(f"Test RMSE:           {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.2f}")
