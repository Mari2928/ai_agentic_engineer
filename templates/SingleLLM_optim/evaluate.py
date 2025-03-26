import sys
import json
import os
import ast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

sys.path.append('./data/human-eval')
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness


class Evaluate():
    """Evaluate LLM responses for a given dataset."""

    def __init__( self, dataset ):
        self.dataset = dataset
        self.train_log_info = []
        self.mses = []
        self.eval_set = self.read_ml_model()
        open('ml_results.txt', 'w').close()
        self.code_start = """
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a bias term (intercept) to the features
X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Optimize weights (manual optimization)
# weights = [bias, coefficient]
            """
        self.code_end = """
def predict(X, weights):
    return X.dot(weights)

# Calculate predictions using the custom weights
y_pred_manual = predict(X_test_bias, manual_weights)

# Calculate Mean Squared Error for the manually set weights
mse_manual = mean_squared_error(y_test, y_pred_manual)

if not os.path.exists("ml_results.txt"):
    open('ml_results.txt', 'w').close()
with open('ml_results.txt', 'a') as f:
    result = "Weights: " + str(manual_weights) + " ; MSE: " + str(mse_manual)\n 
    json.dump(result, f)
    f.write('\\n')

self.mses.append(mse_manual)
print(f"Manual Weights: {manual_weights}")
print(f"Manual Regression MSE: {mse_manual:.2f}")    
"""
        self.optimize_instruction_form = """ 
You are an AI optimizer who is tuning weights for a machine learning regression model in Python, aiming to minimize MSE.
If you can get MSE around 400.00 that would be great.
Be efficient and cautious in your decision.
            """
        X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
        _, X_test, _, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    def read_ml_model( self ):
        ml_model = """
# Optimize weights (manual optimization)
# weights = [bias, coefficient]
manual_weights = [0, 0]  # You can adjust these values to experiment
"""
        return [ml_model for _ in range(30)]

    def get_user_prompt( self, example ):
        if os.path.exists("ml_results.txt"):
            fewshot_prompt = """
Below are some sample results from previous runs. 
             """
            with open("ml_results.txt", 'r') as file:
                for result in file:
                    fewshot_prompt += f"""

                    Result:

                    ```
                    {result}
                    ```
                    """
            base_prompt = self.optimize_instruction_form + fewshot_prompt
        else:
            base_prompt = self.optimize_instruction_form

        base_prompt += f"""
Here is the weights of the machine learning regression model you are asked to tune:
        ```
        {self.eval_set[0]}
        ```

Change the above manual_weights only and try to optimize the MSE as smaller as possible based on the performance in the previous runs. 
Do not use the similar weights to the previous runs.
Respond the weights in a list format like following:

[0, 0]

DO NOT include a python code block with three backticks (```) or any text explanation.
        """
        return base_prompt

    def put_output( self, output, example ):
        return output

    def single_evaluate( self, example, iter_num ):
        code_mid = f"""
manual_weights = {example}
         """
        script = str(self.code_start) + code_mid + str(self.code_end)
        try:
            exec(script)
        except:
            print("Execution error:", script)
            if not self.mses:
                self.mses.append(2213.0)  # put initial MSE
        self.train_log_info.append(
            {
                "iter": iter_num,
                "mse": self.mses[-1],
            }
        )

    def evaluate( self, examples, total_run_time ):
        final_info = {
            "mse": self.mses[-1],
            "total_run_time": total_run_time,
        }
        return final_info
