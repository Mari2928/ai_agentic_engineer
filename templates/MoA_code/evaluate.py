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
        if self.dataset == "human_eval":
            self.eval_set = read_problems()
        elif self.dataset == "ml_regression":
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
        if self.dataset == "human_eval":
            return self.eval_set[example]["prompt"]
        elif self.dataset == "ml_regression":
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
Here is the script of the machine learning model you are asked to tune its weights:
            ```
            {self.eval_set}

Change the manual_weights in the above script and try to optimize the MSE as smaller as possible based on the performance in the previous runs. 
Do not use the similar weights to the previous runs.
Respond the weights in a list format like following:

[0, 0]
 
DO NOT include a python code block with three backticks (```) or any text explanation.
            ```"""
            return base_prompt

    def put_output( self, output, example ):
        if self.dataset == "human_eval":
            return dict(task_id=example, completion=output)
        elif self.dataset == "ml_regression":
            return output

    def get_output_file( self ):
        if self.dataset == "human_eval":
            return 'outputs_human_eval.jsonl'

    def single_evaluate( self, example, iter_num ):
        if self.dataset == "human_eval":
            self.train_log_info.append(
                {
                    "iter": iter_num
                }
            )
        elif self.dataset == "ml_regression":
            code_mid = f"""
manual_weights = {example}
             """
            script = str(self.code_start) + code_mid + str(self.code_end)
            try:
                exec(script)
            except:
                print("Execution error:", script)
                if not self.mses:
                    self.mses.append(2213.0)
            self.train_log_info.append(
                {
                    "iter": iter_num,
                    "mse": self.mses[-1],
                }
            )

    def evaluate( self, examples, total_run_time ):
        if self.dataset == "human_eval":
            open('problems_human_eval.jsonl', 'w').close()
            with open('problems_human_eval.jsonl', 'a') as f:
                for i, task_id in enumerate(self.eval_set):
                    json.dump(self.eval_set[task_id], f)
                    f.write('\n')

            # # write solutions to an output file
            output_file = self.get_output_file()
            open(output_file, 'w').close()
            with open(output_file, 'a') as f:
                for example in examples:
                    json.dump(example, f)
                    f.write('\n')

            pass_at_k, total_exec_time = evaluate_functional_correctness(sample_file=output_file,
                                                                         problem_file='problems_human_eval.jsonl')
            final_info = {
                "pass_at_k": pass_at_k['pass@1'],
                "total_exec_time": total_exec_time,
                "total_run_time": total_run_time,
            }
        elif self.dataset == "ml_regression":
            final_info = {
                "mse": self.mses[-1],
                "total_run_time": total_run_time,
            }
        return final_info
