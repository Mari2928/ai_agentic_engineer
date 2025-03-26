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
        self.dataset == "human_eval"
        self.eval_set = read_problems()

    def get_user_prompt( self, example ):
        return self.eval_set[example]["prompt"]

    def put_output( self, output, example ):
        return dict(task_id=example, completion=output)

    def get_output_file( self ):
        return 'outputs_human_eval.jsonl'

    def single_evaluate( self, example, iter_num ):
        self.train_log_info.append(
            {
                "iter": iter_num
            }
        )

    def evaluate( self, examples, total_run_time ):
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
        return final_info
