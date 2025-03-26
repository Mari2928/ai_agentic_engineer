import ast

import numpy as np
from collections import defaultdict
import random


class Evaluate():
    """Evaluate LLM responses for a given dataset."""

    def __init__( self, dataset ):
        self.dataset = dataset
        self.train_log_info = []
        self.true_next_item = None
        self.NDCGs = []
        self.HTs = []
        self.eval_set = self.read_recsys_problems()
        self.all_items = self.read_all_items('ml-1m.txt')
        self.recsys_instruction_form = """ 
You are an AI recommender who is recommending a list of items that user would buy next based on his previous purchase history, aiming to maximize NDCG@10.
Be efficient and cautious in your decision.
            """

    def read_user_items( self, filename, num ):
        user_items = defaultdict(list)
        with open(filename, 'r') as f:
            for line in f:
                user_id, item_id = map(int, line.strip().split())
                user_items[user_id].append(item_id)
        return [user_items[user_id] for user_id in sorted(user_items.keys())][:num]

    def read_recsys_problems( self ):
        filename = "ml-1m.txt"
        result = self.read_user_items(filename, 100)
        return result

    def read_all_items( self, filename ):
        """Return all the unique item IDs."""
        items = set()
        with open(filename, 'r') as f:
            for line in f:
                # Split each line and take only the item ID (second number)
                _, item_id = map(int, line.strip().split())
                items.add(item_id)
        return list(items)

    def get_random_items( self, item ):
        """Returns 99 random items + 1 true next item = 100 shuffled items."""
        self.true_next_item = item
        random_items = random.sample(self.all_items, 99)
        random_items.append(item)
        shuffled_items = random_items.copy()
        random.shuffle(shuffled_items)
        return shuffled_items

    def get_user_prompt( self, example ):
        random_items = self.get_random_items(example[-1])
        base_prompt = self.recsys_instruction_form

        base_prompt += f"""
Here is the user's purchase history where the last item is the one he bought most recently:
            ```
        {example[:-1]}

Here is the list of random items.

        {random_items}

            ```"""
        return base_prompt

    def put_output( self, output, example ):
        return output

    def single_evaluate( self, example, iter_num ):
        ndcg = 0.0
        ht = 0.0
        try:
            example = ast.literal_eval(example)
            if self.true_next_item in example:
                rank = example.index(self.true_next_item)
                print("rank", rank)
                if rank < 10:
                    ndcg = 1 / np.log2(rank + 2)
                    ht = 1
        except:
            print("Invalid output.")
        print("ndcg_10", ndcg, "ht_10", ht)
        self.NDCGs.append(ndcg)
        self.HTs.append(ht)
        self.train_log_info.append(
            {
                "iter": iter_num,
                "ndcg_10": ndcg,
                "ht_10": ht,
            }
        )

    def evaluate( self, examples, total_run_time ):
        num_user = len(examples)
        final_info = {
            "ndcg_10_ave": sum(self.NDCGs) / num_user,
            "ht_10_ave": sum(self.HTs)/ num_user,
            "total_run_time": total_run_time,
        }
        return final_info
