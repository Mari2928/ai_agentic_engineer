import argparse
import json
import os
import time
import aiohttp
from pathlib import Path
import numpy as np
import asyncio
from evaluate import Evaluate

# for Together API models
# from together import AsyncTogether, Together
# async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

# for OpenAI models
from openai import OpenAI
from openai import AsyncOpenAI
async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

single_model = [
    "gpt-4o"
]

system_prompt_ = """You are an agent assistant, helping to apply the LLM agent to the task requested."""

async def run_llm(model, user_prompt, loop, prev_response=None):
    """Run a single LLM call with a model while accounting for previous responses + rate limits."""
    for sleep_time in [1, 2, 4]:
        try:
            messages = (
                [
                    {
                        "role": "system",
                        "content": system_prompt_,
                    },
                    {"role": "user", "content": user_prompt},
                ]
                if prev_response
                else [{"role": "user", "content": user_prompt}]
            )
            async with aiohttp.ClientSession(loop=loop) as client:
                response = await async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_completion_tokens=512,
                )
                # print("Model: ", model)
                break
        except Exception as e:
            print(e)
            await asyncio.sleep(sleep_time)
    return response

parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
args = parser.parse_args()

async def main(loop):
    num_seeds = {
        "recsys": 1
    }

    out_dir = args.out_dir
    all_results = {}
    final_infos = {}
    for dataset in num_seeds.keys():
        final_info_list = []
        for seed_offset in range(num_seeds[dataset]):
            ds = Evaluate(dataset)            

            og_t0 = time.time()
            examples = []
            for i, example in enumerate(ds.eval_set):
                """Run the main loop of the GPT process."""
                problem = ds.get_user_prompt(example)

                user_prompt = """

Bsed on the above purchase history, please respond a ranked list of items in the order that user is likely to buy next.
For example, if you respond [apple, mango, banana], apple would be the 1st item user would likely to buy next and mango is the 2nd item. 
Respond a ranked list of items in list. 
DO NOT include string format with double-quotation mark (") or any text explanation.
                                """
                final_prompt = problem + user_prompt
                
                result = await asyncio.gather(*[run_llm(model, final_prompt, loop) for model in single_model])
                output = ""
                for chunk in result:
                    out = chunk.choices[0].message.content
                    output += out

                example = ds.put_output(output, example)
                ds.single_evaluate(example, i)
                examples.append(example)
                print("Response ", i, "generated.")

            total_run_time = time.time() - og_t0#

            final_info = ds.evaluate(examples, total_run_time)
            all_results[f"{dataset}_{seed_offset}_final_info"] = final_info
            all_results[f"{dataset}_{seed_offset}_train_info"] = ds.train_log_info
            final_info_list.append(final_info)
        final_info_dict = {
            k: [d[k] for d in final_info_list] for k in final_info_list[0].keys()
        }
        means = {f"{k}_mean": np.mean(v) for k, v in final_info_dict.items()}
        stderrs = {
            f"{k}_stderr": np.std(v) / len(v) for k, v in final_info_dict.items()
        }
        final_infos[dataset] = {
            "means": means,
            "stderrs": stderrs,
            "final_info_dict": final_info_dict,
        }

    print("final_infos", final_infos)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "final_info.json"), "w+") as f:
        json.dump(final_infos, f)

    with open(os.path.join(out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)


loop = asyncio.get_event_loop()
loop.run_until_complete(main(loop))
loop.close()
