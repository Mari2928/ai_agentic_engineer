import argparse
import json
import os
import time
import aiohttp
from pathlib import Path

import numpy as np
import asyncio
from together import AsyncTogether, Together
from evaluate import Evaluate

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

reference_models = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "microsoft/WizardLM-2-8x22B",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "deepseek-ai/DeepSeek-V3"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
]
aggregator_model = "Qwen/Qwen2.5-72B-Instruct-Turbo"
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""
layers = 3


def getFinalSystemPrompt( system_prompt, results ):
    """Construct a system prompt for layers 2+ that includes the previous responses to synthesize."""
    return (
            system_prompt
            + "\n"
            + "\n".join([f"{i + 1}. {str(element)}" for i, element in enumerate(results)])
    )

async def run_llm( model, prompt, loop, prev_response=None ):
    """Run a single LLM call with a model while accounting for previous responses + rate limits."""
    for sleep_time in [2, 4]:
        try:
            messages = (
                [
                    {
                        "role": "system",
                        "content": getFinalSystemPrompt(
                            aggreagator_system_prompt, prev_response
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
                if prev_response
                else [{"role": "user", "content": prompt}]
            )
            async with aiohttp.ClientSession(loop=loop) as client:
                response = await async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=512,
                )
                # print("Model: ", model)
                break
        except Exception as e:
            print(e)
            await asyncio.sleep(sleep_time)
    return response.choices[0].message.content


parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
args = parser.parse_args()


async def main( loop ):
    num_seeds = {
        "ml_regression": 5
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
                # generate here is a placeholder for your models generations
                """Run the main loop o the MOA process."""

                user_prompt = ds.get_user_prompt(example)

                final_prompt = user_prompt

                print("final_prompt", final_prompt)

                results = await asyncio.gather(*[run_llm(model, final_prompt, loop) for model in reference_models])

                for _ in range(1, layers - 1):
                    results = await asyncio.gather(
                        *[run_llm(model, final_prompt, loop, prev_response=results) for model in reference_models]
                    )

                finalStream = client.chat.completions.create(
                    model=aggregator_model,
                    messages=[
                        {
                            "role": "system",
                            "content": getFinalSystemPrompt(aggreagator_system_prompt, results),
                        },
                        {"role": "user", "content": final_prompt},
                    ],
                    stream=True,
                )
                output = ""
                for chunk in finalStream:
                    out = chunk.choices[0].delta.content
                    output += out
                example = ds.put_output(output, example)
                print("example", example)
                ds.single_evaluate(example, i)
                examples.append(example)
                print("Response ", i, "generated.")

            total_run_time = time.time() - og_t0

            final_info = ds.evaluate(examples, total_run_time)
            print("final_info",final_info)
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

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "final_info.json"), "w+") as f:
        json.dump(final_infos, f)

    with open(os.path.join(out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)


loop = asyncio.get_event_loop()
loop.run_until_complete(main(loop))