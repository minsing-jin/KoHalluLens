# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import pandas as pd
import jsonlines
from tqdm.contrib.concurrent import thread_map
import os
import argparse

from utils import exp, lm, eval_utils
import utils.generate_question as precise_qa


class PreciseQAEval:
    def __init__(self, model_path, TASKNAME, language: str = 'kor'):
        self.model_name = model_path.split("/")[-1]
        generations_file_path = f'output/{TASKNAME}/{self.model_name}/generation.jsonl'

        self.output_path = f'output/{TASKNAME}/{self.model_name}'
        self.test_df = pd.read_json(generations_file_path, lines=True)

        self.abtention_evaluator = 'meta-llama/Llama-3.1-70B-Instruct'
        self.halu_evaluator = 'meta-llama/Llama-3.1-70B-Instruct'

        if language == 'kor':
            from tasks.shortform.ko_prompt import KO_IS_HALLUCINATION_RESPONSE, KO_ABSTAIN_PROMPT_UPDATED
            self.is_hallucination_response = KO_IS_HALLUCINATION_RESPONSE
            self.abstain_prompt_updated = KO_ABSTAIN_PROMPT_UPDATED
        elif language == 'eng':
            from tasks.shortform.en_prompt import IS_HALLUCINATION_RESPONSE, ABSTAIN_PROMPT_UPDATED
            self.is_hallucination_response = IS_HALLUCINATION_RESPONSE
            self.abstain_prompt_updated = ABSTAIN_PROMPT_UPDATED
        else:
            raise ValueError(f"Invalid language: {language}")

    def eval_abstention(self, evaluator):
        print("Start abstantion evaluation")
        abs_path = f'{self.output_path}/abstain_eval_raw.jsonl'
        abstain_prompts = [
            self.abstain_prompt_updated.format(
                prompt=g.prompt, generation=g.generation
            )
            for _, g in self.test_df.iterrows()
        ]

        if os.path.exists(abs_path):
            # read from jsonl abspath
            with open(abs_path, "r") as f:
                abstains_eval_raw = [json.loads(line)["eval_res"] for line in f]
        else:
            abstains_eval_raw = thread_map(
                lambda p: lm.generate(p, evaluator),
                abstain_prompts,
                max_workers=32,
                desc=f"using {evaluator}")

            eval_utils.save_eval_raw(abstains_eval_raw, abs_path)

        ABSTAIN_JSON_KEY = 'is_abstaining'
        abstains_eval = eval_utils.jsonify_ans(raw_responses=abstains_eval_raw, \
                                               eval_prompts=abstain_prompts, \
                                               evaluator_model=evaluator, \
                                               key=ABSTAIN_JSON_KEY)
        refusal_res = []
        for o in abstains_eval:
            if ABSTAIN_JSON_KEY in o:
                refusal_res.append(o[ABSTAIN_JSON_KEY])
            else:
                refusal_res.append(False)
        self.test_df['refusal'] = refusal_res

        return refusal_res, abstains_eval_raw

    def judge_hallucination(self, evaluator):
        print("Starting Hallucination Evaluation")

        halu_prompts = [
            self.is_hallucination_response.format(
                prompt=g.prompt, generation=g.generation, gold_answer=g.answer
            ) for _, g in self.test_df.iterrows()
        ]

        if evaluator == "meta-llama/Llama-3.1-70B-Instruct":
            halu_eval_raw = thread_map(
                lambda p: lm.generate(p, evaluator),
                halu_prompts,
                max_workers=64,
                desc=f"using {evaluator}"
            )
        else:
            raise ValueError(f"Invalid evaluator: {evaluator}")

        return halu_eval_raw

    def process_res(self, abstantion_res_raw, halu_eval_raw):
        abstantion_res = [json.loads(x)['is_abstaining'] for x in abstantion_res_raw]
        halu_test_res = []
        for txt in halu_eval_raw:
            if txt.lower() not in ['correct', 'incorrect', 'unverifiable']: print(txt)
            hallucinated_judge = False if txt.lower() == 'correct' or txt.lower() == 'yes' else True
            halu_test_res.append(hallucinated_judge)
        return abstantion_res, halu_test_res

    def run_eval(self):
        abstantion_res, abstantion_raw_gen = self.eval_abstention(self.abtention_evaluator)
        halu_test_raw_gen = self.judge_hallucination(self.halu_evaluator)
        abstantion_res, halu_test_res = self.process_res(abstantion_raw_gen, halu_test_raw_gen)

        not_abstained = sum([1 for x in abstantion_res if x == False])
        if not_abstained == 0:
            hallu_rate_not_abstain = 0
        else:
            hallu_rate_not_abstain = sum([1 for is_abstaining, is_hallucinated in zip(abstantion_res, halu_test_res) \
                                          if is_abstaining == False and is_hallucinated == True]) / not_abstained
        refusal_rate = sum([1 for is_abstaining in abstantion_res if is_abstaining == True]) / len(abstantion_res)
        correct_rate = sum([1 for is_hallucinated in halu_test_res if is_hallucinated == False]) / len(halu_test_res)

        res = {
            'model': self.model_name,
            'halu_Rate': hallu_rate_not_abstain,
            'refusal_rate': refusal_rate,
            'correct_rate': correct_rate,

            'evaluator_abstantion': self.abtention_evaluator,
            'evaluator_hallucination': self.halu_evaluator,

            'abstantion': abstantion_res,
            'halu_test_res': halu_test_res,
            'abstantion_raw_generation': abstantion_raw_gen,
            'is_hallucinated_raw_generation': halu_test_raw_gen,
        }

        # save the results
        res_path = f'output/{TASKNAME}/{self.model_name}/eval_results.json'
        with open(res_path, 'w') as f:
            json.dump(res, f, indent=4)

            # Print the results
        print("=" * 80)
        print(f" Evaluation Results for: <<{self.model_name}>>")
        print("=" * 80)
        print(f"  >> Results saved to: {res_path}")
        print("-" * 80)
        print(f"  Evaluator for Abstention: {self.abtention_evaluator}")
        print(f"  Evaluator for Hallucination: {self.halu_evaluator}")
        print("-" * 80)
        print(f"  Total Number of Samples: {len(abstantion_res)}")
        print(f"  Hallucination Rate (not abstained): {hallu_rate_not_abstain:.3f} %")
        print(f"  False Refusal Rate: {refusal_rate:.3f} %")
        print(f"  Correct Rate: {correct_rate:.3f} %")
        print("-" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_generate_prompt', default=False, action='store_true')
    parser.add_argument('--do_inference', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')

    parser.add_argument('--mode', type=str, default='dynamic', help='static / dynamic')

    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-70B-Instruct',
                        help='model to use for generation')
    parser.add_argument('--inference_method', type=str, default='vllm', help='check and customize util/lm.py')
    parser.add_argument('--max_inference_tokens', type=int, default=256)
    parser.add_argument('--inf_batch_size', type=int, default=64)

    parser.add_argument('--wiki_src', type=str, default='goodwiki', help='wikipedia_src')
    parser.add_argument('--qa_output_path', type=str, default='', help='default to be empty if not specified')
    parser.add_argument('--N', type=int, default=5000)
    args = parser.parse_args()

    # get base path
    base_path = os.path.dirname(os.path.abspath(__name__))
    TASKNAME = f'precise_wikiqa_{args.wiki_src}_{args.mode}'

    model_name = args.model.split("/")[-1]
    print(f"Running {TASKNAME} with model {model_name}")

    QAs_df = None
    if args.do_generate_prompt:
        # 1. Generate QA pairs
        QA_OUTPUT_PATH = f'data/precise_qa/save/qa_{args.wiki_src}_{model_name}_{args.mode}.jsonl' \
            if args.qa_output_path == "" \
            else args.qa_output_path
        print(QA_OUTPUT_PATH)

        if os.path.exists(QA_OUTPUT_PATH):
            QAs = [line for line in jsonlines.open(QA_OUTPUT_PATH, 'r')]
            print("DATA EXISTS!! Reading from existing file ", len(QAs))
            if len(QAs) != args.N:
                print(f"Data size mismatch! N={args.N}, Data size= {len(QAs)}")
        else:
            if 'goodwiki' in args.wiki_src:
                QAs = precise_qa.precise_QA_generation_run_batch(
                    wiki_input_path=f"{base_path}/data/wiki_data/doc_goodwiki_h_score.jsonl",
                    N=args.N,
                    q_generator="meta-llama/Llama-3.1-70B-Instruct",
                    output_path=QA_OUTPUT_PATH)
                print(f"Generated {len(QAs)} QA pairs")

            else:
                raise NotImplementedError(f"mode {args.wiki_src} not implemented")

    if args.do_inference:
        QAs = [line for line in jsonlines.open(QA_OUTPUT_PATH, 'r')][:args.N]
        QAs_df = pd.DataFrame(QAs)

        print(f"Starting Inference for [{args.model}], Testset_N: {QAs_df.shape}")
        exp.run_exp(
            task=f"{TASKNAME}",
            model_path=args.model,
            all_prompts=QAs_df,
            inference_method=args.inference_method, \
            max_tokens=args.max_inference_tokens,
            max_workers=args.inf_batch_size)
        print('Inference completed')

    if args.do_eval:
        print(f"Starting Evaluation for {args.model}")
        PreciseQAEval(model_path=args.model, TASKNAME=TASKNAME).run_eval()
        print(f'{TASKNAME} Evaluation completed')
