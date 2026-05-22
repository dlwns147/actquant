#!/usr/bin/env python3
"""
Llama-3.1-8B-Instruct 모델의 lm_eval GSM8K Benchmark 실행 스크립트.

사용법:
  python gsm8k.py [--model MODEL_PATH] [--batch_size N] [--limit N]
  python gsm8k.py --log_samples --output_path ./results  # 모델 출력 결과도 저장
"""

import argparse
import subprocess
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="모델 경로 또는 HuggingFace 모델 ID",
    )
    parser.add_argument("--batch_size", type=str, default="auto", help="배치 크기 (auto 또는 정수)")
    parser.add_argument("--limit", type=float, default=None, help="평가 샘플 수 제한")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_path", type=str, default=None, help="결과 저장 경로")
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="모델 출력 결과(생성 텍스트)를 per-sample 단위로 저장. --output_path 필요",
    )
    parser.add_argument(
        "--task",
        type=str,
        default='gsm8k_cot_train'
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="lm-eval TaskManager가 스캔할 커스텀 task yaml 디렉토리 "
             "(예: ifeval_pp / apps 사용 시 필요)",
    )
    args = parser.parse_args()

    if args.log_samples and not args.output_path:
        parser.error("--log_samples 사용 시 --output_path를 지정해야 합니다.")

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", f"pretrained={args.model}",
        "--tasks", args.task,
        "--batch_size", args.batch_size,
        "--device", args.device,
        "--apply_chat_template"
    ]
    if args.include_path:
        cmd.extend(["--include_path", args.include_path])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.task in ['gpqa_main_cot_n_shot', 'gpqa_main_cot_n_shot,gpqa_main_cot_n_shot']:
        ## GPQA는 양자화 모델이랑 성능차이 크게 x
        cmd.extend(["--num_fewshot", "3"])
    if args.task in ['hendrycks_math', 'hendrycks_math,hendrycks_math']:
        cmd.extend(["--num_fewshot", "4"])
    if args.output_path:
        cmd.extend(["--output_path", args.output_path])
    if args.log_samples:
        cmd.append("--log_samples")

    # 모델이 생성한 코드를 직접 실행하는 task들은 unsafe code 실행 확인 플래그 필요
    unsafe_code_tasks = {
        'humaneval', 'mbpp', 'mbpp_plus',
        'apps', 'apps_introductory', 'apps_interview', 'apps_competition',
    }
    task_set = {t.strip() for t in args.task.split(',') if t.strip()}
    if task_set & unsafe_code_tasks:
        cmd.append("--confirm_run_unsafe_code")

    print(cmd)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
