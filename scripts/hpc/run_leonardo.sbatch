#!/bin/bash
#SBATCH --job-name=europa-reasoning-run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.run.out
#SBATCH --error=logs/%j.run.err
# Submit with your account and partition, e.g. sbatch -A ACCOUNT -p PARTITION ...
# Full EUROPA Reasoning evaluation of one model in both modes on one 4-GPU node:
# vLLM serves the target (data parallel on GPUs 0-2), Ollama serves the pinned judge on GPU 3,
# and the default-mode and reasoning-mode runs execute concurrently against the same server.
#   MODEL=/path/or/hf-id REV=<immutable revision> NAME=<run name> sbatch scripts/hpc/run_leonardo.sbatch
set -euo pipefail
repo=${REPO:-$PWD}
: "${MODEL:?set MODEL}" "${REV:?set REV}" "${NAME:?set NAME}"
MODEL_ID=${MODEL_ID:-$MODEL}   # public model id recorded in the run (e.g. the HF repo id)
REASONING_TOKENS=${REASONING_TOKENS:-8192}
WORKERS=${WORKERS:-48}
: "${TOOLS:?set TOOLS to a directory with ollama-0.32.8/ and ollama-models-baseline/}"
tools=$TOOLS
: "${CONTAINER:?set CONTAINER to a Singularity image with Python >= 3.11 (and CUDA for vLLM)}"
C=$CONTAINER
BIND=${BIND:-$HOME}
: "${VLLM_PY:?set VLLM_PY to a Python with vLLM installed}"
JUDGE=gemma4:31b
JREV=sha256:6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7
vport=$((33000 + SLURM_JOB_ID % 10000)); oport=$((vport + 1))
mkdir -p "$repo/results/runs" logs

# --- judge: pinned Ollama digest on GPU 3
export OLLAMA_MODELS=$tools/ollama-models-baseline OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=8
CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=127.0.0.1:$oport $tools/ollama-0.32.8/bin/ollama serve > logs/ollama-$SLURM_JOB_ID.log 2>&1 &
until curl -sf http://127.0.0.1:$oport/api/version > /dev/null; do sleep 2; done
got=$(curl -s http://127.0.0.1:$oport/api/tags | python3 -c "import json,sys; print([m['digest'] for m in json.load(sys.stdin)['models'] if m['name']=='$JUDGE'][0])")
[[ "sha256:$got" == "$JREV" ]] || { echo "judge digest $got != $JREV" >&2; exit 2; }

# --- target: vLLM OpenAI-compatible server, data parallel over GPUs 0-2
CUDA_VISIBLE_DEVICES=0,1,2 singularity exec --nv -B "$BIND" "$C" env HF_HUB_OFFLINE=1 \
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$MODEL_ID" \
  --data-parallel-size 3 --max-model-len 16384 --gpu-memory-utilization 0.90 --dtype bfloat16 \
  --port $vport > logs/vllm-$SLURM_JOB_ID.log 2>&1 &
until curl -sf http://127.0.0.1:$vport/v1/models > /dev/null; do sleep 5; done

run() {  # $1 = mode suffix, rest = extra flags
  local mode=$1; shift
  singularity exec -B "$BIND" "$C" env PYTHONPATH="$repo/src" python -m europa_reasoning run \
    --backend openai-compatible --base-url http://127.0.0.1:$vport/v1 --model "$MODEL_ID" --revision "$REV" \
    --timeout 3600 --workers "$WORKERS" "$@" \
    --judge-backend ollama --judge-model "$JUDGE" --judge-revision "$JREV" --judge-base-url http://127.0.0.1:$oport \
    --run-id "$NAME-$mode" --output "$repo/results/runs/$NAME-$mode.json"
}
run default --no-think > logs/$SLURM_JOB_ID.default.log 2>&1 &
p1=$!
run think --think --reasoning-tokens "$REASONING_TOKENS" > logs/$SLURM_JOB_ID.think.log 2>&1 &
p2=$!
status=0; wait $p1 || status=1; wait $p2 || status=1
tail -30 logs/$SLURM_JOB_ID.default.log logs/$SLURM_JOB_ID.think.log
exit $status
