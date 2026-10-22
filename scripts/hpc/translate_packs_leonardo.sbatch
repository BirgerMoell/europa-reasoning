#!/bin/bash
#SBATCH --job-name=europa-reasoning-translate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.translate.out
#SBATCH --error=logs/%j.translate.err
# Submit with your account and partition, e.g. sbatch -A ACCOUNT -p PARTITION ...
# Translate the authored English pack into the other 35 languages with a pinned local Ollama model:
# one server per GPU, languages split round-robin across GPUs. Resumable (valid packs are skipped).
set -euo pipefail
repo=${REPO:-$PWD}
: "${TOOLS:?set TOOLS to a directory with ollama-0.32.8/ and ollama-models-baseline/}"
tools=$TOOLS
: "${CONTAINER:?set CONTAINER to a Singularity image with Python >= 3.11 (and CUDA for vLLM)}"
C=$CONTAINER
BIND=${BIND:-$HOME}
MODEL=${MODEL:-gemma4:31b}
export OLLAMA_MODELS=$tools/ollama-models-baseline OLLAMA_KEEP_ALIVE=-1
mapfile -t langs < <(python3 -c "import json; print('\n'.join(l['code'] for l in json.load(open('$repo/src/europa_reasoning/resources/languages.json'))['languages'] if l['code'] != 'eng'))")
pids=()
for g in 0 1 2 3; do
  port=$((32000 + SLURM_JOB_ID % 10000 + g))
  CUDA_VISIBLE_DEVICES=$g OLLAMA_HOST=127.0.0.1:$port $tools/ollama-0.32.8/bin/ollama serve > logs/ollama-$SLURM_JOB_ID-$g.log 2>&1 &
  until curl -sf http://127.0.0.1:$port/api/version > /dev/null; do sleep 2; done
  digest=$(curl -s http://127.0.0.1:$port/api/tags | python3 -c "import json,sys; print([m['digest'] for m in json.load(sys.stdin)['models'] if m['name']=='$MODEL'][0])")
  only=()
  for i in "${!langs[@]}"; do (( i % 4 == g )) && only+=(--only "${langs[$i]}"); done
  singularity exec -B "$BIND" "$C" python "$repo/scripts/generate_language_packs.py" \
    --model "$MODEL" --model-revision "${digest#sha256:}" --base-url http://127.0.0.1:$port "${only[@]}" ${TRANSLATE_ARGS:-} \
    > logs/translate-$SLURM_JOB_ID-$g.log 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait $p || status=1; done
ls "$repo/src/europa_reasoning/resources/language-packs" | wc -l
exit $status
