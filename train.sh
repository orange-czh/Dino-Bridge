#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <serial|parallel> <train_ssdg_polyp.py|train_dg_polyp.py> polyp <model> <gpu_ids> <ratio> <adam|adamw|sgd> [extra training arguments]"
}
if [[ $# -eq 1 && ( $1 == --help || $1 == -h ) ]]; then
    usage
    exit 0
fi
if [[ $# -lt 7 ]]; then
    usage >&2
    exit 2
fi

mode=$1
entry=$2
dataset=$3
model=$4
gpu_ids=$5
ratio=$6
optimizer=$7
shift 7

case "$mode" in serial|parallel) ;; *) usage >&2; exit 2 ;; esac
case "$entry" in train_ssdg_polyp.py|train_dg_polyp.py) ;; *) usage >&2; exit 2 ;; esac
[[ $dataset == polyp ]] || { echo "Only polyp is included." >&2; exit 2; }
case "$optimizer" in adam|adamw|sgd) ;; *) usage >&2; exit 2 ;; esac
[[ $gpu_ids =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "GPU IDs must be comma-separated integers." >&2; exit 2; }
if [[ $mode == serial && $gpu_ids == *,* ]]; then
    echo "Serial mode takes one GPU ID." >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"
seed=${SEED:-520}
python_bin=${PYTHON_BIN:-python}
method=${entry%.py}
output_root=${OUTPUT_ROOT:-outputs-$seed}
save_dir="$output_root/polyp/$ratio/$model/$method-$optimizer"
mkdir -p "$save_dir"

IFS=',' read -r -a devices <<< "$gpu_ids"
domain_names=(A B C D E)
pids=()
for domain in 0 1 2 3 4; do
    domain_name=${domain_names[$domain]}
    device=${devices[$((domain % ${#devices[@]}))]}
    command=("$python_bin" -u "$entry"
        --config configs/dataset/polyp.yaml
        --shared-config configs/shared.yaml
        --train-config "configs/$optimizer.yaml"
        --save-path "$save_dir/$domain_name"
        --model "$model" --seed "$seed" --domain "$domain" --ratio "$ratio"
        "$@")
    echo "Domain $domain_name (held-out ID $domain), GPU $device: $save_dir/$domain_name"
    if [[ $mode == parallel ]]; then
        CUDA_VISIBLE_DEVICES="$device" "${command[@]}" >> "$save_dir/$domain_name.log" 2>&1 &
        pids+=("$!")
    else
        CUDA_VISIBLE_DEVICES="$device" "${command[@]}" >> "$save_dir/$domain_name.log" 2>&1
    fi
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
