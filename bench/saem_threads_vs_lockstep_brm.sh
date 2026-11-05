#!/usr/bin/env bash
# Runs `saem_threads_vs_lockstep_brm.jl` at each thread count the grant allows, in BRM's
# pinned environment (`saem_threads_vs_lockstep_brm_env.jl`), and appends to the results
# file. A smoke run on a small model goes first, so a broken script fails in seconds.
#
#   kb-acquire-compute-token --anywhere --no-setup --parallel 10800 --why '...' \
#       --outputs bench/saem_threads_vs_lockstep_brm_results.txt \
#       -- bash bench/saem_threads_vs_lockstep_brm.sh
set -euo pipefail
root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
brm_sha=${BRM_SHA:-7aee63944b7d04bde027f38085d6bda1013de197} # BRM 0.2.199, ns/devibe
work=${BENCH_WORK:-${TMPDIR:-/tmp}/saem-threads-vs-lockstep}
results=$root/bench/saem_threads_vs_lockstep_brm_results.txt
cpus=${KB_COMPUTE_CPUS:-2}
sizes=${BENCH_SIZES:-"80:8 80:80"}
mkdir -p "$work"

brm=$work/brm-$brm_sha
if [ ! -d "$brm" ]; then
    mirror=$HOME/.local/state/kb-agents/git-mirrors/BayesianRegressionModels.jl.git
    source=https://github.com/nsiccha/BayesianRegressionModels.jl.git
    git -C "$mirror" cat-file -e "$brm_sha^{commit}" 2>/dev/null && source=$mirror
    git clone --quiet --no-checkout "$source" "$brm.tmp"
    git -C "$brm.tmp" -c advice.detachedHead=false checkout --quiet "$brm_sha"
    mv "$brm.tmp" "$brm"
fi

export BRM_REPO=$brm BRM_ENV=$work/env BRM_BOOTSTRAP_ROOT=$work/bootstrap NLME_REPO=$root
export OPENBLAS_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_PRECOMPILE_TASKS=$cpus
echo "host=${KB_HOST:-?} cpus=$cpus nlme=$(git -C "$root" rev-parse HEAD) brm=$brm_sha started=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
julia --startup-file=no "$root/bench/saem_threads_vs_lockstep_brm_env.jl" > "$work/setup.log" 2>&1 ||
    { echo SETUP_FAILED; tail -40 "$work/setup.log"; exit 70; }
grep -E '^(PINS|ENV_PKG|ENV_READY|KB-DOGFOOD)' "$work/setup.log"
echo "setup done $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"

run() { # threads units rows output [pairs...]
    local threads=$1 units=$2 rows=$3 output=$4
    shift 4
    julia --startup-file=no --project="$BRM_ENV" --threads="$threads" \
        "$root/bench/saem_threads_vs_lockstep_brm.jl" "$units" "$rows" "$output" "$@"
}

run 1 8 4 "$work/smoke.txt" 2 2 2 > "$work/smoke.log" 2>&1 ||
    { echo SMOKE_FAILED; tail -40 "$work/smoke.log"; exit 71; }
echo "smoke passed $(date -u +%FT%TZ)"

thread_counts=$(for t in 1 2 4 8 16; do [ "$t" -lt "$cpus" ] && echo "$t"; done; echo "$cpus")
{
    echo "Host ${KB_HOST:-?}, a grant of $cpus CPUs, NLMEEstimation $(git -C "$root" rev-parse HEAD), BRM $brm_sha"
    grep -E '^(PINS|ENV_PKG)' "$work/setup.log"
} >> "$results"
for size in $sizes; do
    for threads in $thread_counts; do
        echo "LOAD before ${size} at ${threads} threads: $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$results"
        run "$threads" "${size%:*}" "${size#*:}" "$results"
        echo "LOAD after ${size} at ${threads} threads: $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$results"
    done
done
echo "BENCH_DONE $(date -u +%FT%TZ)"
