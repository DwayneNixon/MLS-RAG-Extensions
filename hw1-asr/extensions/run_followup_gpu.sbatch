#!/bin/bash
# Follow-up GPU experiments, in two parts (each part is one job, so every comparison
# inside it shares a session; PyTorch is timed at the start and the end of each).
#
#   PART=kernels  (about 45-60 min)
#     - M sweep at N = K = 2048 for 4x4 and 8x4 register blocks, each with 32- and
#       64-row tiles (bench_linear.cu): GFLOP/s, blocks, threads per block
#     - ncu, if the node allows hardware counters: achieved occupancy of 4x4 and 8x4
#       at M = 64; shared-memory bank conflicts of v2, v2 without padding and v2 with
#       interleaved columns
#     - GEMM v3 (extensions/gemm_v3: 8x4 on M >= 128, 4x4 below; tile depth 16 or 32)
#       against v2: the 10 GEMM tests, racecheck, reference transcription, 3 timed runs
#   PART=clips    (about 60-90 min)
#     - PyTorch, the assigned kernels, v2 and v3 on the reference clip and on three
#       longer clips built from Practice question recordings (make_clips.py)
#
# From hw1-asr, on the head node:
#     sbatch --export=ALL,PART=kernels extensions/run_followup_gpu.sbatch
#     sbatch --export=ALL,PART=clips   extensions/run_followup_gpu.sbatch
#     cat extensions/results/followup_<part>_JOBID/summary.txt
#
# Nothing in the repository is modified: variants are generated in /tmp.
#SBATCH --job-name=mls-followup
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=followup-%j.out
#SBATCH --no-requeue

set -o pipefail
PART=${PART:-kernels}
export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"
[ -f extensions/gemm_v3/task3_gemm.cu ] || { echo "Run from hw1-asr"; exit 1; }

OUT=extensions/results/followup_${PART}_${SLURM_JOB_ID}
WORK=/tmp/$USER/followup_${SLURM_JOB_ID}
mkdir -p "$OUT" "$WORK"
BASE_FLAGS="-O3 -std=c++17 -lineinfo -diag-suppress 177"   # the Makefile's own flags
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')

{
  date -Is; hostname; nvidia-smi -L
  nvidia-smi --query-gpu=name,uuid,compute_cap --format=csv
  echo "part: $PART"
  echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  for f in extensions/gemm_v2/task3_gemm.cu extensions/gemm_v3/task3_gemm.cu; do
    echo "$f sha256: $(sha256sum $f | cut -d' ' -f1)"; done
  echo "ncu: $(command -v ncu || echo 'not found')"
} | tee "$OUT/00_environment.txt"

step () {  # $1 = log name, rest = command; a failing step is logged, not fatal
  local name=$1; shift
  echo; echo "===== $name: $* ====="
  "$@" 2>&1 | tee "$OUT/$name.txt"
  echo "exit status: ${PIPESTATUS[0]}" | tee -a "$OUT/$name.txt"
}

# variant NAME FROM_DIR EXTRA_NVCC_FLAGS [sed expressions...]: copy, edit, build lib + GEMM test
variant () {
  local name=$1 from=$2 extra=$3; shift 3
  local src=$WORK/$name/src build=$WORK/$name/build
  mkdir -p "$src"
  cp -rL "$from/." "$src/"
  for expr in "$@"; do sed -i -e "$expr" "$src/task3_gemm.cu"; done
  {
    echo "variant $name from $from, extra flags: '${extra}'"
    echo "lines differing from $from/task3_gemm.cu: $(diff "$from/task3_gemm.cu" "$src/task3_gemm.cu" | grep -c '^>')"
    diff "$from/task3_gemm.cu" "$src/task3_gemm.cu"
  } > "$OUT/variant_${name}.txt"
  make -s SRC="$src" BUILD="$build" NVCCFLAGS="$BASE_FLAGS $extra" lib >> "$OUT/variant_${name}.txt" 2>&1
  echo "$build"
}

bench () {  # NAME BUILD BM TM TN : compile bench_linear against the variant, run the M sweep
  local name=$1 build=$2
  nvcc $BASE_FLAGS -arch=sm_$SM -I"$WORK/$name/src" -DBENCH_BM=$3 -DBENCH_TM=$4 -DBENCH_TN=$5 \
       extensions/bench_linear.cu "$build"/obj/*.o -o "$build/bench_linear" \
    && "$build/bench_linear" 2048 2048 16 32 64 96 128 256 512 1500
}

if [ "$PART" = "kernels" ]; then
  step 01_pytorch_start ./run.sh --cuda none --profile

  V2=extensions/gemm_v2
  V3=extensions/gemm_v3
  ALWAYS64='s/^  if (M >= 128) {/  if (M >= 0) {/'
  ALWAYS32='s/^  if (M >= 128) {/  if (M >= 1000000000) {/'

  # --- 2a: M sweep, register block x tile height (only TM, TN and the dispatch line change)
  for tm in 4 8; do
    for bm in 32 64; do
      name=sweep_${tm}x4_bm${bm}
      sel=$ALWAYS32; [ $bm = 64 ] && sel=$ALWAYS64
      b=$(variant $name $V2 "" "s/^constexpr int TM = 4;/constexpr int TM = ${tm};/" "$sel")
      step 02_${name} bench $name "$b" $bm $tm 4
    done
  done

  # --- 2b: profiler counters, if the node allows them
  if command -v ncu >/dev/null; then
    OCC="sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_registers,launch__registers_per_thread,launch__block_size,launch__grid_size,sm__maximum_warps_per_active_cycle_pct"
    for name in sweep_4x4_bm32 sweep_8x4_bm32; do
      step 03_ncu_occupancy_${name} ncu --metrics $OCC --kernel-name regex:linear_regtile \
        --launch-skip 3 --launch-count 1 "$WORK/$name/build/bench_linear" 2048 2048 64
    done
  else
    echo "ncu not found on this node" | tee "$OUT/03_ncu_unavailable.txt"
  fi

  # --- 2c: v2, v3 (three tile depths), and the bank-conflict variants of v2
  declare -A BUILD
  BUILD[v2]=$(variant v2 $V2 "")
  BUILD[v3]=$(variant v3 $V3 "")                                   # enc 8x4 depth 16, dec 4x4 depth 32
  BUILD[v3_bk16]=$(variant v3_bk16 $V3 "-DV3_BK_ENC=16 -DV3_BK_DEC=16")
  BUILD[v3_bk32]=$(variant v3_bk32 $V3 "-DV3_BK_ENC=32 -DV3_BK_DEC=32")
  BUILD[v2_interleave]=$(variant v2_interleave $V2 "" \
      's/b\[j\] = ws\[kk\]\[tx \* TN + j\];/b[j] = ws[kk][tx + j * (BN \/ TN)];/' \
      's/c = col0 + tx \* TN + j;/c = col0 + tx + j * (BN \/ TN);/')
  BUILD[v2_nopad]=$(variant v2_nopad $V2 "" 's/xs\[BK\]\[BM + 1\]/xs[BK][BM]/' 's/ws\[BK\]\[BN + 1\]/ws[BK][BN]/')

  for name in v2 v3 v3_bk16 v3_bk32 v2_interleave v2_nopad; do
    step 04_test3_${name} make -s SRC="$WORK/$name/src" BUILD="${BUILD[$name]}" \
      NVCCFLAGS="$BASE_FLAGS $(grep -o "extra flags: '[^']*'" "$OUT/variant_${name}.txt" | cut -d"'" -f2)" test3
  done
  # Racecheck must reach BOTH dispatch paths. The GEMM test's first launches are all
  # M < 128, so the test binary alone would never check v3's 8x4 path. bench_linear on
  # odd shapes covers each path directly: M = 150 (>= 128, 8x4) and M = 37 (< 128, 4x4).
  for name in v3 v3_bk16 v3_bk32 v2_interleave; do
    nvcc $BASE_FLAGS -arch=sm_$SM -I"$WORK/$name/src" extensions/bench_linear.cu \
      "${BUILD[$name]}"/obj/*.o -o "${BUILD[$name]}/bench_linear"
    step 05_racecheck_${name}_M150 compute-sanitizer --tool racecheck "${BUILD[$name]}/bench_linear" 100 33 150
    step 05_racecheck_${name}_M37  compute-sanitizer --tool racecheck "${BUILD[$name]}/bench_linear" 100 33 37
  done
  if command -v ncu >/dev/null; then
    BANK="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum"
    for name in v2 v2_nopad v2_interleave; do
      nvcc $BASE_FLAGS -arch=sm_$SM -I"$WORK/$name/src" extensions/bench_linear.cu \
        "${BUILD[$name]}"/obj/*.o -o "${BUILD[$name]}/bench_linear"
      step 06_ncu_banks_${name} ncu --metrics $BANK --kernel-name regex:linear_regtile \
        --launch-skip 3 --launch-count 1 "${BUILD[$name]}/bench_linear" 2048 2048 64
    done
  fi

  # --- 2d: whole-model timing and the reference transcription, same session
  for name in v2 v3 v3_bk16 v3_bk32 v2_interleave; do
    step 07_e2e_${name} env ASR_LIB="${BUILD[$name]}/libasr_kernels.so" ./run.sh --profile
  done
  step 08_pytorch_end ./run.sh --cuda none --profile

elif [ "$PART" = "clips" ]; then
  RAGDATA=../rag/data/rag_corpus
  [ -f "$RAGDATA/student_audio_manifest.jsonl" ] || { echo "Missing $RAGDATA (install the RAG assets)"; exit 1; }
  step 01_make_clips python extensions/make_clips.py "$RAGDATA" "$WORK/clips"
  cp "$WORK/clips/clips.json" "$OUT/"
  V2LIB=$(variant v2 extensions/gemm_v2 "")/libasr_kernels.so
  V3LIB=$(variant v3 extensions/gemm_v3 "")/libasr_kernels.so
  make -s lib   # the assigned kernels, as ./run.sh would build them

  step 02_pytorch_start ./run.sh --cuda none --profile
  for clip in reference clip_1q clip_3q clip_6q; do
    audio=test_audio.wav; [ $clip != reference ] && audio=$WORK/clips/$clip.wav
    # benchmark.py reports FAIL for any clip other than the reference (it compares
    # with the reference sentence); the transcripts are compared across backends instead
    step 03_${clip}_pytorch  python benchmark.py --cuda none --profile --audio "$audio"
    step 03_${clip}_assigned python benchmark.py --profile --audio "$audio"
    step 03_${clip}_v2       env ASR_LIB="$V2LIB" python benchmark.py --profile --audio "$audio"
    step 03_${clip}_v3       env ASR_LIB="$V3LIB" python benchmark.py --profile --audio "$audio"
  done
  step 04_pytorch_end ./run.sh --cuda none --profile
else
  echo "Unknown PART=$PART (use kernels or clips)"; exit 1
fi

echo; echo "===== SUMMARY ====="
python extensions/summarise_followup.py "$OUT" | tee "$OUT/summary.txt"
date -Is | tee -a "$OUT/00_environment.txt"
sed -i "s/$USER/UUN/g" "$OUT"/*.txt "$OUT"/*.json 2>/dev/null   # keep the saved logs free of the username
rm -rf "$WORK"
