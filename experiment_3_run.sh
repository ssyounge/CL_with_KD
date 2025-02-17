#!/bin/bash
#SBATCH --job-name=multi_kd_ewc_exp
#SBATCH --output=multi_kd_ewc_exp_%j.out
#SBATCH --error=multi_kd_ewc_exp_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --nodes=1

echo "==================== SLURM JOB INFO ===================="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================================"

# module load anaconda3
# source activate myenv

cd /home/suyoung425/FACIL  # 실제 경로로 변경

# (1) PYTHONPATH 추가 (또는 -m 방식 사용)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 절대경로 results path
RESULTS_PATH="/home/suyoung425/FACIL/results"

# (2) Teacher ckpt 스킵 로직
echo "=== [Teacher Training Phase] ==="
mkdir -p checkpoints

# T1
if [ ! -f "./checkpoints/T1.pth" ]; then
  echo "[train_teacher] T1 does not exist -> train it."
  python3 -u src/approach/train_teacher.py --epochs 50 --save-checkpoint ./checkpoints/T1.pth
else
  echo "[train_teacher] Found T1.pth -> skip"
fi

# T2
if [ ! -f "./checkpoints/T2.pth" ]; then
  echo "[train_teacher] T2 does not exist -> train it."
  python3 -u src/approach/train_teacher.py --epochs 50 --save-checkpoint ./checkpoints/T2.pth
else
  echo "[train_teacher] Found T2.pth -> skip"
fi

# T3
if [ ! -f "./checkpoints/T3.pth" ]; then
  echo "[train_teacher] T3 does not exist -> train it."
  python3 -u src/approach/train_teacher.py --epochs 50 --save-checkpoint ./checkpoints/T3.pth
else
  echo "[train_teacher] Found T3.pth -> skip"
fi

# T4
if [ ! -f "./checkpoints/T4.pth" ]; then
  echo "[train_teacher] T4 does not exist -> train it."
  python3 -u src/approach/train_teacher.py --epochs 50 --save-checkpoint ./checkpoints/T4.pth
else
  echo "[train_teacher] Found T4.pth -> skip"
fi

# T5
if [ ! -f "./checkpoints/T5.pth" ]; then
  echo "[train_teacher] T5 does not exist -> train it."
  python3 -u src/approach/train_teacher.py --epochs 50 --save-checkpoint ./checkpoints/T5.pth
else
  echo "[train_teacher] Found T5.pth -> skip"
fi

echo "== Check Teacher ckpts in ./checkpoints =="
ls -lh ./checkpoints

# ------------------------------------------------------------------------------
# (1) 여기서부터는 실험 하이퍼파라미터 설정
# ------------------------------------------------------------------------------
NUM_TASKS=(5)
FIRST_TASKS=(50) # 20, 50
LAMB_VALUES=(3000)     # (200 500 2000 3000) 등 가능
ALPHAS=(0.3 0.5 0.7)   # 쉼표 제거
KD_WEIGHTS=(0.5 1.0 2.0 3.0)   # 쉼표 제거
KD_TEMPS=(2.0 3.0 4.0 5.0)     # 쉼표 제거
EXEMPLARS=(2000 5000)

# Teacher checkpoints (절대경로로)
TEACHER_CKPTS="/home/suyoung425/FACIL/checkpoints/T1.pth,/home/suyoung425/FACIL/checkpoints/T2.pth,/home/suyoung425/FACIL/checkpoints/T3.pth,/home/suyoung425/FACIL/checkpoints/T4.pth,/home/suyoung425/FACIL/checkpoints/T5.pth"
SEED=42
EPOCHS=50
BATCH=64
LR=0.1

for NT in "${NUM_TASKS[@]}"; do
  for NFT in "${FIRST_TASKS[@]}"; do
    for LAMB in "${LAMB_VALUES[@]}"; do
      for ALPHA in "${ALPHAS[@]}"; do
        for KD_W in "${KD_WEIGHTS[@]}"; do
          for KD_T in "${KD_TEMPS[@]}"; do
            for EXEMP in "${EXEMPLARS[@]}"; do

              EXP_NAME="KD_EWC_nt${NT}_ft${NFT}_lamb${LAMB}_alpha${ALPHA}_kdW${KD_W}_kdT${KD_T}_ex${EXEMP}"
              # (부모 logger가 results_path로 폴더를 만듦. 별도 mkdir 생략 가능)

              echo "--------------------------------------------------------"
              echo "[Experiment] ${EXP_NAME}"
              echo "  num_tasks  = ${NT}"
              echo "  nc_first   = ${NFT}"
              echo "  lamb(EWC)  = ${LAMB}"
              echo "  alpha(EWC) = ${ALPHA}"
              echo "  kd_weight  = ${KD_W}"
              echo "  kd_temp    = ${KD_T}"
              echo "  exemplars  = ${EXEMP}"
              echo "  result_dir = ${RESULTS_PATH}/${EXP_NAME}  # parent logger path"
              echo "--------------------------------------------------------"

              srun python3 -u src/main_incremental.py \
                --approach multi_teacher_kd_ewc \
                --teacher-checkpoints "${TEACHER_CKPTS}" \
                --results-path "${RESULTS_PATH}" \
                --datasets cifar100 \
                --network resnet32 \
                --num-tasks ${NT} \
                --nc-first-task ${NFT} \
                --lamb ${LAMB} \
                --alpha ${ALPHA} \
                --exp-name "${EXP_NAME}" \
                --kd-weight ${KD_W} \
                --kd-temperature ${KD_T} \
                --num-exemplars ${EXEMP} \
                --exemplar-selection herding \
                --batch-size ${BATCH} \
                --nepochs ${EPOCHS} \
                --lr ${LR} \
                --seed ${SEED} \
                --gating-input-dim 4096 \
                --gating-hidden-dim 128 \
                --some-other-options "..." \
                2>&1 | tee "${RESULTS_PATH}/${EXP_NAME}_train_log.txt"
            done
          done
        done
      done
    done
  done
done

echo "All experiments finished."