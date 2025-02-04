#!/bin/bash

################################################################################
# File: gridsearch_cifar100_facil_extended.sh
#
# Description:
#   - A comprehensive experiment script for FACIL on CIFAR-100,
#     incorporating the extended research ideas:
#       1) Lambda × Beta cross exploration, Gamma expansions
#       2) Multiple scenarios: (5/20), (5/40-15), (10/10), (2/50), (3/33-34), (4/25)...
#       3) Multiple seeds for repeated trials (stability analysis)
#       4) Approaches: LwF, EWC, MAS, PathIntegral, R-Walk, LwM
#       5) (Optional) If you'd like Bayesian Optimization, you must adapt or replace
#          the grid search loop with a Python-based approach (Optuna, Hyperopt, etc.)
#
#   - This script demonstrates how to unify the extended ideas from your plan
#     into one place. Adjust or comment out lines according to your resources.
################################################################################

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"
RESULTS_DIR="$PROJECT_DIR/results"

echo "Project Dir:  $PROJECT_DIR"
echo "Src Dir:      $SRC_DIR"
echo "Results Dir:  $RESULTS_DIR"

NETWORK="resnet32"     # or "resnet18", or any other
EPOCHS=50
BATCH=128

# 여러 seed로 돌려보며 성능 분산(편차)도 볼 수 있음
SEEDS=(0)          # 예시: 0,1,2 총 3회 반복

GPU=0                  # GPU index to use

################################################################################
# 시나리오 목록
# - 기본 (5/20), (5/40-15), (10/10), (2/50) + 확장 (3/33-34), (4/25)
# - (3/33-34) => 첫 태스크 33, 그 뒤 67클래스를 2태스크로? 자유롭게 설계
################################################################################
SCENARIOS=(
  "(5/20)"
  "(5/40-15)"
  "(10/10)"
  "(2/50)"
  "(3/33-34)"
  "(4/25)"
)

################################################################################
# 1) 접근법 리스트
################################################################################
METHODS=(
  "lwf"
  "ewc"
  "mas"
  "path_integral"
  "r_walk"
  "lwm"
)

################################################################################
# 2) 각 접근법별 하이퍼파라미터 (PARAM_GRIDS)
#
#   A. lambda × beta 교차, gamma 확장 등
#   B. 필요하다면 T, alpha, damping 등도 확장
#   C. 여기서 예시로 크게 확장한 범위를 보여드립니다. 실제로는 취사선택하세요.
################################################################################
declare -A PARAM_GRIDS

# LwF
# - lamb, T 조합 늘림
PARAM_GRIDS["lwf"]="\
lamb=0.5,T=2 lamb=0.5,T=4 \
lamb=1,T=2 lamb=1,T=4 lamb=1,T=5 \
lamb=2,T=2 lamb=2,T=5 \
lamb=3,T=2 lamb=3,T=4 \
lamb=5,T=2 lamb=5,T=4 lamb=5,T=6 \
lamb=10,T=2 \
"

# EWC
# - lambda 폭넓게
PARAM_GRIDS["ewc"]="\
lamb=0.5 lamb=1 lamb=10 lamb=100 lamb=500 \
lamb=1000 lamb=2000 lamb=5000 lamb=10000 lamb=20000 \
"

# MAS
# - lambda 폭넓게
PARAM_GRIDS["mas"]="\
lamb=0.5 lamb=1 lamb=2 lamb=3 lamb=5 \
lamb=10 lamb=100 lamb=500 \
"

# Path Integral
# - lambda, damping 확장
PARAM_GRIDS["path_integral"]="\
lamb=0.05,damping=0.1 lamb=0.1,damping=0.1 lamb=0.15,damping=0.2 \
lamb=0.1,damping=0.05 lamb=0.2,damping=0.1 \
lamb=0.5,damping=0.1 lamb=1,damping=0.1 lamb=2,damping=0.1 \
"

# R-Walk
# - lambda, alpha 확장
PARAM_GRIDS["r_walk"]="\
lamb=1,alpha=0.3 lamb=1,alpha=0.5 \
lamb=2,alpha=0.5 lamb=3,alpha=0.6 \
lamb=5,alpha=0.5 lamb=10,alpha=0.5 \
lamb=100,alpha=0.3 \
"

# LwM
# - beta, gamma, + optional lamb
# - 예: 교차 탐색 (beta ∈ {1,2,3,5}, gamma ∈ {1,5,10,15}), + lambda도 일부 조합
PARAM_GRIDS["lwm"]="\
beta=1,gamma=1 beta=1,gamma=5 beta=1,gamma=10 beta=1,gamma=15 \
beta=2,gamma=1 beta=2,gamma=5 beta=2,gamma=10 beta=2,gamma=15 \
beta=3,gamma=1 beta=3,gamma=5 beta=3,gamma=10 beta=3,gamma=15 \
beta=5,gamma=1 beta=5,gamma=10 beta=5,gamma=15 \
lamb=1,beta=2,gamma=5 lamb=2,beta=2,gamma=5 lamb=10,beta=1,gamma=10 \
"

################################################################################
# 3) 접근법별 메모리(exemplars) 설정 함수
#    - LwF, EWC, MAS, PathIntegral, R-Walk, LwM => 여기선 모두 20개로 세팅
#      (원하시면 접근법/시나리오별로 바꿔보세요)
################################################################################
function custom_memory_setting {
  local m="$1"
  case $m in
    "lwf"|"ewc"|"mas"|"path_integral"|"r_walk"|"lwm")
      echo 20
      ;;
    *)
      echo 0
      ;;
  esac
}

################################################################################
# 4) 파라미터 문자열 -> CLI 인자 변환 ("lamb=1,T=2" -> "--lamb 1 --T 2")
################################################################################
function parse_params_to_args {
  local paramstring="$1"
  local args=""

  IFS=',' read -ra kvpairs <<< "$paramstring"
  for kv in "${kvpairs[@]}"; do
    IFS='=' read -ra pair <<< "$kv"
    local k="${pair[0]}"
    local v="${pair[1]}"

    case $k in
      "lamb")
        args="$args --lamb $v"
        ;;
      "beta")
        args="$args --beta $v"
        ;;
      "gamma")
        args="$args --gamma $v"
        ;;
      "T")
        args="$args --T $v"
        ;;
      "alpha")
        args="$args --alpha $v"
        ;;
      "damping")
        args="$args --damping $v"
        ;;
      *)
        args="$args --$k $v"
        ;;
    esac
  done

  echo "$args"
}

################################################################################
# 5) 시나리오 설정 함수
#    - (5/20), (5/40-15), (10/10), (2/50), (3/33-34), (4/25)
#    - 원하는 방식대로 nc_first_task, num_tasks 조정
#    - (3/33-34), (4/25)는 예시로 해석
################################################################################
function scenario_setting {
  local scen="$1"
  local num_tasks=5
  local nc_first=20
  local exp_tag="5_20"

  if [ "$scen" = "(5/40-15)" ]; then
    num_tasks=5
    nc_first=40
    exp_tag="5_40_15"

  elif [ "$scen" = "(10/10)" ]; then
    num_tasks=10
    nc_first=10
    exp_tag="10_10"

  elif [ "$scen" = "(2/50)" ]; then
    num_tasks=2
    nc_first=50
    exp_tag="2_50"

  elif [ "$scen" = "(3/33-34)" ]; then
    # 예시: 총 태스크 3개, 첫 태스크 33개
    # 이후 67개 클래스를 2개 태스크로 분할(34, 33) or (33, 34)
    num_tasks=3
    nc_first=33
    exp_tag="3_33_34"

  elif [ "$scen" = "(4/25)" ]; then
    # 예시: 총 태스크 4개, 첫 태스크 25개
    # 나머지 75개 클래스를 3개 태스크에 분배(예: 25,25,25)
    num_tasks=4
    nc_first=25
    exp_tag="4_25"
  fi

  echo "$num_tasks $nc_first $exp_tag"
}

################################################################################
# 6) Bayesian Optimization (참고)
#   - 현재는 Grid Search만 구현. BO 활용 시, Python 코드로 main_incremental.py를
#     직접 여러 번 호출하면서 Optuna 등으로 탐색해야 함.
#   - 아래 주석은 예시 아이디어입니다.
################################################################################
: '
function run_bayesian_optimization {
  # Pseudocode:
  #   1) Use optuna/hyperopt to define search space for lamb, beta, gamma, etc.
  #   2) For each trial, system() call:
  #        python3 main_incremental.py ... --lamb <trial_lamb> --beta <trial_beta> ...
  #      measure the final accuracy or BWT as objective
  #   3) BO library finds best hyperparameters
  #   4) Possibly run top combos again
}
'

################################################################################
# 7) 메인 실행 루프
#    - 시나리오 × 접근법 × (파라미터 조합) × seed
################################################################################

for SEED in "${SEEDS[@]}"; do
  echo "----------------------------------------------"
  echo "Running for SEED = $SEED"
  echo "----------------------------------------------"

  for SCEN in "${SCENARIOS[@]}"; do
    echo "==========================================="
    echo "Scenario: $SCEN"
    echo "==========================================="

    read NUM_TASKS NC_FIRST EXP_TAG <<< "$(scenario_setting $SCEN)"

    for METHOD in "${METHODS[@]}"; do

      PARAM_SET="${PARAM_GRIDS[$METHOD]}"
      if [ -z "$PARAM_SET" ]; then
        PARAM_SET="default=1"
      fi

      MEM_PER_CLASS="$(custom_memory_setting $METHOD)"

      for param_combo in $PARAM_SET; do
        EXTRA_ARGS="$(parse_params_to_args $param_combo)"
        COMBO_TAG=$(echo "$param_combo" | sed 's/[=,]/_/g')

        RESULT_DIRNAME="cifar100_${METHOD}_${EXP_TAG}_${COMBO_TAG}_seed${SEED}"
        RESULT_FOLDER="${RESULTS_DIR}/${RESULT_DIRNAME}"

        # 이미 존재하면 스킵
        if [ -d "$RESULT_FOLDER" ]; then
          echo ">>> Skip existing: $RESULT_FOLDER"
          continue
        fi

        EXP_NAME="${METHOD}_${EXP_TAG}_${COMBO_TAG}_seed${SEED}"
        echo ">>> [GPU=$GPU] $METHOD param=($param_combo), SCEN=$SCEN, seed=$SEED, mem=$MEM_PER_CLASS"

        python3 -u "$SRC_DIR/main_incremental.py" \
          --datasets cifar100_icarl \
          --approach $METHOD \
          --network $NETWORK \
          --num-tasks $NUM_TASKS \
          --nc-first-task $NC_FIRST \
          --nepochs $EPOCHS \
          --batch-size $BATCH \
          --seed $SEED \
          --gpu $GPU \
          --results-path $RESULTS_DIR \
          --exp-name $EXP_NAME \
          --exemplar-selection herding \
          --num-exemplars-per-class $MEM_PER_CLASS \
          $EXTRA_ARGS

      done
    done
  done
done

echo "All extended experiments done!"