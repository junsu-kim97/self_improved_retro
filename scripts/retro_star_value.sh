BACKWARD_MODEL=$1
FORWARD_MODEL=$2
ITER=$3
SHARD_FROM=$4
SHARD_TO=$5
GPU_START=$6
GPU_NUM=$7

# shellcheck disable=SC2164
cd retro_star
for ((shard=SHARD_FROM; shard<=SHARD_TO; shard+=1)); do
  python retro_plan.py \
  --test_routes ./dataset/train_routes_shards/shard_${shard}.pkl \
  --mlp_model_dump ${BACKWARD_MODEL} \
  --result_folder ./results/retro_star_value/x${ITER}/multi-step/succ_traj/shard_${shard} \
  --iteration 500 \
  --gpu $(( $GPU_START + shard % $GPU_NUM)) \
  --use_value_fn \
  &
done
wait

echo "success route generation done"
# shellcheck disable=SC2164
cd ../
for ((shard=SHARD_FROM; shard<=SHARD_TO; shard+=1)); do
  python proc_gold_rxn.py \
  --plan \
  ./retro_star/results/retro_star_value/x${ITER}/multi-step/succ_traj/shard_${shard}/plan.pkl \
  --save_path \
  ./retro_star/results/retro_star_value/x${ITER}/multi-step/succ_traj/shard_${shard}/gold.csv \
  --tpl2prod_save_path \
  ./retro_star/results/retro_star_value/x${ITER}/multi-step/succ_traj/shard_${shard}/templates.dat \
  --cut_off \
  -thr 0.8 \
  --aug_forward \
  --forward_model ${FORWARD_MODEL} \
  --fw_backward_validate \
  --gpu $(( $GPU_START + shard % $GPU_NUM)) \
  &
done
wait

echo "proc gold rxn done"

g_list=""
t_list=""
for ((shard=SHARD_FROM; shard<=SHARD_TO; shard+=1)); do
    g_list="${g_list} ./retro_star/results/retro_star_value/x${ITER}/multi-step/succ_traj/shard_${shard}/gold.csv"
    t_list="${t_list} ./retro_star/results/retro_star_value/x${ITER}/multi-step/succ_traj/shard_${shard}/templates.dat"
done
wait

python preprocessing/merge_proc_data.py \
-g ${g_list} \
-t ${t_list} \
--save_path_gold_csv \
./retro_star/results/retro_star_value/x${ITER}/multi-step/succ_traj/gold.csv \
--save_path_tpl \
./retro_star/results/retro_star_value/x${ITER}/multi-step/succ_traj/templates.dat

wait

# shellcheck disable=SC2164
cd retro_star
# shellcheck disable=SC2034
CUDA_VISIBLE_DEVICES=${GPU_START} \
python -m packages.mlp_retrosyn.mlp_retrosyn.mlp_train \
--template_path \
./results/retro_star_value/x${ITER}/multi-step/succ_traj/templates.dat \
--template_path_test \
./results/retro_star_value/x${ITER}/multi-step/succ_traj/templates.dat \
--template_rule_path \
./one_step_model/template_rules_1.dat \
--model_dump_folder \
./results/retro_star_value/x${ITER}/one-step/ \
--fp_dim 2048 \
--batch_size 1024 \
--dropout_rate 0.4 \
--learning_rate 0.0001 \
--train_path \
./results/retro_star_value/x${ITER}/multi-step/succ_traj/gold.csv \
--test_path \
./results/retro_star_value/x${ITER}/multi-step/succ_traj/gold.csv \
--train_from \
${BACKWARD_MODEL} \
--epochs 20