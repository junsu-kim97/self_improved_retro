# Self-Improved Retrosynthetic Planning
This is an official implementation of our paper Self-improved Retrosynthetic Planning (ICML 2021)
This implemented on top of [Retro*](https://github.com/binghong-ml/retro_star).

## Setup

### Install packages with conda

```bash
conda create -n sirp python=3.7 pytorch=1.5.1 cudatoolkit=10.1 torchvision -c pytorch
conda activate sirp
conda install pandas
conda install rdkit -c rdkit
conda install networkx
conda install graphviz
conda install python-graphviz
pip install tqdm
```

### Install Retro* lib
```bash
pip install -e retro_star/packages/mlp_retrosyn
pip install -e retro_star/packages/rdchiral
pip install -e .
```

### Download dataset
Download dataset following [Retro*](https://github.com/binghong-ml/retro_star).

Download 299202 target molecule dataset from [link](https://www.dropbox.com/s/0nezo0xith8bnml/routes_train.pkl?dl=0),
and place the pkl file into ./retro_star/dataset/

### Download pre-trained model
#### a) Reference backward reaction model
For a reference backward reaction model, we use backward reaction model trained by authors of Retro* 
(./one_step_model/saved_rollout_state_1_2048.ckpt).
We initialize parameters of a backward reaction model to that of the reference backward model.

#### b) Forward reaction model
We provide our pre-trained forward reaction model, 
in this [link](https://drive.google.com/drive/u/0/folders/13DdftEV0x55OZ8ZxHNAkmcvi_4x90hPI),
which is trained by reaction dataset constructed by authors of Retro*.
Place the forward.ckpt file in ./retro_star/one_step_model/forward/.

#### c) (Optional) Trained backward reaction model
We provide checkpoint of bacward reaction model trainded by us in this [link](https://drive.google.com/drive/u/0/folders/13DdftEV0x55OZ8ZxHNAkmcvi_4x90hPI).
See retro_star_zero_ours.ckpt and retro_star_value_ours.ckpt 

## 1. Generate reaction pathways
```bash
python retro_plan.py \
  --test_routes ${TARGET_MOL_DATASET} \
  --mlp_model_dump ${BACKWARD_MODEL} \
  --result_folder ${RESULT_FOLDER} \
  --iteration 500
```

## 2. Extract reactions from pathways (with reaction augmentation)
```bash
python proc_gold_rxn.py \
  --plan \
  ${RESULT_FOLDER}/plan.pkl \
  --save_path \
  ${RESULT_FOLDER}/gold.csv \
  --tpl2prod_save_path \
  ${RESULT_FOLDER}/templates.dat \
  --cut_off \
  -thr 0.8 \
  --aug_forward \
  --forward_model ${FORWARD_MODEL} \
  --fw_backward_validate
```

## 3. Train backward reaction model
```bash
CUDA_VISIBLE_DEVICES=${GPU} \
python -m packages.mlp_retrosyn.mlp_retrosyn.mlp_train \
    --template_path \
    ${RESULT_FOLDER}/templates.dat \
    --template_path_test \
    ${RESULT_FOLDER}/templates.dat \
    --template_rule_path 
    ./one_step_model/template_rules_1.dat \
    --model_dump_folder \
    ${MODEL_DUMP_FOLDER} \
    --fp_dim 2048 \
    --batch_size 1024 \
    --dropout_rate 0.4 \
    --learning_rate 0.0001 \
    --train_path \
    ${RESULT_FOLDER}/gold.csv \
    --test_path \
    ${RESULT_FOLDER}/gold.csv \
    --train_from \
    ${BACKWARD_MODEL} \
    --epochs 20
```

## (Optional) Run whole procedure at once (with parallelization)
To parallelize the reaction pathway generation, we recommend to split the target molecule dataset.
```bash
mkdir ./retro_star/dataset/train_routes_shards
python preprocessing/split_big_dataset.py
```
We offer a script which can conduct whole procedure of our framework at once.
(We assume that you have access to 4 GPU. Otherwise, change the GPU_NUM option, i.e., GPU_NUM=1)

To run iteration 1 for our framework with Retro*-0,
```bash
./scripts/retro_star_zero.sh \
    ${backward model_path} \
    ${forward model_path} \
    ${iter} \
    ${shard_from} ${shard_to} \
    ${gpu_start} ${gpu_num}

e.x)
./scripts/retro_star_zero.sh \
    one_step_model/saved_rollout_state_1_2048.ckpt \
    ./retro_star/one_step_model/forward/saved_rollout_state_1_2048_2021-02-09_19:06:41.ckpt \
    1 \
    0 11 \
    0 4
```

If you want iterate one more (iteration 2) with the updated backward model, run the following script.
```bash
./scripts/retro_star_zero.sh \
    ${trained_backward_model} \
    ${forward_model_path} \
    2 \
    0 11 \
    0 4
```


To run iteration 1 for our framework with Retro*,
```bash
./scripts/retro_star_value.sh \
    ${backward model_path} \
    ${forward model_path} \
    ${iter} \
    ${shard_from} ${shard_to} \
    ${gpu_start} ${gpu_num}
```

## Evaluation
### Backward reaction model
```bash
python eval_one_step.py \
    --model_path \
    ${trained_backward_model}
```

### Retrosynthetic planning
You can evaluate retrosynthetic planning on Retro*-0 + Ours with following script.
```bash
cd retro_star
python retro_plan.py \
    --iteration 500 \
    --result_folder \
    ./results/retro_star_zero/x1/multi-step/eval \
    --mlp_model_dump ${trained_backward_model}
```
You can evaluate retrosynthetic planning on Retro* + Ours with following script.
```bash
cd retro_star
python retro_plan.py \
    --iteration 500 \
    --result_folder \
    ./results/retro_star_zero/x1/multi-step/eval \
    --mlp_model_dump ${trained_backward_model} \
    --use_value_fn
```
After that, you can measure length | time | cost from the log file (plan.pkl)
```bash
python evaluate.py ./results/retro_star_zero/x1/multi-step/eval/plan.pkl
```