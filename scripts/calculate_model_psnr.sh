#!/bin/bash

conda activate pytorch_p36

ROOT_CHEKCPOINT=/hiero_efs/HieroExperiments/Sparse_VAE/checkpoint
# DATE_CHEKCPOINT=2020_10_28
DATE_CHEKCPOINT=2020_11_04


SEED="1"
STRIDES=("1" "2")
SELECT_FNS=("van" "omp")
NUM_NONZERO=("1" "2" "4")

function run_psnr_eval () {
    # parameters
    # 1 - experiment name prefix
    # 2 - selection function
    # 3 - stride
    # 4 - num_nonzeros
    # 5 - seed
    # 6 - X normalization
    # 7 - D normalization

    experiment_name="$DATE_CHEKCPOINT/$1"
#     --is_quantize_coefs  -stride=2 -sel=omp -k=4
    declare ARGS=""
    declare ARGS=$ARGS"-n=$experiment_name "
    declare ARGS=$ARGS"-sel=$2 "
    declare ARGS=$ARGS"-stride=$3 "
    declare ARGS=$ARGS"-k=$4 "
    declare ARGS=$ARGS"--seed=$5 "
    if [ "$6" == "False" ];  then
        declare ARGS=$ARGS"--no_normalize_x "
    fi
    if [ "$7" == "False" ];  then
        declare ARGS=$ARGS"--no_normalize_dict "
    fi
    if [ "$2" == "omp" ];  then
        declare ARGS=$ARGS"--is_quantize_coefs "
    fi

    echo "#################"    #| tee "$3"/run_params.txt -a
    echo "command args = $ARGS" #| tee "$3"/run_params.txt -a
    echo "#################"    #| tee "$3"/run_params.txt -a

    python scripts/calculate_model_psnr.py $ARGS
}



# for JZ_NAME in "${!JZ_MODELS[@]}"
for SELECT_FN in ${SELECT_FNS[@]}; do

    echo
    echo '**********************'
    echo '****' $SELECT_FN '****'
    echo '**********************'
    echo

    for STRIDE in  ${STRIDES[@]}; do

        EXP_NAME_PREFIX="${SELECT_FN}_s${STRIDE}_${SEED}"
        case $SELECT_FN in
            "van")
                run_psnr_eval $EXP_NAME_PREFIX "vanilla" $STRIDE "1" $SEED False False
                run_psnr_eval $EXP_NAME_PREFIX"_nrm" "vanilla" $STRIDE "1" $SEED True True
                run_psnr_eval $EXP_NAME_PREFIX"_nrmD" "vanilla" $STRIDE "1" $SEED False True
                ;;
            "omp")
                for K in ${NUM_NONZERO[@]}; do
                    run_psnr_eval $EXP_NAME_PREFIX"_k"$K "omp" $STRIDE $K $SEED True True
                    run_psnr_eval $EXP_NAME_PREFIX"_k"$K"_nrmD" "omp" $STRIDE $K $SEED False True
                done
                ;;
            *)
                echo "bad SELECT_FN '${SELECT_FN}'"
                return 1
                ;;
        esac
    done
done
