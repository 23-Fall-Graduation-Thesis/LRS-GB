# mode="autoGB auto GB standard"
models="resnet18"
datasets="cifar100 cub cars"
k_values="1.95 2.6 3.9 7.8 15.6 23.4 31.2"
scale_values="1.25 1.5 2"
init_score="0.97"

for dataset in $datasets
do
    for k in $k_values
    do
        for scale in $scale_values
        do
            python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True --max_f= --min_f=
        done
    done
done
