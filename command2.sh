models="resnet50"
datasets="cub svhn"
k_values="1 2 4 0.5 0.25"
scale_values="1 2 4 0.5 0.25"

for model in $models
do
    for dataset in $datasets
    do
        # k_multiply에 대한 반복문
        for k in $k_values
        do 
            # scale_multiply에 대한 반복문
            for scale in $scale_values
            do
                k_float=$(echo "scale=10; $k" | bc)
                scale_float=$(echo "scale=10; $scale" | bc)
                python main.py --dataset=$dataset --model=$model --k_multiply=$k_float --scale_multiply=$scale_float --device=1 --mode=GB --max_f=0.2 --min_f=1.0
                python main.py --dataset=$dataset --model=$model --k_multiply=$k_float --scale_multiply=$scale_float --device=1 --mode=GB --max_f=0.2 --min_f=2.0
                python main.py --dataset=$dataset --model=$model --k_multiply=$k_float --scale_multiply=$scale_float --device=1 --mode=GB --max_f=0.2 --min_f=4.0
                python main.py --dataset=$dataset --model=$model --k_multiply=$k_float --scale_multiply=$scale_float --device=1 --mode=GB --max_f=0.1 --min_f=1.0
                python main.py --dataset=$dataset --model=$model --k_multiply=$k_float --scale_multiply=$scale_float --device=1 --mode=GB --max_f=0.1 --min_f=2.0
                python main.py --dataset=$dataset --model=$model --k_multiply=$k_float --scale_multiply=$scale_float --device=1 --mode=GB --max_f=0.1 --min_f=1.0
                python main.py --dataset=$dataset --model=$model --k_multiply=$k_float --scale_multiply=$scale_float --device=1 --mode=GB --max_f=0.05 --min_f=1.0
                python main.py --dataset=$dataset --model=$model --k_multiply=$k_float --scale_multiply=$scale_float --device=1 --mode=GB --max_f=0.05 --min_f=2.0
                python main.py --dataset=$dataset --model=$model --k_multiply=$k_float --scale_multiply=$scale_float --device=1 --mode=GB --max_f=0.05 --min_f=4.0
            done
        done
    done
done