# mode="autoGB auto GB standard"
mode="autoGB"
model="resnet18"
datasets="cifar100 cub cars"
k_values="1.95 2.6 3.9 7.8 15.6 23.4 31.2"
scale_values="1.25 1.5 2"
init_score="0.97"

############### 여기는 autoGB 코드 #################

# for dataset in $datasets
# do
#     python main.py --dataset=$dataset --mode=$mode --model=$model --K=15.6 --scale_factor=2 --bound=weva --thr_init_score=$init_score --increase_bound=False --max_f=0.1 --min_f=1.0
#     python main.py --dataset=$dataset --mode=$mode --model=$model --K=15.6 --scale_factor=2 --bound=weva --thr_init_score=$init_score --increase_bound=False --max_f=0.1 --min_f=2.0
# done

# k_values="23.4 31.2"
# scale_values="1.25 1.5 2"

# for dataset in $datasets
# do
#     for k in $k_values
#     do
#         for scale in $scale_values
#         do
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=False --max_f=0.1 --min_f=1.0
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=False --max_f=0.1 --min_f=2.0
#         done
#     done
# done

# for dataset in $datasets
# do
#     for k in $k_values
#     do
#         for scale in $scale_values
#         do
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=diff --thr_init_score=$init_score --increase_bound=False --max_f=0.1 --min_f=1.0
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=diff --thr_init_score=$init_score --increase_bound=False --max_f=0.1 --min_f=2.0
#         done
#     done
# done

# for dataset in $datasets
# do
#     for k in $k_values
#     do
#         for scale in $scale_values
#         do
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True --max_f=0.1 --min_f=1.0
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True --max_f=0.1 --min_f=2.0
#         done
#     done
# done

# for dataset in $datasets
# do
#     for k in $k_values
#     do
#         for scale in $scale_values
#         do
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=diff --thr_init_score=$init_score --increase_bound=True --max_f=0.1 --min_f=1.0
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=diff --thr_init_score=$init_score --increase_bound=True --max_f=0.1 --min_f=2.0
#         done
#     done
# done


# for dataset in $datasets
# do
#     for k in $k_values
#     do
#         for scale in $scale_values
#         do
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True --K_mode=4 --max_f=0.1 --min_f=1.0
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True --K_mode=4 --max_f=0.1 --min_f=2.0
#         done
#     done
# done

datasets="cifar100"
k_values="31.2"
scale_values="1.25 1.5 2"
for dataset in $datasets
do
    for k in $k_values
    do
        for scale in $scale_values
        do
            python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True --K_mode=5 --max_f=0.1 --min_f=1.0
            python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True --K_mode=5 --max_f=0.1 --min_f=2.0
        done
    done
done
datasets="cub cars"
k_values="1.95 2.6 3.9 7.8 15.6 23.4 31.2"
scale_values="1.25 1.5 2"
for dataset in $datasets
do
    for k in $k_values
    do
        for scale in $scale_values
        do
            python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True --K_mode=5 --max_f=0.1 --min_f=1.0
            python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True --K_mode=5 --max_f=0.1 --min_f=2.0
        done
    done
done


############# 아래는 GB 코드 ##############

# mode="GB"

# for dataset in $datasets
# do
#     for k in $k_values
#     do
#         for scale in $scale_values
#         do
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=False
#         done
#     done
# done

# for dataset in $datasets
# do
#     for k in $k_values
#     do
#         for scale in $scale_values
#         do
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=diff --thr_init_score=$init_score --increase_bound=False
#         done
#     done
# done

# for dataset in $datasets
# do
#     for k in $k_values
#     do
#         for scale in $scale_values
#         do
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True
#         done
#     done
# done

# for dataset in $datasets
# do
#     for k in $k_values
#     do
#         for scale in $scale_values
#         do
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=diff --thr_init_score=$init_score --increase_bound=True
#         done
#     done
# done

############ standard ##############
# mode="standard"
# for dataset in $datasets
# do
#     python main.py --dataset=$dataset --mode=$mode --model=$model
# done

# mode="auto"
# for dataset in $datasets
# do
#     python main.py --dataset=$dataset --mode=$mode --model=$model --max_f=0.1 --min_f=1.0
#     python main.py --dataset=$dataset --mode=$mode --model=$model --max_f=0.1 --min_f=2.0
#     python main.py --dataset=$dataset --mode=$mode --model=$model --max_f=0.1 --min_f=4.0
#     python main.py --dataset=$dataset --mode=$mode --model=$model --max_f=0.2 --min_f=1.0
#     python main.py --dataset=$dataset --mode=$mode --model=$model --max_f=0.2 --min_f=2.0
#     python main.py --dataset=$dataset --mode=$mode --model=$model --max_f=0.2 --min_f=4.0
#     python main.py --dataset=$dataset --mode=$mode --model=$model --max_f=0.05 --min_f=1.0
#     python main.py --dataset=$dataset --mode=$mode --model=$model --max_f=0.05 --min_f=2.0
#     python main.py --dataset=$dataset --mode=$mode --model=$model --max_f=0.05 --min_f=4.0
# done