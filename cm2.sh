# mode="autoGB auto GB standard"
mode="GBweva"
model="resnet50"
datasets="cifar100 cub cars"
k_values="1.95 2.6 3.9 7.8 15.6 23.4 31.2"
scale_values="1.25 1.5 2"
init_score="0.97"

############### GB weva hyperOPT #################
# for dataset in $datasets
# do
#     python hptune.py --epoch=50 --dataset=$dataset --model=$model --mode=$mode --MIN_K=1.0 --MAX_K=40.0 --MIN_scale_factor=1.0 --MAX_scale_factor=5.0 --max-evals=20
# done

python main.py --epoch=50 --dataset="cub" --model=$model --mode=$mode --K=11.749325261051407 --scale_factor=3.2781245211344094


# for dataset in $datasets
# do
#     python main.py --dataset=$dataset --mode=$mode --model=$model --K=15.6 --scale_factor=2 --bound=weva --thr_init_score=$init_score --increase_bound=False --max_f=0.1 --min_f=1.0
#     python main.py --dataset=$dataset --mode=$mode --model=$model --K=15.6 --scale_factor=2 --bound=weva --thr_init_score=$init_score --increase_bound=False --max_f=0.1 --min_f=2.0
# done

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

# for dataset in $datasets
# do
#     for k in $k_values
#     do
#         for scale in $scale_values
#         do
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True --K_mode=5 --max_f=0.1 --min_f=1.0
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True --K_mode=5 --max_f=0.1 --min_f=2.0
#         done
#     done
# done

# for dataset in $datasets
# do
#     for k in $k_values
#     do
#         for scale in $scale_values
#         do
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True --K_mode=5 --max_f=0.1 --min_f=1.0
#             python main.py --dataset=$dataset --mode=$mode --model=$model --K=$k --scale_factor=$scale --bound=weva --thr_init_score=$init_score --increase_bound=True --K_mode=5 --max_f=0.1 --min_f=2.0
#         done
#     done
# done


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