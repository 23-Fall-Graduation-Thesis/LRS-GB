# models="resnet50"
# datasets="cifar10 cifar100 svhn cub"

# for model in $models
# do
#     for dataset in $datasets
#     do         
#         python main.py --dataset=$dataset --model=$model --device=3 --mode=auto --max_f=0.2 --min_f=2.0
#         python main.py --dataset=$dataset --model=$model --device=3 --mode=auto --max_f=0.4 --min_f=2.0 --thr_score=0.9
#     done
# done
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=standard
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=standard

python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.2 --min_f=1.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.2 --min_f=2.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.2 --min_f=4.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.1 --min_f=1.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.1 --min_f=2.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.1 --min_f=1.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.05 --min_f=1.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.05 --min_f=2.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.05 --min_f=4.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.03 --min_f=0.5
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.03 --min_f=1.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.03 --min_f=2.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.03 --min_f=4.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=1.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=2.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=4.0
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=0.7
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=0.5
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=0.3
python main.py --model=resnet18 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=0.1

python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.2 --min_f=1.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.2 --min_f=2.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.2 --min_f=4.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.1 --min_f=1.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.1 --min_f=2.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.1 --min_f=4.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.05 --min_f=1.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.05 --min_f=2.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.05 --min_f=4.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.03 --min_f=0.5
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.03 --min_f=1.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.03 --min_f=2.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.03 --min_f=4.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=1.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=2.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=4.0
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=0.7
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=0.5
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=0.3
python main.py --model=resnet50 --dataset=cifar100 --device=3 --mode=auto --max_f=0.01 --min_f=0.1
