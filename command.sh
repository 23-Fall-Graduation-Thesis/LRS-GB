# ResNet18, Baseline (standard, AutoLR)
datasets="cifar100 cub cars"

for dataset in $datasets
do
    python main.py --dataset=$dataset --model=resnet18 --mode=standard
    python main.py --dataset=$dataset --model=resnet18 --mode=auto --max_f=0.1 --min_f=1.0
    python main.py --dataset=$dataset --model=resnet18 --mode=auto --max_f=0.1 --min_f=2.0
done

python hptune.py --dataset=cifar100 --model=resnet18 --mode=GBweva --MIN_K=1.0 --MAX_K=40.0 --MIN_scale_factor=1.0 --MAX_scale_factor=5.0 --max-evals=20
python hptune.py --dataset=cifar100 --model=resnet18 --mode=GBweva --MIN_K=1.0 --MAX_K=40.0 --MIN_scale_factor=1.0 --MAX_scale_factor=5.0 --max-evals=20
'python', 'main.py', '--dataset=cifar100', '--model=resnet18', '--mode=GB', '--epoch=30', '--increase_bound=False', '--norm=L2', '--opt=True', '--K=10.018596198596681', '--scale_factor=4.649464041687234'