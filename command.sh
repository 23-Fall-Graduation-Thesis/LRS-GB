# mode="autoGB auto GB standard"
mode="GB"
model="resnet18"
#datasets="cifar100 cub cars"
dataset="cifar100"

python hptune.py --dataset=$dataset --mode=$mode --model=$model --bound=weva --increase_bound=False
