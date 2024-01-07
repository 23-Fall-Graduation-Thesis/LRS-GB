# LRS_GB
### *Learning Rate Schedule for Fine-Tuning with Generalization Bound Guarantees*
2024-1 Graduation Project code (Dept.of CSE, Konkuk Univ)  
> BaseLine : [AutoLR](https://github.com/youngminPIL/AutoLR)

## Member
- [Vaughan](https://github.com/webb-c)
- [UiJin](https://github.com/youuijin)
- [HyunHee](https://github.com/aesa117)

## Commit Rule
- `add` : 새로운 기능 추가
- `remove` : 기능 삭제  
- `fix` : 오류 수정  
- `refactor` : 코드, 폴더 리팩토링
- `test` : 테스트 코드 관련 수정
- `etc` : 그 외 모든 수정

## Commands
프로젝트를 실행하려면 다음 명령어를 사용하세요.

```bash
python main.py [Common Options]
```

|Common Options|Arguments|Default|Description|
|-------|---------|-------|-----------|
|`--dataset`|`cifar10`, `cifar100`, `svhn`, `cub`|`cifar10`|dataset to use for learning|
|`--model`|`Conv4`, `resnet18`, `resnet34`,<br> `resnet50`, `resnet101`, `resnet152`,<br> `alexnet`, `vgg16`, `vgg19`, `WRN50`, `WRN101`|`alexnet`| CNN Model Architecture to use for learning|
|`--batch_size`|**int type**|`64`|size of mini-batch size|
|`--epoch`|**int type**|`50`|Training epochs|
|`--lr`|**float type**|`0.001`|(initial) learning rate|
|`--device`|**int type**|`0`|id of GPU to use|
|`--pretrain`|`True`, `False`|`False`|Training mode, pretrain or fine-tune|

### Model Pretraining

```bash
python main.py --pretrain=True [Common Options]
```

### Model Fine-Tuning
```bash
python main.py --pretrain=False [Fine-tuning Options] [Common Options]
```
|Fine-tuning Options|Arguments|Default|Description|
|-------|---------|-------|-----------|
|`--mode`|`standard`, `auto`, `ours`|`standard`|Fine-tuning mode to use, Standard(standard), AutoLR(auto), Our new method(ours)|
|`--model_path`|**str type**|`''`|path of pretrained model, <br> if `''`, use ImageNet provided by torchvision|
|`--max_f`|**int type**|`0.4`|hyperparameter for AutoLR|
|`--min_f`|**int type**|`2`|hyperparameter for AutoLR|
