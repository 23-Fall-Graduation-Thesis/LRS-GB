import argparse
from hyperopt import fmin, hp, tpe, Trials
import subprocess as sp
from utils.utils import str2bool


def arg_parse(parser):
    parser = argparse.ArgumentParser()
    
    # Common Options
    parser.add_argument('--epoch', type=int, default=30, help='Epoch')
    parser.add_argument('--dataset', default='cifar10', help='Dataset type')
    parser.add_argument('--model', default='resnet18', help='Model type')

    parser.add_argument('--mode', type=str, default='standard', help='Standard(standard), LRS-GB(GB), AutoLR(auto), Auto-start-GB(autoGB)')
    parser.add_argument('--increase_bound', type=str2bool, default=False, help='')
    parser.add_argument('--bound', default='weva', type=str, help='diff or weva')
    parser.add_argument('--norm', type=str, default='L2', help='weight calculation using L1 norm or L2 norm')
    
    parser.add_argument('--thr_score', default=0.94, type=float, help='score threshold for AutoLR')
    parser.add_argument('--thr_init_score', default=0.97, type=float, help='score threshold for LRS')
    
    # NOTE
    # Hptune Options
    parser.add_argument('--MIN_max_f', default=0.0, type=float, help='range of max_f for hpopt')
    parser.add_argument('--MAX_max_f', default=3.0, type=float, help='range of max_f for hpopt')
    parser.add_argument('--MIN_min_f', default=0.0, type=float, help='range of min_f for hpopt')
    parser.add_argument('--MAX_min_f', default=3.0, type=float, help='range of max_f for hpopt')

    parser.add_argument('--MIN_K', default=0.1, type=float, help='range of Lipschitz constant') 
    parser.add_argument('--MAX_K', default=20.0, type=float, help='range of Lipschitz constant')
    parser.add_argument('--MIN_scale_factor', default=1.0, type=float, help='range of layer-wise constraint scaling')
    parser.add_argument('--MAX_scale_factor', default=10.0, type=float, help='range of layer-wise constraint scaling')

    parser.add_argument("--max-evals", dest="max_evals", action="store", default="20")
    
    return parser.parse_args()


def objective(search_space):
    global args
    cmd = ["python", "main.py", "--dataset=" + args.dataset, "--model=" + args.model, "--mode=" + args.mode, "--epoch=" + str(args.epoch), "--increase_bound=" + str(args.increase_bound), "--bound=" + args.bound, "--norm=" + args.norm]
    cmd.append("--opt=" + str(True))
    if args.mode == "autoGB":
        cmd.append("--min_f=" + str(1.0)) # or 2.0 #TODO
        cmd.append("--max_f-" + str(0.1))
    
    if args.mode == "auto":
        cmd.append("--min_f=" + str(search_space["min_f"]))
        cmd.append("--max_f=" + str(search_space["max_f"]))
    else:
        cmd.append("--K=" + str(search_space["K"]))
        cmd.append("--scale_factor=" + str(search_space["scale_factor"]))
    
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL, universal_newlines=True)
    loss = 0.0

    line = None
    for raw_line in proc.stdout:
        # print(type(raw_line))
        line = raw_line.strip()

    if line is not None and line.startswith("t"):
        fields = line.split(':')
        loss = -float(fields[-1])  # test accuracy

    print(str(loss) + ": " + str(search_space))
    return loss

if __name__ == '__main__':
    # arguments parsing
    global args
    args = arg_parse(argparse.ArgumentParser())
    if args.mode == "autoGB":
        min_f=1.0 # or 2.0
        max_f=0.1

    space = {}
    if args.mode == "auto":
        space["min_f"] = hp.loguniform("min_f", args.MIN_min_f, args.MAX_min_f)
        space["max_f"] = hp.loguniform("max_f", args.MIN_max_f, args.MAX_max_f)
    else:
        space["K"] = hp.loguniform("K", args.MIN_K, args.MAX_K)
        space["scale_factor"] = hp.loguniform("scale_factor", args.MIN_scale_factor, args.MAX_scale_factor)
    
    trials = Trials()
    best_params = fmin(objective, space=space, algo=tpe.suggest, max_evals=int(args.max_evals), trials=trials)

    print(best_params)