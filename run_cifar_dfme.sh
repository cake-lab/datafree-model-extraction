cd dfme;

python3 train.py --dataset cifar10 --ckpt checkpoint/teacher/cifar10-resnet34_8x.pt --device 0 --grad_m 1 --query_budget 20 --log_dir save_results/cifar10  --lr_G 1e-4 --student_model resnet18_8x --loss l1;
