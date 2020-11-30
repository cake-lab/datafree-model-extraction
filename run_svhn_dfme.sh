cd dfme;

python3 train.py --dataset svhn --ckpt checkpoint/teacher/svhn-resnet34_8x.pt --device 0 --grad_m 1 --query_budget 2 --log_dir save_results/svhn  --lr_G 5e-5 --student_model resnet18_8x --loss l1 --steps 0.5 0.8;