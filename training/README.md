# ProxylessNAS-Training-PyTorch

The implementation is based on PyTorch (>= 1.0) and Horovod (0.15.4). Paper claimed accuracy can be reproduced by following scripts.

```
mpirun -np 8 \
    -H localhost:8\
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python main.py --arch proxyless_gpu --fp16-allreduce \
        --color-jitter --label-smoothing --epochs 300 
```

Arch choices:
    proxyless_cpu, proxyless_gpu, proxyless_mobile, proxyless_mobile_14
