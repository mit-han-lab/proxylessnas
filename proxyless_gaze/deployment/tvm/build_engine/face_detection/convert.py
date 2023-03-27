import tvm
from tvm import relay
import onnx
from tvm import auto_scheduler
import torch

def convert_to_nhwc(mod):
    """Convert to NHWC layout"""
    desired_layouts = {"nn.conv2d": ["NHWC", "default"],
                       "qnn.conv2d": ["NHWC", "default"]}
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )
    with tvm.transform.PassContext(opt_level=4):
        mod = seq(mod)
    return mod



model_path = "./onnx_models/yolox.torchscript.pt"
device_key = 'rasp4b'
remote_host = "g1.mit.edu"
remote_port = 9190
resume = True

torch_model = torch.jit.load(model_path)
relay_module, params = relay.frontend.from_pytorch(torch_model, [('input0', [1, 3, 160, 128])])
relay_module = convert_to_nhwc(relay_module)

runner = auto_scheduler.RPCRunner(
        key=device_key,
        host=remote_host,
        port=remote_port,
        timeout=300,  # Set larger. It is easy to timeout if this is small when the network connection is unstable!
        repeat=1,
        min_repeat_ms=200,
        enable_cpu_cache_flush=True,
        n_parallel=13  # The number of devices for parallel tuning. You could set to the free Raspberry Pis you registered to the tracker!
    )

target = 'llvm -mtriple=aarch64-linux-gnu' # If you want to deploy the model on cuda, then target='cuda'
target_host = 'llvm -mtriple=aarch64-linux-gnu'

tasks, task_weights = auto_scheduler.extract_tasks(relay_module['main'], params, target, target_host=tvm.target.Target(target, host=target_host), opt_level=4)
for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)


print(remote_host)
print(remote_port)
print("number of task:", len(tasks))

if not resume:
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights) # You could resume the tuning here by pass the argument load_log_file. Just pass the path to the json log file here.
else:
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file="./log.json")
    print("load log file")
tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=len(tasks)*400,  # Typically set this number to num_tasks*800, e.g., 31*800=24800 for MobileNetV2. I set to 200 for demo use.
        runner=runner,
        measure_callbacks=[auto_scheduler.RecordToFile('./log.json')],
    )


tuner.tune(tune_option)

# with auto_scheduler.ApplyHistoryBest('./log.json'):
#     with tvm.transform.PassContext(opt_level=4, config={"relay.backend.use_auto_scheduler": True}):
#         lib = relay.build(relay_module, target, params=params, target_host=target_host)
# # Export the binary lib
# lib.export_library('./gaze.tar')
# print("Success")
