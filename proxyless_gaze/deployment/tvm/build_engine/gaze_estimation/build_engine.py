import tvm
from tvm import relay
import onnx
from tvm import auto_scheduler
model_path = "./onnx_models/gaze-sim-5.67.onnx"
target = 'llvm -mtriple=aarch64-linux-gnu' # If you want to deploy the model on cuda, then target='cuda'
target_host = 'llvm -mtriple=aarch64-linux-gnu'


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


onnx_model = onnx.load(model_path)
relay_module, params = relay.frontend.from_onnx(onnx_model, {"left_eye": (1,3,60,60), "right_eye": (1,3,60,60), "face": (1,3,120,120)})
relay_module = convert_to_nhwc(relay_module)


with auto_scheduler.ApplyHistoryBest('./log.json'):
    with tvm.transform.PassContext(opt_level=4, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(relay_module, target, params=params, target_host=target_host)
# Export the binary lib
lib.export_library('./gaze.tar')
print("Success")
