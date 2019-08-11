import torch
import horovod.torch.compression
from horovod.torch.compression import Compressor

percent = 0.1


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    @staticmethod
    def compress(tensor):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed


class DGC(Compressor):
    @staticmethod
    def compress(tensor):
        # aux information
        percent = 0.1
        origin_dtype = tensor.dtype
        total_nums = tensor.nelement()
        send_nums = int(total_nums * percent)

        # get topk items for sending
        t_shape = tensor.size()
        flat_t = tensor.view(-1)
        abs_t = torch.abs(flat_t)
        topk_result = torch.topk(abs_t, send_nums)
        topk_t, topk_mask = topk_result
        # threashold = torch.min(topk_t)
        topk_t_raw = flat_t[topk_mask]

        # send sparse tensor only
        ii = topk_mask.view(-1, 1)
        vv = topk_t_raw
        # torch.sparse.FloatTensor(ii.t(), vv, torch.Size([75, ])).to_dense()
        sparse_tensor = torch.sparse.FloatTensor(ii.t(), vv ,torch.Size([flat_t.nelement(),]) )
        return sparse_tensor, origin_dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        return tensor.to_dense()


class VarianceBasedGC(Compressor):

    @staticmethod
    def compress(tensor):
        pass

    @staticmethod
    def decompress(tensor, ctx):
        pass




class INT8Compressor(Compressor):
    @staticmethod
    def compress(tensor):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        pass
