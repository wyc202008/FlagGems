import torch

from . import testing  # noqa: F401
from . import runtime
from .fused import *  # noqa: F403
from .ops import *  # noqa: F403
from .runtime.commom_utils import Autograd
from .runtime.register import Register

__version__ = "2.2"
device = runtime.device.name
vendor_name = runtime.device.vendor_name
aten_lib = torch.library.Library("aten", "IMPL")
registrar = Register
current_work_registrar = None
runtime.replace_customized_ops(globals())


def enable(lib=aten_lib, unused=None, registrar=registrar):
    global current_work_registrar
    current_work_registrar = registrar(
        (
            ("add.Tensor", add, Autograd.disable),
            ("addmm", addmm, Autograd.disable),
            ("arange", arange, Autograd.disable),
            ("bitwise_and.Tensor", bitwise_and_tensor, Autograd.disable),
            ("bitwise_and_.Tensor_", bitwise_and_tensor_, Autograd.disable),
            ("bitwise_and.Scalar", bitwise_and_scalar, Autograd.disable),
            ("bitwise_and_.Scalar", bitwise_and_scalar_, Autograd.disable),
            ("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor, Autograd.disable),
            ("bitwise_not", bitwise_not, Autograd.disable),
            ("bitwise_not_", bitwise_not_, Autograd.disable),
            ("bitwise_or.Tensor", bitwise_or_tensor, Autograd.disable),
            ("bitwise_or_.Tensor", bitwise_or_tensor_, Autograd.disable),
            ("bitwise_or.Scalar", bitwise_or_scalar, Autograd.disable),
            ("bitwise_or_.Scalar", bitwise_or_scalar_, Autograd.disable),
            ("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor, Autograd.disable),
            ("bmm", bmm, Autograd.disable),
            ("cos", cos, Autograd.disable),
            # #--------------------------------------------------------------
            ("div.Tensor", true_divide, Autograd.disable),
            ("div_.Tensor", true_divide_, Autograd.disable),
            ("div.Scalar", true_divide, Autograd.disable),
            ("div_.Scalar", true_divide_, Autograd.disable),
            ("div.Tensor_mode", div_mode, Autograd.disable),
            ("div_.Tensor_mode", div_mode_, Autograd.disable),
            ("div.Scalar_mode", div_mode, Autograd.disable),
            ("div_.Scalar_mode", div_mode_, Autograd.disable),
            (
                "divide.Tensor",
                true_divide,
                Autograd.disable,
            ),  # divide, an alias for div
            (
                "divide_.Tensor",
                true_divide_,
                Autograd.disable,
            ),  # divide, an alias for div
            ("divide.Scalar", true_divide, Autograd.disable),
            ("divide_.Scalar", true_divide_, Autograd.disable),
            ("divide.Tensor_mode", div_mode, Autograd.disable),
            ("divide_.Tensor_mode", div_mode_, Autograd.disable),
            ("divide.Scalar_mode", div_mode, Autograd.disable),
            ("divide_.Scalar_mode", div_mode_, Autograd.disable),
            (
                "true_divide.Tensor",
                true_divide,
                Autograd.disable,
            ),  # true_divide, an alias for div
            (
                "true_divide_.Tensor",
                true_divide_,
                Autograd.disable,
            ),  # true_divide, an alias for div
            ("true_divide.Scalar", true_divide, Autograd.disable),
            ("true_divide_.Scalar", true_divide_, Autograd.disable),
            ("floor_divide", floor_divide, Autograd.disable),
            ("floor_divide_.Tensor", floor_divide_, Autograd.disable),
            ("floor_divide.Scalar", floor_divide, Autograd.disable),
            ("floor_divide_.Scalar", floor_divide_, Autograd.disable),
            ("remainder.Tensor", remainder, Autograd.disable),
            ("remainder_.Tensor", remainder_, Autograd.disable),
            ("remainder.Scalar", remainder, Autograd.disable),
            ("remainder_.Scalar", remainder_, Autograd.disable),
            ("remainder.Scalar_Tensor", remainder, Autograd.disable),
            ("embedding", embedding, Autograd.enable),
            ("eq.Tensor", eq, Autograd.disable),
            ("eq.Scalar", eq_scalar, Autograd.disable),
            ("exponential_", exponential_, Autograd.disable),
            ("ge.Tensor", ge, Autograd.disable),
            ("ge.Scalar", ge_scalar, Autograd.disable),
            ("isin.Tensor_Tensor", isin, Autograd.disable),
            ("isin.Scalar_Tensor", isin, Autograd.disable),
            ("isin.Tensor_Scalar", isin, Autograd.disable),
            #-----------------------part2----------------------------------
            ("le.Tensor", le, Autograd.disable),
            ("le.Scalar", le_scalar, Autograd.disable),
            ("lt.Tensor", lt, Autograd.disable),
            ("lt.Scalar", lt_scalar, Autograd.disable),
            ("zeros", zeros, Autograd.disable),
            ("full", full, Autograd.disable),
            ("ones_like", ones_like, Autograd.disable),
            ("resolve_conj", resolve_conj, Autograd.disable),
            ("mean", mean, Autograd.disable),
            ("mm", mm, Autograd.disable),
            ("mul.Tensor", mul, Autograd.disable),
            ("mul_.Tensor", mul_, Autograd.disable),
            ("multinomial", multinomial, Autograd.disable),
            ("neg", neg, Autograd.disable),
            ("neg_", neg_, Autograd.disable),
            ("rsqrt", rsqrt, Autograd.disable),
            ("rsqrt_", rsqrt_, Autograd.disable),
            ("silu", silu, Autograd.enable),
            ("silu_", silu_, Autograd.enable),
            ("sin", sin, Autograd.disable),
            ("sin_", sin_, Autograd.disable),
            ("softmax.int", softmax, Autograd.enable),
            ("sub.Tensor", sub, Autograd.disable),
            ("sub_.Tensor", sub_, Autograd.disable),
            ("max", max, Autograd.disable),
            ("max.dim", max_dim, Autograd.disable),
            ("min", min, Autograd.disable),
            ("min.dim", min_dim, Autograd.disable),
            ("sum", sum, Autograd.disable),
            ("all", all, Autograd.disable),
            ("any", any, Autograd.disable),
            
            ("scatter.src", scatter, Autograd.disable),
            ("scatter.reduce", scatter, Autograd.disable),

            #("index_select", index_select, Autograd.disable),
            ("masked_fill.Tensor", masked_fill, Autograd.disable),
            ("masked_fill.Scalar", masked_fill, Autograd.disable),
            ("masked_fill_.Tensor", masked_fill_, Autograd.disable),
            ("masked_fill_.Scalar", masked_fill_, Autograd.disable),
            ("cat", cat, Autograd.disable),
            #------------------------------issue--------------------------------
            # ("log_softmax.int", log_softmax, Autograd.enable),
            # ("cumsum", cumsum, Autograd.disable),
            # ("gather", gather, Autograd.disable),
            # (
            #     "scaled_dot_product_attention",
            #     scaled_dot_product_attention,
            #     Autograd.disable,
            # ),
        ),
        user_unused_ops_list=[] if unused is None else unused,
        lib=lib,
    )


class use_gems:
    def __init__(self, unused=None):
        self.lib = torch.library.Library("aten", "IMPL")
        self.unused = [] if unused is None else unused
        self.registrar = Register

    def __enter__(self):
        enable(lib=self.lib, unused=self.unused, registrar=self.registrar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global current_work_registrar
        del self.lib
        del self.unused
        del self.registrar
        del current_work_registrar


def all_ops():
    return current_work_registrar.get_all_ops()


__all__ = [
    "enable",
    "use_gems",
]
