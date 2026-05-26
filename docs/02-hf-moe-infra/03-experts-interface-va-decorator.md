---
title: ExpertsInterface và decorator
---

# `ExpertsInterface` và decorator

Chương này đi sâu vào hai abstraction trung tâm: `ExpertsInterface` (registry) và `use_experts_implementation` (decorator). Khi đọc bất kỳ model MoE 2024+ nào trong Phần 3, hai thứ này xuất hiện đầu file. Hiểu chúng đầu tiên.

## `GeneralInterface` parent class

`ExpertsInterface` thừa kế `GeneralInterface` từ `src/transformers/utils/generic.py`:

```python
class GeneralInterface(MutableMapping):
    """Generic interface for registering arbitrary mappings (e.g., functions, classes)."""

    _global_mapping: dict[str, Any] = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        if key in self._local_mapping:
            return self._local_mapping[key]
        if key in self._global_mapping:
            return self._global_mapping[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        self._local_mapping[key] = value

    def register(self, key: str, value):
        self._local_mapping[key] = value

    ...
```

(Lược trích.)

Pattern hai tầng: `_global_mapping` (class-level, shared) và `_local_mapping` (instance-level, user register thêm).

Mọi interface trong HF (Attention, Experts) đều dùng pattern này.

## `ExpertsInterface` cụ thể

```python
class ExpertsInterface(GeneralInterface):
    """Interface for registering custom experts forward functions."""

    _global_mapping = {
        "batched_mm": batched_mm_experts_forward,
        "grouped_mm": grouped_mm_experts_forward,
        "sonicmoe": sonicmoe_experts_forward,
    }

    def get_interface(self, experts_implementation: str, default: Callable) -> Callable:
        """Return the requested `experts_implementation`. Strictly check validity."""
        if experts_implementation is None:
            logger.warning_once(
                "You tried to access the `ExpertsInterface` with a `config._experts_implementation` "
                "set to `None`. This is expected if you use an Expert Module as a standalone Module..."
            )
        elif experts_implementation != "eager" and experts_implementation not in self:
            raise KeyError(
                f"`{experts_implementation}` is not a valid experts implementation registered in `ExpertsInterface`"
            )
        return super().get(experts_implementation, default)
```

(`src/transformers/integrations/moe.py`, class `ExpertsInterface`.)

Ba backend default:

1. `batched_mm`: `batched_mm_experts_forward` (xem Chương 2).
2. `grouped_mm`: `grouped_mm_experts_forward`.
3. `sonicmoe`: custom kernel từ IBM/Mosaic (experimental).

Validation:

1. `None` -> warning, fall back to default (original eager forward).
2. `"eager"` -> không lookup, dùng default forward.
3. String khác chưa register -> raise.
4. String đã register -> return function.

## Singleton

```python
ALL_EXPERTS_FUNCTIONS = ExpertsInterface()
```

Một instance dùng chung toàn module transformers. Tương đương `ALL_ATTENTION_FUNCTIONS` cho attention.

User register custom:

```python
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

def my_custom_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    ...

ALL_EXPERTS_FUNCTIONS.register("my_custom", my_custom_experts_forward)

# Use it
model.config._experts_implementation = "my_custom"
```

Cách extend mà không touch HF code.

## Cấu trúc decorator

```python
def use_experts_implementation(
    experts_class: type[torch.nn.Module] | None = None,
    *,
    experts_interface: ExpertsInterface = ALL_EXPERTS_FUNCTIONS,
    is_concatenated: bool = True,
    is_transposed: bool = False,
    has_bias: bool = False,
    has_gate: bool = True,
) -> type[torch.nn.Module]:
```

Decorator có thể dùng hai cách:

```python
# Cách 1: dùng default flags
@use_experts_implementation
class MixtralExperts(nn.Module):
    ...

# Cách 2: specify flags
@use_experts_implementation(is_concatenated=False, has_bias=True, is_transposed=True)
class GptOssExperts(nn.Module):
    ...
```

Khi không có argument, `experts_class` = class được decorate, function trả về class đã modify. Khi có argument, function trả về một decorator (closure) để Python áp dụng tiếp.

## Modify class

```python
def wrapper(experts_class: type[torch.nn.Module]) -> type[torch.nn.Module]:
    original_init = experts_class.__init__
    original_forward = experts_class.forward

    @wraps(original_init)
    def __init__(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        self.config = config
        self.has_gate = has_gate
        self.has_bias = has_bias
        self.is_transposed = is_transposed
        self.is_concatenated = is_concatenated

    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        experts_forward = experts_interface.get_interface(
            self.config._experts_implementation, original_forward
        )
        return experts_forward(self, *args, **kwargs)

    if not hasattr(experts_class, "_apply_gate"):
        experts_class._apply_gate = _default_apply_gate

    experts_class.__init__ = __init__
    experts_class.forward = forward
    return experts_class
```

Ba thay đổi:

**1. `__init__` wrap**: gọi original_init xong, attach các flag (`has_gate`, `has_bias`, ...) lên instance. Flag được closure capture từ decorator argument.

**2. `forward` wrap**: thay vì gọi original (eager loop), lookup interface dispatch. Nếu lookup fail (return default), gọi original.

**3. `_apply_gate` fallback**: nếu class chưa define `_apply_gate`, gán `_default_apply_gate`. Hữu ích cho expert custom gating (ví dụ GPT-OSS có clamp đặc biệt, override `_apply_gate`).

Sau decorator, class có:

- `instance.has_gate`, `.has_bias`, `.is_transposed`, `.is_concatenated`, `.config`.
- `instance.forward(...)` dispatch qua interface.
- `instance._apply_gate(gate_up_out)` để combine.

## `_default_apply_gate`

```python
def _default_apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
    """Default gating: split, activate gate, multiply with up."""
    gate, up = gate_up_out.chunk(2, dim=-1)
    return self.act_fn(gate) * up
```

SwiGLU mặc định. Phần lớn model dùng. Một số model override:

```python
class GptOssExperts(nn.Module):
    ...
    def _apply_gate(self, gate_up_out):
        gate, up = gate_up_out.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * (up + 1)  # custom expression
```

(Pseudocode dựa trên GPT-OSS code.)

GPT-OSS có clamp + offset, không default. Class define `_apply_gate` riêng, decorator bypass fallback.

## Flag interpretation

Decoder hay nhầm các flag. Bảng giải thích:

**`is_concatenated`**:

- `True` (Mixtral, Qwen3-MoE, DeepSeek-V3): `gate_up_proj` là một tensor `(E, 2*intermediate, hidden)`. Linear single, chunk thành (gate, up).
- `False` (một số model legacy): `gate_proj` và `up_proj` riêng, hai tensor `(E, intermediate, hidden)`. Hai linear call.

Lợi của concatenated: tiết kiệm 1 linear call mỗi forward. Lý do model legacy chưa concat: thường vì checkpoint history hoặc compatibility.

**`is_transposed`**:

- `False` (Mixtral default): weight shape `(E, out_dim, in_dim)`. Dùng `F.linear(x, w)` convention.
- `True` (GPT-OSS): weight shape `(E, in_dim, out_dim)`. Dùng matmul native `x @ w`.

`F.linear` internally `x @ w.T`, nên `(out, in)` shape. Transposed format `(in, out)` cần matmul direct. Lý do: một số kernel (như `torch._grouped_mm`) yêu cầu specific layout.

**`has_bias`**:

- `False` (default): expert linear không bias. Standard cho LLM (Llama, Mixtral, ...).
- `True` (GPT-OSS): bias cho gate_up + down. OpenAI design decision.

Bias trong expert có thể giúp với 4-bit quantization (MXFP4), tránh accuracy drop.

**`has_gate`**:

- `True` (default): expert có gating SwiGLU/GeGLU.
- `False`: chỉ `act(up_proj(x))`. Hiếm, dùng cho ReLU MoE (ST-MoE) hoặc kiến trúc khác.

## Lifecycle khi đọc forward

Khi `LlamaForCausalLM.forward(...)` chạy đến `decoder_layer.mlp(hidden_states)` mà `mlp` là `MixtralSparseMoeBlock`:

```
MixtralSparseMoeBlock.forward
  ├── (jitter noise nếu training)
  ├── gate(hidden_states) -> top_k_weights, top_k_index
  └── experts(hidden_states, top_k_index, top_k_weights)
       │  # experts là MixtralExperts đã decorated
       │
       ├── (decorator wrap) forward
       │   ├── interface.get_interface(config._experts_implementation, original_forward)
       │   │     └── return batched_mm_experts_forward OR grouped_mm_experts_forward OR original
       │   └── return chosen_forward(self, hidden_states, top_k_index, top_k_weights)
       │
       └── (Inside chosen forward)
           ├── access self.gate_up_proj, self.down_proj (3D tensors)
           ├── access self.has_gate, self.has_bias, etc. (flags)
           ├── call self._apply_gate(gate_up_out)
           └── return final_hidden_states
```

Hai indirection: decorator dispatch -> interface lookup -> chosen backend. Mỗi indirection thêm function call (~µs), không matter.

## Khi default forward vẫn được dùng

Default forward (eager loop) được giữ trong class:

```python
class MixtralExperts(nn.Module):
    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Original eager forward
        final_hidden_states = torch.zeros_like(hidden_states)
        ...
        for expert_idx in expert_hit:
            ...
        return final_hidden_states
```

Decorator wrap forward này. Khi config `_experts_implementation = "eager"` hoặc `None`, decorator dispatch về **default** = original forward = loop.

Lý do giữ original:

1. **Debug**: loop dễ inspect bằng pdb hơn batched/grouped backend.
2. **Fallback**: nếu PyTorch/kernel không support, eager always work.
3. **Test**: aux loss và bahavior verify qua eager trước khi compare với batched/grouped.

## Pitfall

**1. Quên `@use_experts_implementation`**: class chạy được nhưng không có `config`, `has_gate`, `has_bias` attribute. Crash khi gọi `_apply_gate`. Phải dùng decorator.

**2. Decorator argument sai**: ví dụ class concatenate gate_up nhưng pass `is_concatenated=False`. Forward chạy nhưng weight layout không khớp -> wrong math.

**3. Override `_apply_gate` không tương thích với batched_mm**: nếu custom logic giả định eager input format, batched format khác (extra dim). Phải test cả ba mode.

**4. `config._experts_implementation` không set**: model load với `from_pretrained` thường set theo default ("eager" hoặc "batched_mm"). Nếu user instance manually, phải set explicit.

**5. Register custom backend nhưng quên `register_fake`**: nếu backend là custom op, Dynamo cần fake function cho shape inference khi compile. Quên gây compile error khó debug.

Chương sau ta so sánh `batched_mm` và `grouped_mm` chi tiết.
