---
title: "MiniSGL Engine"
date: 2025-12-28
author: "Zikai Wang"
description: "The engine of MiniSGL."
summary: "The engine of MiniSGL."
---
# Introduction
[[MiniSGL Walk Through]]
The Engine is an orchestration layer, it brings together the KVCache, model and attention backend. It receives the request batches from the scheduler, run the forward and sampling, and return the results to the scheduler.
A basic workflow is:
```shell
Scheduler.step()
    │
    ├── 1. Select requests (prefill or decode)
    ├── 2. Allocate memory slots
    ├── 3. Build Batch object
    ├── 4. Prepare sampling args
    │
    └── 5. engine.forward_batch(batch, args)  ──────────►  Engine
                                                              │
                                                              ├── model.forward()
                                                              ├── sample tokens
                                                              └── return ForwardOutput
                                                              
    ◄───────────────────────────────────────────────────────────┘
    │
    ├── 6. Write output tokens
    └── 7. Update request states
```
# Engine Configuration
`python/minisgl/engine/config.py`
```python
@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:23333"
```
Some key configurations:
- `max_running_req`: Max concurrent requests (256)
- `memory_ratio`: How much GPU memory to use for KV cache (90%)
- `cuda_graph_bs`: Batch sizes to capture CUDA graphs for
- `use_pynccl`: Use custom NCCL wrapper for multi-GPU

# The Main Engine
`python/minisgl/engine/engine.py`
### Engine Initialization
It initializes the engine with the configuration, creates the model, KV cache, page table, attention backend, context, sampler and graph runner.
##### Tensor Parallelism Setup
Load engine configuration and model configuration. Sets global TP (Tensor Parallelism) info so all components know:
- `rank`: Which GPU this process controls (0, 1, 2, ...)
- `size`: Total number of GPUs
```python
class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.model_config = config.model_config
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
```
The distribution information is set with a predefined global `DistributedInfo` object.
```python
@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0
# Global TP info
_TP_INFO: DistributedInfo | None = None

# Set global TP info: current rank and world size 
def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)
```

##### CUDA Device and Stream Setup
Initialize CUDA device and stream for this process. The device is set to the current rank, and we use a dedicated stream to overlap CPU work with GPU work on engines.
```python
        # make sure CUDA is not initialized yet
        assert not torch.cuda.is_initialized()
        # set the device to the current rank
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        # set the device to the current rank
        torch.cuda.set_device(self.device)
        # create a dedicated stream to overlap CPU work with GPU work on engines
        self.stream = torch.cuda.Stream()
        # set the stream to the current stream
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
```

##### Communication Setup
Initialize the communication group for this process.
```python
        self.tp_cpu_group = self._init_communication()
```
It calls the init_communication function to initialize the communication group for this process.
```python
    def _init_communication(self) -> torch.distributed.ProcessGroup:
        config = self.config
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            if config.use_pynccl:
                max_bytes = (
                    config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
                )
                enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group
```
MiniSGL supports two communication backends:
- `PyNCCL`: Use custom NCCL wrapper
- `NCCL`: Use native NCCL for multi-GPU
Gloo is used for CPU communication, in NCCL mode, it uses a separate group for gloo, and in PyNCCL mode, it uses the same group for both CPU and GPU. See [[MiniSGL Communication]] for more details.

##### Get Initial Free Memory
```python
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")
```
It uses the `_sync_get_memory` function to get the initial free memory among all GPUs, it is the maximum free memory among all GPUs (see below, and why we use the maximum free memory here?).
It first synchronizes all GPUs to make sure that it can access a consistent snapshot of the available memory across all GPUs. 
```python
    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
```
This snippet is used to clean GPU states before measuring the free memory.
- `synchronize`: An enforced barrier to wait for all GPU operations to complete, ensuring no pending ops that might allocate memory
- `empty_cache`: PyTorch caches freed memory; this releases it back to CUDA
- `reset_peak_memory_stats`: Clean slate for future measurements

Then it gets local free memory.
```python
        free_memory = _get_free_memory(self.device)
```
Then it performs a min/max reduction across all GPUs to get the minimum and maximum free memory. It first creates a tensor with the local free memory (for min) and the negative of the local free memory (for max), and then performs a min reduction across all GPUs. 
The first element of the tensor is the minimum free memory, and the second element is the negative of the maximum free memory. We get the minimum and maximum free memory with only one all-reduce operation.
It also checks if the memory is imbalanced across all GPUs by checking if the difference between the maximum and minimum free memory is greater than a threshold (2GB here). The imbalanced memory might indicate problems with the environment, like different GPU models or CUDA memory leaks, and the bottleneck will be the smallest GPU.
```python
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory
```
An example of the all-reduce operation:
```shell
┌───────────────────────────────────────────────────────┐
│  GPU 0: 40 GB free                                    │
│  GPU 1: 35 GB free                                    │
│  GPU 2: 42 GB free                                    │
├───────────────────────────────────────────────────────┤
│                                                       │
│  Before all_reduce:                                   │
│    GPU 0: [40, -40]                                   │
│    GPU 1: [35, -35]                                   │
│    GPU 2: [42, -42]                                   │
│                                                       │
│  After all_reduce with MIN:                           │
│    All GPUs: [35, -42]                                │
│               ↑    ↑                                  │
│             min  min of negatives = -max              │
│                                                       │
│  Result:                                              │
│    min_free = 35                                      │
│    max_free = -(-42) = 42                             │
│                                                       │
└───────────────────────────────────────────────────────┘
```
> Why CPU tensor? Because 1. the tensor is quite small, not worth the GPU communication overhead. 2. Gloo is used for CPU communication, and it can even work before NCCL is initialized. 3. Initilization itself is not performance critical, so it is okay to use CPU tensor.

##### Load Model
```python
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_path, config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict())
```
It first sets the rope device to the current GPU device, and then creates the model on the meta device. After that, it loads the model weights to the real device.
*What is meta device?* [Meta device](https://pytorch.org/docs/stable/meta.html) is an abstract device that is not backed by any physical device, provided by PyTorch. It is used to avoid allocating and writing real tensors to the physical device during model creation. Model creation on real devices will allocate empty tensors with the correct shape and dtype, this can be heavy for large models. For large models, the device during runtime may only hold a small portion of the model weights if the model cannot be accommodated in the device, and the device may not know what portion of the model weights is needed during initialization. On meta device, all the tensor allocations will only allocate a placeholder (MetaTensor) with no actual memory, they are only used to induce the shape and dtype, and the actual memory will be allocated when the model weights are loaded to the real device. This way, the model creation is much faster and the device can hold the entire model weights. This makes the model creation much faster and easier for large models.
*Why set rope device first?* Tensors on meta device are not real tensors, they do not support data-dependent operations, and that's why we need to set the rope device first to the current GPU device. During model creation, the attention layer needs to initialize the rope information, and it needs to precompute the cos/sin tables:
```python
# rotary.py lines 24-32
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2)))
t = torch.arange(max_position_embeddings)
freqs = torch.einsum("i,j -> ij", t, inv_freq)
cos = freqs.cos()
sin = freqs.sin()
self._cos_sin_cache = torch.cat((cos, sin), dim=-1)  # ← Actual data!
```
The rope device must be set to a real device for this operation to work.

After the model is created, it loads the model weights to the real device. The model loading includes download, load and sharding of the model weights. The `_load_weight_state_dict` function calls the `load_hf_weight` function to load the model weights to the real device.
```python
def load_hf_weight(model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    if os.path.isdir(model_path):
        # if the model path is a local directory, use it directly
        hf_folder = model_path
    else:
        # if the model path is a Hugging Face repository ID, download it
        try:
            hf_folder = snapshot_download(
                model_path,
                allow_patterns=["*.safetensors"],
                tqdm_class=DisabledTqdm,
            )
        except Exception:
            raise ValueError(
                f"Model path '{model_path}' is neither a local directory nor a valid HuggingFace repository ID"
            )

    # find the all *.pt files in the hf_folder
    files = glob.glob(f"{hf_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)
    # If tensor parallel is enabled (more than 1 GPU), shard the model weights
    if get_tp_info().size > 1:
        state_dict = _shard_state_dict(state_dict)

    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    # The merge function merges qkv weights in a single tensor, and gate and up weights in a single tensor.
    return _merge_state_dict(state_dict)
```
It returns the final state dictionary that is ready to be loaded to the model. We will cover details of the model sharding and merging in the [[MiniSGL Model]] section.
The complete model loading workflow is:
```shell
┌─────────────────────────────────────────────────────────────────────────┐
│  Model Loading Flow                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: set_rope_device(cuda:0)                                        │
│          → RoPE will create cos/sin cache on GPU                        │
│                                                                         │
│  Step 2: with torch.device("meta"):                                     │
│              model = LlamaForCausalLM(config)                           │
│                                                                         │
│          ┌────────────────────────────────────────────────────────┐     │
│          │  model.embed    = MetaTensor([32000, 4096])            │     │
│          │  model.layers[0].qkv = MetaTensor([12288, 4096])       │     │
│          │  model.layers[0].o   = MetaTensor([4096, 4096])        │     │
│          │  ...                                                   │     │
│          │  model.rope._cos_sin_cache = RealTensor on GPU!        │     │
│          │                              (exception for RoPE)      │     │
│          └────────────────────────────────────────────────────────┘     │
│          Memory: ~0 (except RoPE cache)                                 │
│                                                                         │
│  Step 3: load_hf_weight(path, device)                                   │
│                                                                         │
│          ┌─────────────────────────────────────────────────────┐        │
│          │  Disk (safetensors)                                 │        │
│          │  ├── model-00001.safetensors                        │        │
│          │  ├── model-00002.safetensors                        │        │
│          │  └── ...                                            │        │
│          └───────────────────┬─────────────────────────────────┘        │
│                              │ load to CPU                              │
│                              ▼                                          │
│          ┌─────────────────────────────────────────────────────┐        │
│          │  CPU RAM (temporary)                                │        │
│          │  { "q_proj": [4096,4096], "k_proj": [...], ... }    │        │
│          └───────────────────┬─────────────────────────────────┘        │
│                              │ _shard_state_dict() (if TP > 1)          │
│                              ▼                                          │
│          ┌─────────────────────────────────────────────────────┐        │
│          │  CPU RAM (sharded)                                  │        │
│          │  { "q_proj": [2048,4096], ... }  ← only my shard    │        │
│          └───────────────────┬─────────────────────────────────┘        │
│                              │ .to(device)                              │
│                              ▼                                          │
│          ┌─────────────────────────────────────────────────────┐        │
│          │  GPU Memory                                         │        │
│          │  { "q_proj": CudaTensor, ... }                      │        │
│          └───────────────────┬─────────────────────────────────┘        │
│                              │ _merge_state_dict()                      │
│                              ▼                                          │
│          ┌─────────────────────────────────────────────────────┐        │
│          │  GPU Memory (merged)                                │        │
│          │  { "qkv_proj": CudaTensor, "gate_up_proj": ... }    │        │
│          └───────────────────┬─────────────────────────────────┘        │
│                                                                         │
│  Step 4: model.load_state_dict(weights)                                 │
│          → Meta tensors replaced with real GPU tensors                  │
│                                                                         │
│          ┌────────────────────────────────────────────────────────┐     │
│          │  model.embed    = CudaTensor([32000, 4096]) ✓          │     │
│          │  model.layers[0].qkv = CudaTensor([12288, 4096]) ✓     │     │
│          │  ...                                                   │     │
│          └────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
##### Determine Number of Pages for KV Cache
Calculate the number of pages for the KV cache based on the free memory and the memory ratio.
```python
        self.num_pages = self.dummy_page = self._determine_num_pages(init_free_memory)
        self.kv_cache = create_kvcache(
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            head_dim=self.model_config.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
```
create the KV cache with the configuration and available pages (see [[MiniSGL KVCache]] for more details on `create_kvcache` function).
We add 1 for the dummy page, because the dummy page is used to pad the batch for CUDA graphs.
The number of pages is computed by the `_determine_num_pages_impl` function:
```python
    def _determine_num_pages_impl(self, old_free_memory: int) -> Tuple[int, int]:
        # sync another round the get the new free memory
        new_free_memory = self._sync_get_memory()[1]
        # how many bytes per page?
        cache_per_page = (
            2  # key + value
            * self.model_config.head_dim
            * divide_even(self.model_config.num_kv_heads, self.config.tp_info.size)
            * self.config.page_size
            * self.dtype.itemsize
            * self.model_config.num_layers
        )
        if self.config.num_page_override is not None:
            return self.config.num_page_override, cache_per_page
        # calculate the available memory for KV cache
        # new_free_memory computes the free memory after loading the model. old_free_memory * (1 - memory_ratio) is the size that the engine should keep free. delta is the available memory for KV cache, and we compute the number of pages by dividing the available memory by the bytes per page.
        delta = new_free_memory - int(old_free_memory * (1 - self.config.memory_ratio))
        num_pages = delta // cache_per_page
        return num_pages, cache_per_page    
```

##### Create Page Table
Create the page table with the configuration and the number of pages.
```python
        self.page_table = create_page_table(  # + 1 for dummy request
            (config.max_running_req + 1, config.max_seq_len),
            device=self.device,
        )
```
See [[MiniSGL Page Table]] for more details on `create_page_table` function.
We add 1 for the dummy request, because the dummy request is used to pad the batch for CUDA graphs.

##### Create Attention Backend
Creates FlashInfer or FlashAttention backend with access to KV cache and page table.
See [[MiniSGL Attention]] for more details on `create_attention_backend` function.
```python
        self.attn_backend = create_attention_backend(
            config.model_config,
            self.kv_cache,
            config.attention_backend,
            self.page_table,
        )
```

##### Create Context
Creates the context for the engine, `Context` is a global singleton that attention layers access to get the current batch and KV cache.
```python
        self.ctx = Context(
            page_size=1,
            kv_cache=self.kv_cache,
            attn_backend=self.attn_backend,
            page_table=self.page_table,
        )
        set_global_ctx(self.ctx)

##### Create Sampler
Creates the sampler for the engine, `Sampler` is responsible for sampling the next tokens.
```python
        self.sampler = Sampler(self.device)
```

##### CUDA Graph Runner
Creates the CUDA graph runner for the engine, `GraphRunner` is responsible for running the CUDA graphs.
```python
        # gets minimum free memory after initialization
        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # cuda graph related
        # Create the dummy request for padding the batch for CUDA graphs
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        # Fill the dummy page in the page table
        self.page_table[self.dummy_req.table_idx].fill_(self.dummy_page)
        # Create the graph runner
        # It will run dummy requests with various batch sizes to capture the CUDA graphs for the decode batches.
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=config.max_seq_len,
            vocab_size=self.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )
```
##### Engine Initialization Summary
Now the engine is initialized, the workflow is:
```shell
┌────────────────────────────────────────────────────────┐
│              Engine.__init__() Flow                    │  
├────────────────────────────────────────────────────────┤
│                                                        │
│  1. Set TP info (rank, size)                           │
│                    ↓                                   │
│  2. Initialize CUDA device + stream                    │
│                    ↓                                   │
│  3. Set up distributed communication (NCCL + Gloo)     │
│                    ↓                                   │
│  4. Load model (meta device → real weights)            │
│                    ↓                                   │
│  5. Calculate KV cache size from free memory           │
│                    ↓                                   │
│  6. Allocate KV cache + page table                     │
│                    ↓                                   │
│  7. Create attention backend (FlashInfer)              │
│                    ↓                                   │
│  8. Set up global context                              │
│                    ↓                                   │
│  9. Capture CUDA graphs for decode batches             │
│                    ↓                                   │
│  Done! Engine ready to process batches                 │
│                                                        │
└────────────────────────────────────────────────────────┘
```
And the engine is ready to process forward batches.

### Engine Forward
`python/minisgl/engine/engine.py`
The `forward_batch` function is the main entry point for the engine to process a forward batch.
It first checks if the current stream is on the dedicated stream for the engine, it is critical for the CUDA graphs because they are bound to the engine stream.
```python
    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream, "Current stream must be the engine stream"
```
Then it sets the batch to the global context. The `forward_batch` function is a context manager that sets the batch to the global context. 
```python
        with self.ctx.forward_batch(batch):
```
It sets the batch in the global context structure in `Context` class in the `core.py` file.
```python
@contextmanager
def forward_batch(self, batch: Batch):
    self.set_batch(batch)  # _batch = batch
    try:
        yield
    finally:
        self.reset_batch()  # _batch = None
```
The reason we use global context because the model layers need to access to batch info like attention metadata, positions, etc. We don't want to pass it through each layer.

It then executes the forward pass. See details in [[MiniSGL Model]] section.
```python
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()
```
`can_use_cuda_graph` function checks if the CUDA graph can be used for the batch. Only decode batches with size less than `cuda_graph_max_bs` can be used for CUDA graphs.

After the forward pass, it increments the request `cached_len` (to the current device length) and `device_len` (to the current device length + 1) to prepare for the next forward pass.
```python
        for req in batch.reqs:
            req.complete_one()
```
> Example:
> Before:
> cached_len = 10    (KV computed and stored)
> device_len = 15    (positions we just processed)
> After:
> cached_len = 15    (new KV is now cached)
> device_len = 16    (ready for next token)


Then it samples the next tokens. Sampling means selecting the next token from the logits.
Logits are the model's output after decoding. It is a tensor of shape (batch_size, vocab_size), representing the log probabilities of each token as the next one.
```python
        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
```
- `logits[: batch.size]` — Only take real requests (ignore padding)
- `sampler.sample()` — Sampling with designated sampling parameters (see the sampler section for more details)
- `.to(torch.int32)` — Token IDs are integers

After sampling and getting the next token IDs, we can copy the tokens to the CPU memory asynchronously. 
```python
        # It pins the CPU memory to avoid GPU stuck by page in-and-out.
        next_tokens_cpu = torch.empty_like(next_tokens_gpu, device="cpu", pin_memory=True)
        # async copy to CPU memory
        next_tokens_cpu.copy_(next_tokens_gpu, non_blocking=True)
        # The event can be used to sync the copy operation, when the data is needed on CPU, the scheduler can call the copy_done_event.synchronize() to wait for the copy operation to complete.
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        # Return the forward output to the scheduler
        # We need next_tokens_gpu to append to the token pool (in GPU memory) for the next pass
        # next_tokens_cpu is required by the scheduler to check EOS and updating host state (on the CPU side)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```
The main workflow is:
```shell
1. Enter context (sets up batch state)
          │
          ▼
2. Can use CUDA graph? ──Yes──► graph_runner.replay()
          │                         │
          No                        │
          ▼                         │
   model.forward()                  │
          │                         │
          ◄─────────────────────────┘
          │
          ▼
3. Update request state (complete_one)
          │
          ▼
4. Sample next tokens
          │
          ▼
5. Async copy to CPU (with event for sync)
          │
          ▼
6. Return ForwardOutput to scheduler
```

# CUDA Graph 
`python/minisgl/engine/graph.py`
[CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) allow work to be define as a graph of operations rather than a single operation. They solve the problem of CPU-GPU back-and-forth in a sequence of lauching and executing fast kernel calls. They provide a mechanism to launch multiple GPU kernel operations through a single CPU operation, and reduce the kernel launch overhead.
![[MiniSGL Engine-20251226155204850.png]]
> https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/?utm_source=chatgpt.com

### CUDA Graph Runner
The `GraphRunner` class is responsible for capturing and running the CUDA graphs for the decode batches. 
```python
class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ):
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        if len(cuda_graph_bs) == 0:
            logger.info_rank0("CUDA graph is disabled.")
            self.max_graph_bs = 0
            self.graph_map = {}
            return

        cuda_graph_bs = sorted(set(cuda_graph_bs), reverse=True)
        self.max_graph_bs = max(cuda_graph_bs)
        self.logits = torch.empty(
            (self.max_graph_bs, vocab_size),
            dtype=torch.float16,
            device=device,
        )
        self.attn_backend = attn_backend
        attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=cuda_graph_bs)

        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {cuda_graph_bs}")
        free_memory = torch.cuda.mem_get_info(device)[0]
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        # warm up by capturing a graph and then destroying it
        g = torch.cuda.CUDAGraph()
        batch = Batch(reqs=[dummy_req] * self.max_graph_bs, phase="decode")
        attn_backend.prepare_for_capture(batch)
        with get_global_ctx().forward_batch(batch):
            self.logits[:] = model.forward()
            with torch.cuda.graph(g, stream=stream):
                self.logits[:] = model.forward()
        del g

        graph_list: List[Tuple[int, torch.cuda.CUDAGraph]] = []
        pbar = tqdm(
            cuda_graph_bs,
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )

        pool = None
        for bs in pbar:
            remaining_memory, _ = torch.cuda.mem_get_info(device)
            pbar.desc = (
                "Capturing graphs: "
                f"bs = {bs:<3} | "
                f"avail_mem = {remaining_memory / (1 << 30):.2f} GiB"
            )
            pbar.refresh()
            g = torch.cuda.CUDAGraph()
            if bs != self.max_graph_bs:
                batch = Batch(reqs=[dummy_req] * bs, phase="decode")
                self.attn_backend.prepare_for_capture(batch)
            with get_global_ctx().forward_batch(batch):
                self.logits[:bs] = model.forward()
                with torch.cuda.graph(g, pool=pool, stream=stream):
                    self.logits[:bs] = model.forward()
            if pool is None:
                pool = g.pool()
            graph_list.append((bs, g))

        free_memory = torch.cuda.mem_get_info(device)[0]
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

        # Sort by batch size ascendingly for easy searching
        self.graph_map = dict(graph_list)
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
```

During initialization, it captures the pre-defined batch sizes for CUDA graphs by running dummy requests with those batch sizes.

##### Determine CUDA Graph Batch Sizes
```python 
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        if len(cuda_graph_bs) == 0:
            logger.info_rank0("CUDA graph is disabled.")
            self.max_graph_bs = 0
            self.graph_map = {}
            return
```
It first determines the batch sizes based on the free memory and the maximum batch size.
```python
def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    # check if the batch sizes are predefined
    if cuda_graph_bs is not None:
        return cuda_graph_bs
    # if not predefined, determine the batch sizes based on the free memory and the maximum batch size.
    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
```
##### Allocate Output Buffer
Deduplicate the batch sizes and sort them in descending order, allocate the logits tensor for the CUDA graphs.
```python
        cuda_graph_bs = sorted(set(cuda_graph_bs), reverse=True)
        self.max_graph_bs = max(cuda_graph_bs)
        self.logits = torch.empty(
            (self.max_graph_bs, vocab_size),
            dtype=torch.float16,
            device=device,
        )
        # Initialize the attention backend for capturing the CUDA graphs.
        self.attn_backend = attn_backend
        attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=cuda_graph_bs)
```
Note that CUDA graphs capture specific memory addresses, so the output must go to the same tensor every time (self.logits in this case).

##### Warm Up
```python
        # warm up by capturing a graph and then destroying it
        g = torch.cuda.CUDAGraph()
        batch = Batch(reqs=[dummy_req] * self.max_graph_bs, phase="decode")
        attn_backend.prepare_for_capture(batch)
        with get_global_ctx().forward_batch(batch):
            self.logits[:] = model.forward()
            with torch.cuda.graph(g, stream=stream):
                self.logits[:] = model.forward()
        del g
```
We need to warm up the CUDA graphs to do some lazy initializations:
- CUDA lazy initialization: allocates internal buffers
- cuBLAS handle creation: it does a one-time setup of the math library
- Memory fregmentation: after warm up and deletion, we have a contiguous memory region for the CUDA graphs.
If we don't warm up, the CUDA graphs will capture those initilizations and cause failure when running the graphs.

##### Capture CUDA Graphs
```python
        # uses a pool for memory sharing, because there will only be one CUDA graph running at a time
        pool = None
        for bs in pbar:
            remaining_memory, _ = torch.cuda.mem_get_info(device)
            pbar.desc = (
                "Capturing graphs: "
                f"bs = {bs:<3} | "
                f"avail_mem = {remaining_memory / (1 << 30):.2f} GiB"
            )
            pbar.refresh()
            g = torch.cuda.CUDAGraph()
            if bs != self.max_graph_bs:
                batch = Batch(reqs=[dummy_req] * bs, phase="decode")
                # prepare the batch for capturing the CUDA graphs
                self.attn_backend.prepare_for_capture(batch)
            with get_global_ctx().forward_batch(batch):
                # dry run, make sure to skip kernel selection, cache, etc.
                self.logits[:bs] = model.forward()
                # capture the CUDA graph
                with torch.cuda.graph(g, pool=pool, stream=stream):
                    self.logits[:bs] = model.forward()
            if pool is None:
                # assign the pool
                pool = g.pool()
            graph_list.append((bs, g))
```
The difference between the per-batch dry run and the warmup is that dry run allows cuBLAS to select the best algorithm (including benchmarking), the graph should not capture those behaviors. Warmup does one-time initializations. We still cannot skip the warmup because the initializations in the warmup will pollute the pool.

##### Replay
```python
    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.logits[: batch.size]
```
The attention backend prepares the batch for replay. It will copy the new data (input_ids, positions, attention metadata) into the output buffer because the CUDA graph only captures the addresses. Then it calls `g.replay()` to run the CUDA graph. Finally, it returns the logits for the batch.

##### Batch Padding
```python
    def pad_batch(self, batch: Batch) -> int:
        if not batch.is_decode or batch.size > self.max_graph_bs:
            padded_size = batch.size
        else:  # only pad decode batch smaller than max_graph_bs
            padded_size = next(bs for bs in self.graph_bs_list if bs >= batch.size)
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
        return batch.padded_size - batch.size
```
CUDA graphs require fixed batch sizes. Pad the batch to its closest fixed size with dummy requests.

##### CUDA Graph Runner Summary
```shell
┌─────────────────────────────────────────────────────────────────────────┐
│                      GraphRunner Lifecycle                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INITIALIZATION (once at startup)                                       │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  1. Determine batch sizes: [1, 2, 4, 8, 16, 24, ..., 160]               │
│                                                                         │
│  2. Pre-allocate logits buffer: [160, vocab_size]                       │
│                                                                         │
│  3. Warm-up capture (and discard)                                       │
│                                                                         │
│  4. For each batch size:                                                │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │  batch = [dummy_req] × bs                                   │     │
│     │  dry_run: model.forward()                                   │     │
│     │  with torch.cuda.graph(g):                                  │     │
│     │      capture: model.forward()                               │     │
│     │  graph_map[bs] = g                                          │     │
│     └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  Result: graph_map = {1: g1, 2: g2, 4: g4, 8: g8, ...}                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  RUNTIME (every decode batch)                                           │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Input: batch with 5 requests                                           │
│                                                                         │
│  1. pad_batch(): 5 → 8 (add 3 dummy requests)                           │
│                                                                         │
│  2. prepare_for_replay():                                               │
│     Copy real input_ids, positions, etc. into capture buffers           │
│                                                                         │
│  3. graph_map[8].replay()                                               │
│     Single CUDA call → all kernels execute                              │
│                                                                         │
│  4. return logits[:5]                                                   │
│     Discard dummy outputs, return real results                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Attention Backend Initialization and Preparation for CUDA Graphs
Take `python/minisgl/backends/fa3.py` as an example.
The attention backend needs special handling for CUDA graphs because graphs capture memory addresses, not values. During graph capture, the kernel launch, memory addresses and tensor shapes are captured, but the tensor values are not. However, the attention backend always creates new tensors every batch during the forward pass. So the attention backend needs to pre-allocate buffers for the CUDA graphs and copy new data into the buffers during the replay.
##### Pre-allocate Buffers
The `BaseCaptureData` class provides the base method to create the capture data. It creates stable addresses for the capture data: `input_ids`, `seq_lens`, `positions`, `cu_seqlens_k`, `cu_seqlens_q`, `page_table`, `out_loc`.
```python
@dataclass
class BaseCaptureData:
    input_ids: torch.Tensor 
    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor
    out_loc: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        return cls(
            input_ids=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            out_loc=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            **kwargs,
        )
```
- `input_ids`: input token IDs, shape (batch_size,)
- `seq_lens`: sequence length of each request, shape (batch_size,)
- `positions`: position of the next token of each request, shape (batch_size,)
- `cu_seqlens_k`: cumulative sequence lengths for the key, shape (batch_size + 1,) (the extra 1 is for the end)
- `cu_seqlens_q`: cumulative sequence lengths for the query, shape (batch_size + 1,) (the extra 1 is for the end)
- `page_table`: page table, shape (batch_size, max_seq_len)
- `out_loc`: output locations, shape (batch_size,)

##### Initialize Capture Data
The engine calls the `init_capture_graph` function once at startup. 
```python
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FA3CaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)
```
It creates the capture buffers sized for the maximum batch, all smaller batches will use the same buffers.

##### Prepare for Capture
The engine calls the `prepare_for_capture` function before capturing the CUDA graphs. It creates the metadata with the pre-allocated buffers and assigns it to the batch.
```python
    def prepare_for_capture(self, batch: Batch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = FA3Metadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            positions=capture.positions[:bs],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1),
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata
        batch.input_ids = capture.input_ids[:bs]
        batch.out_loc = capture.out_loc[:bs]
```
The metadata stores all the pre-allocated buffers for the CUDA graphs.

##### Prepare for Replay
The engine calls the `prepare_for_replay` function before replaying the CUDA graphs. It (asynchronously) copies the new data (input_ids, positions, out_loc) into the capture buffers so that the CUDA graph can use the same addresses with new data to run the same kernels.
```python
    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FA3Metadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        self.capture.input_ids[:bs].copy_(batch.input_ids)
        self.capture.out_loc[:bs].copy_(batch.out_loc)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.positions[:bs].copy_(metadata.positions)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, : metadata.max_seqlen_k].copy_(metadata.page_table)
```

##### Attention Backend Preparation for CUDA Graphs Summary
```shell
┌─────────────────────────────────────────────────────────────────────────┐
│                    Graph Lifecycle on Attention Backend                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INITIALIZATION (once)                                                  │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  GraphRunner calls: attn_backend.init_capture_graph(8192, [1,2,4,...])  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  Allocate capture buffers (max_bs=160, max_seq=8192)         │       │
│  │    capture.input_ids   = zeros([160])                        │       │
│  │    capture.positions   = zeros([160])                        │       │
│  │    capture.page_table  = zeros([160, 8192])                  │       │
│  │    ...                                                       │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  CAPTURE PHASE (for each batch size)                                    │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  GraphRunner calls: attn_backend.prepare_for_capture(batch)             │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  Set batch.attn_metadata to use capture buffer SLICES        │       │
│  │    metadata.positions = capture.positions[:bs]  ← Same addr  │       │
│  │                                                              │       │
│  │  Set batch.input_ids = capture.input_ids[:bs]                │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  Then: model.forward() is called inside torch.cuda.graph()              │
│  Graph records: "Read from address of capture.positions"                │
│                                                                         │
│  REPLAY PHASE (every inference)                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  GraphRunner calls: attn_backend.prepare_for_replay(batch)              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  Copy NEW values into capture buffers                        │       │
│  │    capture.input_ids[:bs].copy_(batch.input_ids)             │       │
│  │    capture.positions[:bs].copy_(metadata.positions)          │       │
│  │    capture.page_table[:bs].copy_(metadata.page_table)        │       │
│  │    ...                                                       │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  Then: graph.replay() runs                                              │
│  Graph reads from capture buffer addresses → gets new values            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```


# Sampler
`python/minisgl/engine/sample.py`
After the forward pass, the engine gets the logits for the next tokens for each request. It then samples the next tokens from the logits.
### Data Structure
```python
@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
```
MiniSGL only supports greedy and temperature sampling for now. If temperature is None, it will use greedy sampling, otherwise it will use temperature sampling.

### Prepare - Convert Req Parameters to GPU temperatures
The engine calls the `prepare` function to convert the request parameters to GPU temperatures.
```python
    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        if all(r.sampling_params.temperature <= 0.0 for r in batch.reqs):
            return BatchSamplingArgs(temperatures=None)
        MIN_T = 1e-5
        return BatchSamplingArgs(
            temperatures=torch.tensor(
                [max(r.sampling_params.temperature, MIN_T) for r in batch.reqs],
                dtype=torch.float32,
                pin_memory=True,
            ).to(self.device, non_blocking=True)
        )
```
If all the request parameters have temperature 0, it will return None and skip GPU work. If temperature is not 0, it creates a CPU tensor in pinned memory and asynchronously copies it to the GPU. It clamp the temperature to be at least 1e-5 to avoid division by zero.

### Sample - Choose Next Token IDs
The engine calls the `sample` function to choose the next token IDs from the logits.
```python
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:
                return torch.argmax(logits, dim=-1)
            return self._sample(logits, args.temperatures)

    def _sample(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        logits.div_(temperatures.unsqueeze(-1))
        torch.softmax(logits, dim=-1, out=logits)
        return torch.multinomial(logits, num_samples=1).view(-1)
```
The sampling is wrapped in an nvtx range so NSight can profile it. If temperature is None, it will use greedy sampling, simply select the token with the highest probability (torch.argmax). Otherwise, it calls the `_sample` function to do temperature sampling. By setting a temperature, the logits will be divided by the temperature and then softmaxed to get the probabilities. Finally, it samples the next token IDs from the probabilities (torch.multinomial).
The formula of computing the probabilities with temperature: 
$$
P(x) = \frac{e^{z_x / T}}{\sum_{y} e^{z_y / T}}
$$
where $z_x$ is the logit of the token $x$, $T$ is the temperature, and $P(x)$ is the probability of the token $x$, $y$ is all the tokens in the vocabulary.

By scaling with T, we can control the diversity of the sampling. If $T < 1$, the probabilities will be more concentrated on the top tokens, so it's closer to greedy sampling. If $T > 1$, the probabilities will be more uniform, so it's closer to random sampling.
In practice, the parameters also include top-k and top-p, but MiniSGL doesn't support them for simplicity. The top-k sampling is to keep the top $k$ tokens with the highest probabilities, and the top-p sampling is to keep the cumulative probability of the top $p$ tokens with the highest probabilities.
