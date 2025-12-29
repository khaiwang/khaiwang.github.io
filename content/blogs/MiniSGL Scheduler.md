---
title: "MiniSGL Scheduler"
date: 2025-12-28
author: "Zikai Wang"
description: "The scheduler of MiniSGL."
summary: "The scheduler of MiniSGL."
---
[[MiniSGL Walk Through]]
**Contents:**
- **Core datastructures** — Req, Batch
- **Scheduler Initialization** — Engine, Communication, Managers, Memory Layout
- **Main Overlap Loop** — `_schedule_next_batch`, `_process_last_data`

# Core datastructures
`python/minisgl/core.py`
### Req:
- cached_len
- device_len
- fn remain_len (remains to be compute, max_device - device)
- fn extend_len (need to compute, device - cached)
### Batch
A batch has two phases, prefill and decode
```python
class Batch:

	def __init__(self, *, reqs: List[Req], phase: Literal["prefill", "decode"]):
	
		self.reqs = reqs
		self.phase: Literal["prefill", "decode"] = phase
		# these fields should be set by scheduler
		self.input_ids: torch.Tensor
		self.out_loc: torch.Tensor
		self.padded_reqs: List[Req] # may contain some dummy reqs for padding
	
		# this field should be set by attention backend
	
		self.attn_metadata: BaseAttnMetadata
```
Why we need padding:
1. padding the batch size for kernel optimization
2. to make the batch size consistent among tp groups
3. remove bubbles in continuous batching
`out_loc` directs writes to the padding requests to a trash space. How?

# Scheduler Initialization
`python/minisgl/scheduler/scheduler.py`
#### Basic Setup
Engine initialization (sets up envs, cuda devices, communications, KV cache memory):
```python
from minisgl.engine import Engine
self.config = config
self.engine = Engine(config)
self.tp_info = config.tp_info
```
Initialize I/O mixin (ZeroMQ messaging for inter-process communication):
```python
super().__init__(config, self.engine.tp_cpu_group)
```

#### Communication
Use separate streams to overlap metadata processing with computation:
```python
self.device = self.engine.device
self.stream = torch.cuda.Stream(device=self.device)  # metadata (ZMQ)
self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)  # data (NCCL)
torch.cuda.set_stream(self.stream)  # default to ZMQ stream
```
```shell
                 ┌─────────────────┐
                 │    Tokenizer    │
                 └────────┬────────┘
                          │ ZMQ PUSH/PULL
                          ▼
┌─────────────────────────────────────────────────────┐
│                   Scheduler Rank 0                  │
│   - Receives from tokenizer                         │
│   - Broadcasts to other ranks (ZMQ PUB/SUB)         │
│   - Sends results to detokenizer                    │
└──────────────────────┬──────────────────────────────┘
                       │ ZMQ PUB/SUB
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ Rank 1   │ │ Rank 2   │ │ Rank N   │
   └──────────┘ └──────────┘ └──────────┘
```

#### Managers
**Table Manager** - pre-allocate slots for requests (allocate on arrival, free on finish):
```python
self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
```

**Cache Manager** - KV cache page pool (pre-allocate, allocate new, evict, prefix match):
```python
self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
```

**Decode Manager** - track in-flight decode requests (batching, token tracking):
```python
self.decode_manager = DecodeManager()
```

**Prefill Manager** - process new requests (queue pending, chunk long requests, schedule batches):
```python
self.prefill_manager = PrefillManager(self.cache_manager, self.table_manager, self.decode_manager)
```
```shell
                    New Request
                         │
                         ▼
              ┌─────────────────────┐
              │   Table Manager     │  "Here's your slot (table_idx=5)"
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Prefill Manager    │  "Added to pending queue"
              └──────────┬──────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                                 ▼
┌───────────────┐                ┌───────────────┐
│ Cache Manager │                │ Cache Manager │
│ "Found cached │                │ "Allocating   │
│  prefix!"     │                │  new pages"   │
└───────┬───────┘                └───────┬───────┘
        │                                │
        └────────────────┬───────────────┘
                         ▼
              ┌─────────────────────┐
              │   [GPU Forward]     │  Prefill phase
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Decode Manager     │  "Now tracking for decode"
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   [GPU Forward]     │  Decode phase (loops until done)
              └──────────┬──────────┘
                         │
                         ▼ (when finished)
              ┌─────────────────────┐
              │  Decode Manager     │  "Removed from tracking"
              └──────────┬──────────┘
                         │
                         ▼
        ┌────────────────┼────────────────┐
        ▼                                 ▼
┌───────────────┐                ┌───────────────┐
│ Cache Manager │                │ Table Manager │
│ "Caching for  │                │ "Slot freed"  │
│  future reuse"│                │               │
└───────────────┘                └───────────────┘
```
[[MiniSGL Prefill Manager]]
#### Auxiliary State
```python
self.finished_reqs: Set[Req] = set()
self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
self.eos_token_id = self.tokenizer.eos_token_id
self.page_table = self.engine.page_table
self.token_pool = self.table_manager.token_pool
```

#### Final Memory Layout
```shell
┌─────────────────────────────────────────────────────────────────┐
│                          GPU Memory                             │
├─────────────────────────────────────────────────────────────────┤
│  Model Weights        (~X GiB depending on model)               │
├─────────────────────────────────────────────────────────────────┤
│  KV Cache             (num_pages × cache_per_page)              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ page 0 │ page 1 │ page 2 │ ... │ page N │ dummy_page   │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  Page Table           (max_running_req × max_seq_len) int32     │
│  Token Pool           (max_running_req × max_seq_len) int32     │
├─────────────────────────────────────────────────────────────────┤
│  CUDA Graph Buffers   (if enabled)                              │
├─────────────────────────────────────────────────────────────────┤
│  Free Slots Tensor    (tracking available KV pages)             │
└─────────────────────────────────────────────────────────────────┘
```

# Main -- Overlap loop
```python
    @torch.inference_mode()
    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        Overlap execution of current batch and processing of last batch's results.
        """
        # 1. Receive new messages 
        # blocking: if there is no work, we should block to wait for new messages
        # non-blocking: if there is work (unprocessed data, non-empty prefill/decode requests)
        blocking = not (last_data or self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # 2. Schedule next batch (input for forward pass)
        forward_input = self._schedule_next_batch()
        
        # 3. Run forward pass (on engine stream)
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        # 4. Process last batch's results (while current batch is running)
        self._process_last_data(last_data, ongoing_data)

        # So it interleaves the computation on gpu and data processing on cpu
        return ongoing_data
```
#### Scheduling - \_schedule_next_batch
`minisgl/scheduler/scheduler.py`
Schedule the next batch for execution.
```python
def _schedule_next_batch(self) -> ForwardInput | None:
    prefill_budget = self.config.max_extend_tokens
    batch = (
        self.prefill_manager.schedule_next_batch(prefill_budget)
        or self.decode_manager.schedule_next_batch()
    )
    if batch is None:
        return None

    needed_size = sum(r.extend_len for r in batch.reqs)
    batch.out_loc = self.cache_manager.allocate(needed_size)

    padding_size = self.engine.graph_runner.pad_batch(batch)
    if padding_size > 0:
        batch.out_loc = torch.nn.functional.pad(
            batch.out_loc, (0, padding_size), value=self.engine.dummy_page
        )

    load_indices = make_2d_indices(
        self.token_pool, [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
    )
    write_indices = make_2d_indices(
        self.token_pool, [(r.table_idx, r.device_len, r.device_len + 1) for r in batch.reqs]
    )

    self.page_table.view(-1)[load_indices] = batch.out_loc
    self.engine.attn_backend.prepare_metadata(batch)

    return ForwardInput(
        batch=batch,
        sample_args=self.engine.sampler.prepare(batch),
        load_indices=load_indices,
        write_indices=write_indices,
    )
```

##### 1. Decide what to run:
```python
prefill_budget = self.config.max_extend_tokens
batch = (
    self.prefill_manager.schedule_next_batch(prefill_budget)
    or self.decode_manager.schedule_next_batch()
)
if batch is None:
    return None
```
Prefill budget is the maximum number of tokens that can be processed in a prefill batch.
Prefill next batch: See Prefill Scheduler
Decode next batch: See Decode Scheduler

##### 2. Allocate KV cache pages:
In the req data structure, we have the extend_len for each request.
- extend_len for prefill is len - cached_len
- for decode is 1 (after prefill, complte_one function sets the cached_len to device_len and device_len adds 1)
```python
needed_size = sum(r.extend_len for r in batch.reqs)
batch.out_loc = self.cache_manager.allocate(needed_size)
```
##### 3. Pad the batch:
Why we need padding:
CUDA graphs require fixed batch sizes. Pad the batch to its closest fixed size with dummy requests.
```python
padding_size = self.engine.graph_runner.pad_batch(batch)
if padding_size > 0:
    batch.out_loc = torch.nn.functional.pad(
        batch.out_loc, (0, padding_size), value=self.engine.dummy_page
    )
```
In `minisgl/engine/graph.py`, we have the function `pad_batch` to pad the batch to the closest fixed size.
```python
def pad_batch(self, batch: Batch) -> int:
    if not batch.is_decode or batch.size > self.max_graph_bs: # not decode batch or batch size is greater than max_graph_bs
        padded_size = batch.size
    else:  # only pad decode batch smaller than max_graph_bs
        padded_size = next(bs for bs in self.graph_bs_list if bs >= batch.size)
    # each dummy req has an extend_len of 1, so we need padded_size - batch.size dummy reqs
    batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
    return batch.padded_size - batch.size
```
CUDA graphs only work in decode phase.
##### 4. Compute load/write locations:
make_2d_indices is a helper function to convert the 2D indices and given array of (row_id, col_start, col_end) to flat indices.
Load indices are the indices in the token_pool to load from, use all padded requests because input batch is padded.
Write indices denote the write-out locations for the output batch, it only uses the non-dummy requests and discards the outputs of dummy requests.
```python
load_indices = make_2d_indices(
    self.token_pool, [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs] # load from cached_len to device_len
)
write_indices = make_2d_indices(
    self.token_pool, [(r.table_idx, r.device_len, r.device_len + 1) for r in batch.reqs] # write from device_len to device_len + 1
)
```
##### 5. Write page locations to page table:
We have allocated cache pages for the batch, and we assign those pages to the corresponding indices in the page table.
It tells the attention backend to read those KV cache pages for those token positions.
```python
self.page_table.view(-1)[load_indices] = batch.out_loc 
```
How does page table and token pool work together? see [[MiniSGL Page Table]]

##### 6. Prepare attention metadata:
```python
self.engine.attn_backend.prepare_metadata(batch)
```
This prepares backend-specific attention metadata (e.g., FlashAttention):
- Sequence lengths
- Page table pointers
- Prefill vs decode flags

##### 7. Return the batch as forward input:
```python
return ForwardInput(
    batch=batch,
    sample_args=self.engine.sampler.prepare(batch),
    load_indices=load_indices,
    write_indices=write_indices,
)
```
Forward input is the input for the forward pass.
- batch: the batch to be executed (Requests, phase, attention metadata, out_loc)
- sample_args: the arguments for the sampler (e.g., top-k sampling, temperature, etc.)
- load_indices: the indices to load from the token pool
- write_indices: the indices to write to the token pool


#### Prefill Scheduler - Chunked Prefill
[[MiniSGL Prefill Manager]]

#### Decode Scheduler - Round Robin
`minisgl/scheduler/decode.py`
```python
def schedule_next_batch(self) -> Batch | None:
    if len(self.pending_list) == 0:
        return None
    return Batch(reqs=self.pending_list, phase="decode")
```
Simple round robin scheduler: all running decode requests are batched together (continuous batching).

#### Result Processing - \_process_last_data
`minisgl/scheduler/scheduler.py`
Process the results of the last batch. Need both `last_data` and `ongoing_data` because of the overlap loop — we need to check ongoing data to make sure no resources are freed while the batch is running.
```python
def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
    if last_data is None:
        return
    batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
    copy_done.synchronize()
    reply = BatchTokenizerMsg(data=[])
    max_seq_len = self.config.max_seq_len
    
    for i, req in enumerate(batch.reqs):
        if req in self.finished_reqs or isinstance(req, ChunkedReq):
            continue
        next_token_id = next_tokens_cpu[i]
        req.append_host(next_token_id.unsqueeze(0))
        next_token = int(next_token_id.item())
        finished = req.remain_len <= 0
        if not req.sampling_params.ignore_eos:
            finished |= next_token == self.eos_token_id
        if req.device_len >= max_seq_len - 1:
            finished = True
            logger.warning_rank0(f"Request {req.uid} reached {max_seq_len = }, dropped.")
        reply.data.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))
        if finished:
            self.finished_reqs.add(req)
            self.decode_manager.remove_req(req)
            logger.debug_rank0("Request %s is finished", req)

    ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
    for req in self.finished_reqs.difference(ongoing_reqs):
        self.table_manager.free(req.table_idx)
        self.cache_manager.free_and_cache_finished_req(
            req.cache_handle,
            req.host_ids[: req.cached_len],
            self.page_table[req.table_idx, : req.cached_len]
        )
    
    self.finished_reqs.intersection_update(ongoing_reqs)
    self.send_result(reply)
```

##### 1. Unpack forward data:
- `last_data[0]` is the forward input of the last batch (batch, sample_args, load_indices, write_indices)
- `last_data[1]` is the forward output of the last batch (next_tokens_gpu, next_tokens_cpu, copy_done)
```python
def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
    if last_data is None:
        return
    batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
    copy_done.synchronize()  # wait for GPU→CPU copy
    reply = BatchTokenizerMsg(data=[])
    max_seq_len = self.config.max_seq_len
```

##### 2. Process each request:
Skip already-finished requests: because of the overlap scheduler, we don't know whether a request is finished or not when scheduling the next batch. The previous batch may have finished some requests (seeing EOS token) and added them to the `finished_reqs` set. We should skip here because it is already processed, otherwise the user will get duplicate tokens. Also skip chunked requests because they are not finished until all chunks are processed.
```python
    for i, req in enumerate(batch.reqs):
        if req in self.finished_reqs or isinstance(req, ChunkedReq):
            continue
```

##### 3. Append next token:
```python
        next_token_id = next_tokens_cpu[i]
        req.append_host(next_token_id.unsqueeze(0))
        next_token = int(next_token_id.item())
```

##### 4. Check finish conditions:
A request is finished if:
- It has no remaining length (`remain_len <= 0`)
- The EOS token is encountered (unless `ignore_eos` is set)
- It has reached the maximum sequence length
```python
        finished = req.remain_len <= 0
        if not req.sampling_params.ignore_eos:
            finished |= next_token == self.eos_token_id
        if req.device_len >= max_seq_len - 1:
            finished = True
            logger.warning_rank0(f"Request {req.uid} reached {max_seq_len = }, dropped.")
```

##### 5. Send to detokenizer and mark finished:
```python
        reply.data.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))
        if finished:
            self.finished_reqs.add(req)
            self.decode_manager.remove_req(req)
            logger.debug_rank0("Request %s is finished", req)
```

##### 6. Free resources for finished requests:
Only free resources for finished requests that are not in the ongoing batch (because of overlap scheduler). Free the slot in the page table and token pool. Insert the finished request into the radix tree and free any newly found existing prefix — only the remaining part is cached.
```python
    ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
    for req in self.finished_reqs.difference(ongoing_reqs):
        self.table_manager.free(req.table_idx)
        self.cache_manager.free_and_cache_finished_req(
            req.cache_handle,
            req.host_ids[: req.cached_len],
            self.page_table[req.table_idx, : req.cached_len]
        )
```

##### 7. Finalize and send reply:
Keep only finished requests that are still in the ongoing batch in `finished_reqs` (for next iteration). Send the reply to the detokenizer.
```python
    self.finished_reqs.intersection_update(ongoing_reqs)
    self.send_result(reply)
```

- Append the next token to the request
```shell
                    Batch of 3 requests
                           │
                           ▼
              ┌─────────────────────────┐
              │ GPU Forward Pass        │
              │ → logits (3, vocab_size)│
              └───────────┬─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │ Sampling                │
              │ → next_tokens_gpu       │
              │   tensor([1234, 5678,   │
              │           9012])        │
              └───────────┬─────────────┘
                          │
                          ▼ (async copy)
              ┌─────────────────────────┐
              │ next_tokens_cpu         │
              │ tensor([1234, 5678,     │
              │         9012])          │
              └───────────┬─────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
    i=0: ReqA         i=1: ReqB         i=2: ReqC
        │                 │                 │
        ▼                 ▼                 ▼
  next_token_id     next_token_id     next_token_id
  = tensor(1234)    = tensor(5678)    = tensor(9012)
        │                 │                 │
        ▼                 ▼                 ▼
  .unsqueeze(0)     .unsqueeze(0)     .unsqueeze(0)
  = tensor([1234])  = tensor([5678])  = tensor([9012])
        │                 │                 │
        ▼                 ▼                 ▼
  req.append_host   req.append_host   req.append_host
  (concat to        (concat to        (concat to
   host_ids)         host_ids)         host_ids)
        │                 │                 │
        ▼                 ▼                 ▼
  .item()           .item()           .item()
  = 1234            = 5678            = 9012
  (Python int)      (Python int)      (Python int)
        │                 │                 │
        ▼                 ▼                 ▼
  DetokenizeMsg     DetokenizeMsg     DetokenizeMsg
  (uid, 1234, ...)  (uid, 5678, ...)  (uid, 9012, ...)
```
