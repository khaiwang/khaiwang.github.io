---
title: "MiniSGL KVCache"
date: 2025-12-28
author: "Zikai Wang"
description: "The KVCache of MiniSGL."
summary: "The KVCache of MiniSGL."
---
[[MiniSGL Walk Through]]
The KVCache mainly contains:
```shell
┌─────────────────────────────────────────────────────────────────┐
│                        KVCache Module                           │
├─────────────────────────────┬───────────────────────────────────┤
│   STORAGE (Where to store)  │   MANAGEMENT (What to cache)      │
├─────────────────────────────┼───────────────────────────────────┤
│  BaseKVCache (base.py)      │  BaseCacheManager (base.py)       │
│       ↓                     │       ↓                           │
│  MHAKVCache (mha_pool.py)   │  RadixCacheManager ← prefix cache │
│                             │  NaiveCacheManager ← no caching   │
└─────────────────────────────┴───────────────────────────────────┘
```

# BaseKVCache Definition
`python/minisgl/kvcache/base.py`
### BaseKVCache Definition
`BaseKVCache` is the base class for all the KV caches. It defines the interface of k_cache, v_cache, and store_kv.
```python
class BaseKVCache(ABC):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used.
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...
```
### Type and Layout Definition
```python
class KVCacheLayout(enum.Enum):
    LayerFirst = enum.auto()
    PageFirst = enum.auto()


class KVCacheType(enum.Enum):
    MHA = enum.auto()
```
LayerFirst groups by layers, PageFirst groups by token slots.
MiniSGL only supports MHA type of KV cache for now.

### BaseCache Handle Definition
```python
@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int


class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size
```
A handle is a reference to a matched cache slot:
- cached_len: how many tokens were matched
Subclasses add implementation details (e.g., RadixCacheHandle includes the tree node)
SizeInfo tracks cache occupancy:
- evictable_size: slots that can be reclaimed
- protected_size: slots currently in use (locked)

### BaseCacheManager Definition
The `BaseCacheManager` is the the prefix caching interface, it tracks what's cached and manages eviction:
```python
class BaseCacheManager(ABC):
    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[BaseCacheHandle, torch.Tensor]:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            handle (BaseCacheHandle): The handle to the matched prefix.
            indices (torch.Tensor): The indices of the longest-matched prefix in the cache.
        """

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            int: The length of prefix that is already in the cache. This part is not
                 inserted, so the caller should free these indices.
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.
        """

```

# Actual KVCache Storage - MHAKVCache
`python/minisgl/kvcache/mha_pool.py`
`MHAKVCache` allocates the GPU memory for K/V tensors

### MHAKVCache Definition
```python
class MHAKVCache(BaseKVCache):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """
```
##### Constructor
```python
    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        dtype: torch.dtype,
        kv_layout: KVCacheLayout,
        device: torch.device,
    ):
        tp_info = get_tp_info()
        local_kv_heads = divide_even(num_kv_heads, tp_info.size)
        match kv_layout:
            case KVCacheLayout.PageFirst:
                kv_buffer = torch.empty(
                    (2, num_pages, num_layers, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                ).permute(0, 2, 1, 3, 4)
            case KVCacheLayout.LayerFirst:
                kv_buffer = torch.empty(
                    (2, num_layers, num_pages, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                )
            case _:
                raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        self._kv_buffer = kv_buffer.view(2, num_layers, num_pages, 1, local_kv_heads, head_dim)
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages, local_kv_heads, head_dim)
```
It first get tp info and divide the number of kv heads to the number of tp groups.
```python
tp_info = get_tp_info()
local_kv_heads = divide_even(num_kv_heads, tp_info.size)
```
Then it allocates the GPU memory for K/V tensors.
```python
match kv_layout:
    case KVCacheLayout.PageFirst:
        kv_buffer = torch.empty(
            (2, num_pages, num_layers, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        ).permute(0, 2, 1, 3, 4)
    case KVCacheLayout.LayerFirst:
        kv_buffer = torch.empty(
            (2, num_layers, num_pages, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
    case _:
        raise ValueError(f"Unsupported kv_layout: {kv_layout}")
```
Two memory layouts supported:
1. PageFirst: `[2, pages, layers, heads, dim]` — better for page-oriented ops
2. LayerFirst: `[2, layers, pages, heads, dim]` — better for layer-oriented ops
```shell
   [2, num_layers, num_pages, 1, local_kv_heads, head_dim]
    ↑       ↑          ↑      ↑        ↑           ↑
   K/V   layer idx   slot   batch   kv heads    dim
```
In PageFirst layout, the buffer is permuted to `[2, num_layers, num_pages, 1, local_kv_heads, head_dim]`, this gives the caller a unified shape to access the K/V tensors. Although the buffer is permuted, the physical memory is still unchanged, so that it still works for page-oriented ops like paged attention.
```python
self._kv_buffer = kv_buffer.view(2, num_layers, num_pages, 1, local_kv_heads, head_dim)
self._num_layers = num_layers
self._k_buffer = self._kv_buffer[0]
self._v_buffer = self._kv_buffer[1]
self._device = device
self._storage_shape = (num_pages, local_kv_heads, head_dim)
```
All the buffer data and metadata for convenience.
- The `_kv_buffer` is the main buffer, it is a 6D tensor with shape `[2, num_layers, num_pages, 1, local_kv_heads, head_dim]`.
- The `_k_buffer` and `_v_buffer` are the K/V buffers, they are 3D tensors with shape `[num_layers, num_pages, local_kv_heads, head_dim]`.
- The `_storage_shape` is the shape of the K/V tensors, it is `(num_pages, local_kv_heads, head_dim)`.

##### Accessors
```python
def k_cache(self, layer_id: int) -> torch.Tensor:
    return self._k_buffer[layer_id].view(self._storage_shape)

def v_cache(self, layer_id: int) -> torch.Tensor:
    return self._v_buffer[layer_id].view(self._storage_shape)
```
The `k_cache` and `v_cache` accessors return the K/V tensors for the given layer.
The `store_kv` method stores the K/V tensors for the given layer.

##### Store KV
```python
def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None:
    from minisgl.kernel import store_cache
    store_cache(
        k_cache=self._k_buffer[layer_id].view(self._storage_shape),
        v_cache=self._v_buffer[layer_id].view(self._storage_shape),
        indices=out_loc,
        k=k,
        v=v,
    )
```
This is the write operation — storing computed K/V into the cache at specific slot indices.
Uses a custom CUDA kernel (`python/minisgl/kernels/store.py/store_cache`, `python/minisgl/kernels/csrc/jit/store.cu`) for efficient scatter writes:
- `out_loc`: tensor of indices like [5, 12, 47] — where to write each token's KV
- The kernel writes `k[i] → k_cache[out_loc[i]]` for all i in parallel.

# Prefix Caching with Radix Tree
`python/minisgl/kvcache/radix_manager.py`
### Radix Tree
##### RadixTreeNode Definition
```python
class RadixTreeNode:
    counter: int = 0

    def __init__(self, tic: int | None = None) -> None:
        self.children: Dict[int, RadixTreeNode] = {} # Child nodes, keyed by first token
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0 # How many requests are using this prefix
        self.uuid = RadixTreeNode.counter # Unique identifier for the node
        RadixTreeNode.counter += 1 # Global counter for the node
        self.timestamp = tic or time.monotonic_ns() # Last access time (for LRU eviction)

        # these fields should be updated later
        self._key: torch.Tensor # 	Token IDs (edge label in the tree)
        self._value: torch.Tensor # Cache slot indices where KV is stored
        self._length: int
```
An example of the radix tree:
```shell
Requests:
  "Hello world how are you"
  "Hello world what's up"
  "Hi there"

Radix Tree:
                    [root]
                   /      \
        key="Hello"        key="Hi there"
        val=[0,1,2,3,4]    val=[10,11,12,13,14,15,16]
              |
     key=" world"
     val=[5,6,7,8,9,10]
        /           \
  key=" how..."    key=" what..."
  val=[11,12,...]  val=[20,21,...]
```
##### Core Operations of Radix Tree
###### match_prefix: find the longest matching prefix
```python
    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[RadixCacheHandle, torch.Tensor]:
        node, prefix_len = self._walk(input_ids)
        if prefix_len == 0:
            assert node.is_root() and node is self.root_node and prefix_len == 0
            return RadixCacheHandle(prefix_len, node), self.empty_tensor
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return RadixCacheHandle(prefix_len, node), torch.cat(value_list)
```
It walks the tree to find the longest matching prefix, collect the cache slot indices along the way, return the handle and the indices.
```python
    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()
        # Walk the tree top down
        while prefix_len < indice_len:
            this_id = int(input_ids[prefix_len].item())
            if this_id not in node.children:
                return node, prefix_len

            node = node.children[this_id]

            # NOTE: at least 1 char is matched, so match_len >= 1
            match_len = node.get_match_len(input_ids[prefix_len:])
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node._split_at(match_len)
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len
```
It walks the tree top down, if the current node does not have the child node with the current token, it returns the current node and the prefix length. If the current node is fully matched, it updates the timestamp and continues to the next token. If the current node is not fully matched, it splits the node at the given index and returns the new node and the prefix length.
`get_match_len` is a custom kernel function to do fast comparison between the input token and the node key and find the longest matching prefix length.
`_split_at` splits the node at the given index and returns the new node and the prefix length.
```python
    def _split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node
```
It creates a new node from 0 to the given index and sets the current node to the rest. The new node is the parent of the current node.

###### insert_prefix: adding new cached prefix
```python
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        node, prefix_len = self._walk(input_ids)
        assert prefix_len <= len(input_ids)
        if prefix_len < len(input_ids):
            new_node = RadixTreeNode()
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:])
            new_node.set_parent(node)
            self.evictable_size += new_node.length
        return prefix_len
```
It walsk the tree to find the matching prefix node, if the prefix is smaller than the input, it creates a new node with the rest of the input and indices, add it to the tree and increase the evictable size.
It returns the prefix so the caller knows which slots can be reused.

###### lock_handle: lock the handle to protect the prefix from eviction
```python
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node = node.parent
                node.ref_count -= 1
                assert node.ref_count > 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
        else:
            while not node.is_root():
                node = node.parent
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
```
For lock, it increases the ref_count along the way to the root, increases the protected size and decreases the evictable size if the node was evictable.
For unlock, it decreases the ref_count along the way to the root, decreases the protected size and increases the evictable size if the node now has 0 ref_count and is evictable.

###### evict: LRU eviction
```python
    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[int(node._key[0].item())]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)
```
The evict operation first collects all the evictable leaves and puts them into a timestamp min-heap. It then evicts the least recently used leaf node, and if the parent of the evicted node is now a leaf and is evictable (has 0 ref_count), it puts the parent into the heap.
The `_collect_leave_nodes_for_evict` function collects all the evictable leaves and puts them into a list.
```python
    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes
```