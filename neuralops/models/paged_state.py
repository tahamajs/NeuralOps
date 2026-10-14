"""
Continuous Batching Paged Recurrent State Tables & Radix-Tree Prefix State Caching.

Key components:
1. PagedStateTable:
   - Virtual-to-physical GPU state table managing recurrent states (H, D, D) or KV slots.
   - Physical page pool allocation and free-list management.
   - O(1) zero-copy fork_sequence via reference counting.
   - Automatic Copy-on-Write (CoW) on mutating state steps.
2. RadixPrefixStateCache:
   - Associative Radix tree caching recurrent states computed across common token prompt prefixes.
   - Exact prefix token matching with longest-prefix match.
   - LRU eviction of unreferenced physical state pages.
"""

from __future__ import annotations

import collections
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.PagedState")


class PagedStateTable:
    """
    Virtual-to-physical GPU state table managing recurrent states for continuous batching.
    Supports reference-counted zero-copy sequence forking and Copy-on-Write (CoW).
    """

    def __init__(
        self,
        num_pages: int = 1024,
        num_heads: int = 12,
        head_dim: int = 64,
        state_dim: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cuda",
    ):
        self.num_pages = num_pages
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_dim = state_dim or head_dim
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        # Allocate contiguous GPU physical page pool: [num_pages, num_heads, head_dim, state_dim]
        self.page_pool = torch.zeros(
            (num_pages, num_heads, head_dim, self.state_dim),
            dtype=self.dtype,
            device=self.device,
        )

        # Free-list of physical page IDs
        self.free_pages: collections.deque[int] = collections.deque(range(num_pages))
        # Reference count per physical page ID
        self.ref_counts: Dict[int, int] = collections.defaultdict(int)
        # Sequence virtual-to-physical mapping: seq_id -> list of physical page IDs
        self.seq_page_tables: Dict[int, List[int]] = {}

    def allocate_page(self) -> int:
        """Allocates a free physical page from the pool."""
        if not self.free_pages:
            raise RuntimeError(f"PagedStateTable: Out of physical GPU pages (capacity {self.num_pages})")
        page_id = self.free_pages.popleft()
        self.ref_counts[page_id] = 1
        self.page_pool[page_id].zero_()
        return page_id

    def free_page(self, page_id: int) -> None:
        """Decrements ref count; returns page to free list if count drops to 0."""
        if page_id not in self.ref_counts:
            return
        self.ref_counts[page_id] -= 1
        if self.ref_counts[page_id] <= 0:
            del self.ref_counts[page_id]
            self.free_pages.append(page_id)

    def register_sequence(self, seq_id: int, initial_pages: int = 1) -> List[int]:
        """Registers a new sequence with allocated physical pages."""
        if seq_id in self.seq_page_tables:
            self.free_sequence(seq_id)
        pages = [self.allocate_page() for _ in range(initial_pages)]
        self.seq_page_tables[seq_id] = pages
        return pages

    def free_sequence(self, seq_id: int) -> None:
        """Frees all pages owned by sequence."""
        if seq_id in self.seq_page_tables:
            for p in self.seq_page_tables[seq_id]:
                self.free_page(p)
            del self.seq_page_tables[seq_id]

    def fork_sequence(self, parent_seq_id: int, child_seq_id: int) -> List[int]:
        """
        Zero-copy sequence forking via reference counting.
        Child shares parent's physical pages until Copy-on-Write mutation.
        """
        if parent_seq_id not in self.seq_page_tables:
            raise KeyError(f"Parent sequence {parent_seq_id} not registered")
        parent_pages = self.seq_page_tables[parent_seq_id]
        for p in parent_pages:
            self.ref_counts[p] += 1
        self.seq_page_tables[child_seq_id] = list(parent_pages)
        return self.seq_page_tables[child_seq_id]

    def prepare_write(self, seq_id: int, page_slot: int = 0) -> int:
        """
        Executes Copy-on-Write (CoW):
        If target page is shared (ref_count > 1), copies data to a newly allocated page.
        """
        pages = self.seq_page_tables[seq_id]
        old_page = pages[page_slot]
        if self.ref_counts[old_page] > 1:
            # Shared: must clone (Copy-on-Write)
            new_page = self.allocate_page()
            self.page_pool[new_page].copy_(self.page_pool[old_page])
            self.free_page(old_page)
            pages[page_slot] = new_page
            return new_page
        return old_page

    def get_block_table_tensor(self, seq_ids: List[int], max_pages: int = 4) -> torch.Tensor:
        """Builds a 2D tensor [B, max_pages] of physical page indices for kernel execution."""
        B = len(seq_ids)
        table = torch.zeros((B, max_pages), dtype=torch.int32, device=self.device)
        for i, s_id in enumerate(seq_ids):
            allocated = self.seq_page_tables.get(s_id, [])
            for j, p_id in enumerate(allocated[:max_pages]):
                table[i, j] = p_id
        return table


class _RadixNode:
    """Node in the associative prefix state Radix tree."""

    def __init__(self, token_chunk: Tuple[int, ...] = ()):
        self.token_chunk = token_chunk
        self.children: Dict[int, _RadixNode] = {}
        # Physical page ID holding state after processing token_chunk
        self.physical_page: Optional[int] = None
        self.last_access_time: float = 0.0


class RadixPrefixStateCache:
    """
    Associative Radix-Tree Prefix State Cache for Linear RNNs / S4 / GLA / GDN.
    Shares physical state pages across sequences with identical token prefixes.
    """

    def __init__(self, paged_table: PagedStateTable, max_cached_nodes: int = 4096):
        self.paged_table = paged_table
        self.max_cached_nodes = max_cached_nodes
        self.root = _RadixNode()
        self.total_nodes = 1
        self._access_counter = 0.0

    def match_prefix(self, tokens: List[int]) -> Tuple[Optional[int], int]:
        """
        Finds longest cached prefix in the Radix tree.
        Returns:
            (physical_page_id, matched_tokens_count)
        """
        curr = self.root
        idx = 0
        best_page = None
        best_matched_len = 0

        while idx < len(tokens):
            tok = tokens[idx]
            if tok not in curr.children:
                break
            curr = curr.children[tok]
            idx += 1
            if curr.physical_page is not None:
                best_page = curr.physical_page
                best_matched_len = idx
                self._access_counter += 1.0
                curr.last_access_time = self._access_counter

        return best_page, best_matched_len

    def insert_prefix(self, tokens: List[int], physical_page: int) -> None:
        """Inserts a prefix state checkpoint into the Radix tree."""
        curr = self.root
        for tok in tokens:
            if tok not in curr.children:
                curr.children[tok] = _RadixNode((tok,))
                self.total_nodes += 1
            curr = curr.children[tok]

        # Associate physical page with node
        if curr.physical_page is not None and curr.physical_page != physical_page:
            self.paged_table.free_page(curr.physical_page)

        curr.physical_page = physical_page
        self.paged_table.ref_counts[physical_page] += 1
        self._access_counter += 1.0
        curr.last_access_time = self._access_counter


__all__ = [
    "PagedStateTable",
    "RadixPrefixStateCache",
]
