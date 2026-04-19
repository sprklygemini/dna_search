from __future__ import annotations
import random
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass


def generate_dna(length: int) -> str:
    bases = ["A", "C", "G", "T"]
    return "".join(random.choice(bases) for _ in range(length))


dna = generate_dna(100_000)


# =========================================================
#  DNA SEARCH PIPELINE
#  Algorithm A: Merge Sort on (k-mer_code, position) pairs
#  Algorithm B: Binary Search over sorted k-mer index
#  Algorithm C: Dynamic Programming (Edit Distance)
# =========================================================


DNA_MAP: Dict[str, int] = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


@dataclass(frozen=True)
class SearchResult:
    position: int
    matched_text: str
    edit_distance: int


class DNAFuzzySearchSystem:
    def __init__(self, dna: str, k: int = 8) -> None:
        self.dna = self._sanitize_dna(dna)
        self.k = k

        if len(self.dna) < self.k:
            raise ValueError("DNA length must be at least k.")

        # Compact encoded DNA: 1 integer per base.
        # In real systems you would go even smaller (2 bits/base),
        # but this keeps the code readable for presentation.
        self.encoded_dna: List[int] = [DNA_MAP[ch] for ch in self.dna]

        # Sorted list of (kmer_code, position)
        self.index: List[Tuple[int, int]] = self._build_index()

    # ---------------------------
    # Input cleaning
    # ---------------------------
    def _sanitize_dna(self, dna: str) -> str:
        dna = dna.upper().strip()
        if not dna:
            raise ValueError("DNA string is empty.")

        for ch in dna:
            if ch not in DNA_MAP:
                raise ValueError(f"Invalid DNA character: {ch}")
        return dna

    # ---------------------------
    # K-mer encoding
    # ---------------------------
    def _encode_kmer(self, s: str) -> int:
        code = 0
        for ch in s:
            code = (code << 2) | DNA_MAP[ch]
        return code

    # ---------------------------
    # Algorithm A: Merge Sort
    # ---------------------------
    def _merge_sort(self, arr: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = self._merge_sort(arr[:mid])
        right = self._merge_sort(arr[mid:])
        return self._merge(left, right)

    def _merge(
        self,
        left: List[Tuple[int, int]],
        right: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        merged: List[Tuple[int, int]] = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i][0] <= right[j][0]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1

        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    def _build_index(self) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []

        for pos in range(len(self.dna) - self.k + 1):
            kmer = self.dna[pos:pos + self.k]
            code = self._encode_kmer(kmer)
            pairs.append((code, pos))

        return self._merge_sort(pairs)

    # ---------------------------
    # Algorithm B: Binary Search
    # Find [left, right) range of equal k-mer codes
    # ---------------------------
    def _lower_bound(self, target: int) -> int:
        l, r = 0, len(self.index)
        while l < r:
            m = (l + r) // 2
            if self.index[m][0] >= target:
                r = m
            else:
                l = m + 1
        return l

    def _upper_bound(self, target: int) -> int:
        l, r = 0, len(self.index)
        while l < r:
            m = (l + r) // 2
            if self.index[m][0] > target:
                r = m
            else:
                l = m + 1
        return l

    def _find_seed_positions(self, seed: str) -> List[int]:
        code = self._encode_kmer(seed)
        left = self._lower_bound(code)
        right = self._upper_bound(code)
        return [self.index[i][1] for i in range(left, right)]

    # ---------------------------
    # Algorithm C: Edit Distance DP
    # Space-optimized 2-row version
    # ---------------------------
    def _edit_distance_with_cutoff(self, a: str, b: str, max_dist: int) -> int:
        n, m = len(a), len(b)

        if abs(n - m) > max_dist:
            return max_dist + 1

        prev = list(range(m + 1))

        for i in range(1, n + 1):
            curr = [i] + [0] * m
            row_min = curr[0]

            for j in range(1, m + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(
                        prev[j],      # delete
                        curr[j - 1],  # insert
                        prev[j - 1],  # replace
                    )
                row_min = min(row_min, curr[j])

            if row_min > max_dist:
                return max_dist + 1

            prev = curr

        return prev[m]

    # ---------------------------
    # Full pipeline
    # ---------------------------
    def search(self, query: str, max_mismatches: int = 2) -> List[SearchResult]:
        query = self._sanitize_dna(query)

        if len(query) < self.k:
            raise ValueError("Query length must be at least k.")

        # Break query into overlapping seeds
        seeds: List[Tuple[int, str]] = []
        for i in range(len(query) - self.k + 1):
            seeds.append((i, query[i:i + self.k]))

        candidate_starts: Set[int] = set()

        # Stage B: binary search candidate retrieval
        for offset, seed in seeds:
            positions = self._find_seed_positions(seed)

            for pos in positions:
                start = pos - offset
                if 0 <= start <= len(self.dna) - len(query):
                    candidate_starts.add(start)

        # Stage C: fuzzy verification
        results: List[SearchResult] = []
        for start in sorted(candidate_starts):
            window = self.dna[start:start + len(query)]
            dist = self._edit_distance_with_cutoff(
                query, window, max_mismatches)
            if dist <= max_mismatches:
                results.append(
                    SearchResult(
                        position=start,
                        matched_text=window,
                        edit_distance=dist,
                    )
                )

        return results


# =========================================================
# Demo
# =========================================================
if __name__ == "__main__":
    dna = "ACGTACGTGACCTGACGTACGTTACGTGACCTGAACGTACGT"
    system = DNAFuzzySearchSystem(dna=dna, k=4)

    queries = [
        ("ACGTGACC", 1),
        ("ACGTTACC", 2),
        ("TTTTTTTT", 2),
    ]

    for query, max_mm in queries:
        print(f"\nQuery: {query} | max mismatches: {max_mm}")
        matches = system.search(query, max_mismatches=max_mm)

        if not matches:
            print("No matches found.")
        else:
            for match in matches:
                print(
                    f"Position={match.position}, "
                    f"Matched='{match.matched_text}', "
                    f"EditDistance={match.edit_distance}"
                )
