from collections import defaultdict
import numpy as np
import random
from typing import List, Dict, Set, Tuple, Iterable, Optional

from sklearn.manifold import TSNE
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools

TupleSet = Set[Tuple[int, ...]]


# Class to define a structure with its domain and relations, in the model theoretical sense

class Structure:
    """
    Finite structure with finite domain {0,...,n-1} and  relations.
    Relations are sets of tuples of domain elements.
    0-ary relations are represented as {()} (present) or set() (absent).
    """
    def __init__(self, domain_size: int):
        self.n = domain_size
        self.domain = tuple(range(domain_size))
        self.relations: Dict[str, TupleSet] = {}

    def add_relation(self, name: str, tuples: Iterable[Tuple[int, ...]]):
        self.relations[name] = set(tuples)

    def add_0ary(self, name: str, truth: bool):
        self.relations[name] = {()} if truth else set()

    def arity(self, name: str) -> int:
        R = self.relations[name]
        return 0 if not R or next(iter(R)) == () else len(next(iter(R)))


def preserves_atomic(A: Structure, B: Structure, fAB: Dict[int, int]) -> bool:
    """
    Check atomic back-and-forth on the current selected elements.
    For every relation name R and every tuple t over the elements,
    membership is preserved under the partial bijection fAB.
    Also enforce injectivity on the mapping.
    """
    # Injectivity of partial bijection
    if len(set(fAB.values())) != len(fAB):
        return False

    # Build reverse map (for the 'forth' direction)
    fBA = {v: u for u, v in fAB.items()}

    A_rels = A.relations
    B_rels = B.relations

    for name in set(A_rels.keys()).union(B_rels.keys()):
        RA = A_rels.get(name, set())
        RB = B_rels.get(name, set())

        # 0-ary: presence/absence must match
        if A.arity(name) == 0:
            if (RA == set()) != (RB == set()):
                return False
            continue

        # For arity r >= 1, check only tuples fully within selected ("pebbled") elements
        r = A.arity(name)
        # Build the set of A-elements and B-elements
        pebbled_A = set(fAB.keys())
        pebbled_B = set(fAB.values())

        # Enumerate all tuples over pebbled_A that appear in RA
        for tA in filter(lambda t: all(x in pebbled_A for x in t), RA):
            tB = tuple(fAB[x] for x in tA)
            if tB not in RB:
                return False

        # forth part: tuples over pebbled_B that appear in RB must map back into RA
        for tB in filter(lambda t: all(y in pebbled_B for y in t), RB):
            tA = tuple(fBA[y] for y in tB)
            if tA not in RA:
                return False

    return True


def ef_game_basic(A: Structure, B: Structure, k: int, rounds: int) -> str:
    """
    Ehrenfeucht–Fraïssé game for plain FO (no counting/mod).
    Returns 'Duplicator' if Duplicator has a survival strategy up to `rounds`,
    else 'Spoiler'. Uses simple state expansion with pruning.
    NOTE: for simplicity, assume rounds <= k (no pebble reuse).
    """
    assert rounds <= k, "This simple version assumes rounds <= k (no reuse)."

    # Each state is a partial bijection fAB: A_pebbled -> B_pebbled
    states: List[Dict[int, int]] = [ {} ]

    for _ in range(rounds):
        next_states: List[Dict[int, int]] = []

        for fAB in states:
            # Spoiler may play on A or B
            for side in ('A', 'B'):
                if side == 'A':
                    for a in A.domain:
                        if a in fAB:  # already pebbled at A-side : skip fresh pebble
                            continue
                        # Duplicator replies with some b in B not already used
                        for b in B.domain:
                            if b in fAB.values():
                                continue
                            new_map = dict(fAB)
                            new_map[a] = b
                            if preserves_atomic(A, B, new_map):
                                next_states.append(new_map)
                                # print(next_states)
                else:  # Spoiler plays on B; symmetric via reverse bijection
                    fBA = {v: u for u, v in fAB.items()}
                    for b in B.domain:
                        if b in fBA:  # already pebbled at B-side
                            continue
                        for a in A.domain:
                            if a in fAB:
                                continue
                            new_map = dict(fAB)
                            new_map[a] = b
                            if preserves_atomic(A, B, new_map):
                                next_states.append(new_map)
        # Duplicator loses if no legal replies remain
        if not next_states:
            return 'Spoiler'

        canonical = {}
        for m in next_states:
            key = tuple(sorted(m.items()))
            canonical[key] = m
        states = list(canonical.values())

    return 'Duplicator'


def string_structure(s: str, include_parity_0ary: bool = False) -> Structure:
    n = len(s)
    A = Structure(n)
    # unary predicates
    A.add_relation('One', ((i,) for i,ch in enumerate(s) if ch == '1'))
    A.add_relation('First', {(0,)} if n > 0 else set())
    A.add_relation('Last', { (n-1,) } if n > 0 else set())
    # successor
    A.add_relation('Succ', ((i, i+1) for i in range(n-1)))
    # optional 0-ary "EvenOnes" to simulate a MOD2 oracle
    if include_parity_0ary:
        even = (sum(1 for ch in s if ch=='1') % 2 == 0)
        A.add_0ary('EvenOnes', even)
    return A

# Modified EF game: Duplicator wins if it has at least one potential strategy. However, there is an option to stochastically limit the states it can explore. This offers some flexibility in varying the logical power of this EF game
def ef_game(
    A: Structure,
    B: Structure,
    k: int,
    rounds: int,
    max_states: Optional[int] = None,
    spoiler_samples: Optional[int] = None,
    seed: Optional[int] = None
) -> str:
    """
    Ehrenfeucht–Fraïssé game with optional bounded memory and stochastic behavior.
    Returns 'Duplicator' or 'Spoiler'.
    """
    assert rounds <= k, "This version assumes no pebble reuse."

    if seed is not None:
        random.seed(seed)

    states: List[Dict[int, int]] = [ {} ] # empty initial mapping

    for _ in range(rounds):
        next_states: List[Dict[int, int]] = []

        for fAB in states:
            # Try both sides (A or B) for Spoiler move - Spoiler can pick from structure A or B
            for side in ('A', 'B'):
                if side == 'A': # suppose spoiler picks A
                    unpebbled_A = [a for a in A.domain if a not in fAB]
                    a_choices = (
                        random.sample(unpebbled_A, k=min(spoiler_samples, len(unpebbled_A)))
                        if spoiler_samples else unpebbled_A
                    )
                    for a in a_choices:
                        for b in B.domain: # for every choice of spoiler, this is the duplicator attempted response
                            if b in fAB.values():
                                continue
                            new_map = dict(fAB)
                            new_map[a] = b
                            # if a==0: print('a',new_map,preserves_atomic(A, B, new_map)) - debug print
                            if preserves_atomic(A, B, new_map):
                                next_states.append(new_map)
                else: # spoiler picks B
                    fBA = {v: u for u, v in fAB.items()}
                    unpebbled_B = [b for b in B.domain if b not in fBA]
                    b_choices = (
                        random.sample(unpebbled_B, k=min(spoiler_samples, len(unpebbled_B)))
                        if spoiler_samples else unpebbled_B
                    )
                    for b in b_choices:
                        for a in A.domain: # for every choice of spoiler, this is the duplicator attempted response
                            if a in fAB:
                                continue
                            new_map = dict(fAB)
                            new_map[a] = b
                            if preserves_atomic(A, B, new_map):
                                next_states.append(new_map)
            
        if not next_states:
            return 'Spoiler'

        # Deduplicate by canonical key
        canonical = {}
        for m in next_states:
            key = tuple(sorted(m.items()))
            canonical[key] = m
        states = list(canonical.values())

        if max_states is not None and len(states) > max_states:
            states = random.sample(states, max_states)

    return 'Duplicator'

# function to generate equivalence matrices: if two strings are equivalent per EF game, they form an edge
def ef_equivalence_matrix(strings: List[str], k: int, rounds: int, 
                          include_parity: bool = False, max_states=None, spoiler_samples=None) -> np.ndarray:
    n = len(strings)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            A = string_structure(strings[i], include_parity)
            B = string_structure(strings[j], include_parity)
            result = ef_game(A, B, k=k, rounds=rounds, max_states=max_states, spoiler_samples=spoiler_samples)
            # if include_parity and i!=j and result=='Duplicator':
            #     print(strings[i],strings[j],result)
            if result == 'Duplicator':
                mat[i, j] = mat[j, i] = 1
    return mat


def ef_equivalence_labels(strings: List[str], k: int = 2, rounds: int = 2,max_states: int = None) -> Dict[Tuple[str, str], int]:
    labels = {}
    for s1, s2 in itertools.combinations_with_replacement(strings, 2):
        A = string_structure(s1, include_parity_0ary=True)
        B = string_structure(s2, include_parity_0ary=True)
        label = int(ef_game(A, B, k=k, rounds=rounds,max_states=max_states) == 'Duplicator')
        labels[(s1, s2)] = labels[(s2, s1)] = label
    return labels

# Step 1: Build EF-graph (logical environment)
def build_ef_graph(strings, k=2, rounds=2):
    G = nx.Graph()
    G.add_nodes_from(strings)
    for s1, s2 in itertools.combinations(strings, 2):
        A = string_structure(s1, include_parity_0ary=True)
        B = string_structure(s2, include_parity_0ary=True)
        if ef_game(A, B, k=k, rounds=rounds) == 'Duplicator':
            G.add_edge(s1, s2)
    return G
