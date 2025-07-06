#!/usr/bin/env python3
"""
Comprehensive test suite for inputs.py to verify roundtrip correctness
and identify any bugs in the space implementations.
"""

import numpy as np
import torch
from enum import IntEnum
from inputs import (
    EnumSpace, IntSpace, SequenceSpace, FixedVecSpace, 
    TupleAddSpace, TupleConcatSpace, DictSpace
)
import slaythespire as sts


class TestEnum(IntEnum):
    A = 0
    B = 1 
    C = 2
    D = 3


def test_enum_space_roundtrip():
    """Test EnumSpace for roundtrip correctness."""
    print("Testing EnumSpace roundtrip...")
    space = EnumSpace(TestEnum)
    rng = np.random.default_rng(42)
    
    # Test with multiple samples
    for i in range(100):
        sample = space.sample(rng)
        # EnumSpace is a ScalarSpace, so no path operations
        print(f"  Sample {i}: {sample}")
    
    print("EnumSpace test passed")


def test_int_space_roundtrip():
    """Test IntSpace for roundtrip correctness."""
    print("Testing IntSpace roundtrip...")
    space = IntSpace(10)
    rng = np.random.default_rng(42)
    
    # Test with multiple samples
    for i in range(100):
        sample = space.sample(rng)
        assert 0 <= sample < 10, f"Sample {sample} out of bounds"
        print(f"  Sample {i}: {sample}")
    
    print("IntSpace test passed")


def test_sequence_space_roundtrip():
    """Test SequenceSpace for roundtrip correctness."""
    print("Testing SequenceSpace roundtrip...")
    element_space = EnumSpace(TestEnum)
    space = SequenceSpace(element_space)
    rng = np.random.default_rng(42)
    
    # Test with multiple samples
    for i in range(20):
        sample = space.sample(rng)
        print(f"  Sample {i}: {sample}")
        
        # Create a mock tensor representation for testing
        mock_tensor = {
            'value': torch.zeros(1, len(sample), dtype=torch.int32),
            'mask': torch.zeros(1, len(sample), dtype=torch.bool)
        }
        
        # Test roundtrip for each valid index
        for idx in range(len(sample)):
            path = space.ix_to_path(mock_tensor, idx)
            recovered_idx = space.path_to_ix(mock_tensor, path)
            assert recovered_idx == idx, f"Roundtrip failed: {idx} -> {path} -> {recovered_idx}"
        
        # Test length calculation
        assert space.length(mock_tensor) == len(sample), f"Length mismatch: expected {len(sample)}, got {space.length(mock_tensor)}"
    
    print("SequenceSpace test passed")


def test_fixed_vec_space_roundtrip():
    """Test FixedVecSpace for roundtrip correctness."""
    print("Testing FixedVecSpace roundtrip...")
    limits = [10, 5, 3, 7]
    space = FixedVecSpace(limits)
    rng = np.random.default_rng(42)
    
    # Test with multiple samples
    for i in range(20):
        sample = space.sample(rng)
        print(f"  Sample {i}: {sample}")
        assert len(sample) == len(limits), f"Sample length mismatch"
        
        for j, (val, limit) in enumerate(zip(sample, limits)):
            assert 0 <= val < limit, f"Value {val} at index {j} exceeds limit {limit}"
        
        # Test roundtrip - FixedVecSpace only has one valid index (0)
        length = space.length(sample)
        assert length == 1, f"FixedVecSpace should have length 1, got {length}"
        
        path = space.ix_to_path(sample, 0)
        recovered_idx = space.path_to_ix(sample, path)
        assert recovered_idx == 0, f"Roundtrip failed: 0 -> {path} -> {recovered_idx}"
    
    print("FixedVecSpace test passed")


def test_tuple_add_space_roundtrip():
    """Test TupleAddSpace for roundtrip correctness."""
    print("Testing TupleAddSpace roundtrip...")
    spaces = [EnumSpace(TestEnum), IntSpace(5)]
    space = TupleAddSpace(*spaces)
    rng = np.random.default_rng(42)
    
    # Test with multiple samples
    for i in range(20):
        sample = space.sample(rng)
        print(f"  Sample {i}: {sample}")
        assert len(sample) == 2, f"Tuple should have 2 elements"
        assert sample[0] in TestEnum.__members__.values(), f"First element should be TestEnum value, got {sample[0]} of type {type(sample[0])}"
        if not (isinstance(sample[1], int) and 0 <= sample[1] < 5):
            print(f"DEBUG: sample[1] = {sample[1]}, type = {type(sample[1])}")
        assert isinstance(sample[1], (int, np.integer)) and 0 <= sample[1] < 5, f"Second element should be int in [0,5), got {sample[1]} of type {type(sample[1])}"
    
    print("TupleAddSpace test passed")


def test_tuple_concat_space_roundtrip():
    """Test TupleConcatSpace for roundtrip correctness."""
    print("Testing TupleConcatSpace roundtrip...")
    seq_space1 = SequenceSpace(EnumSpace(TestEnum))
    seq_space2 = SequenceSpace(IntSpace(3))
    space = TupleConcatSpace(seq_space1, seq_space2)
    rng = np.random.default_rng(42)
    
    # Test with multiple samples
    for i in range(10):
        sample = space.sample(rng)
        print(f"  Sample {i}: lengths = ({len(sample[0])}, {len(sample[1])})")
        assert len(sample) == 2, f"Tuple should have 2 elements"
        
        # Create mock tensor representations
        mock_seq1 = {
            'value': torch.zeros(1, len(sample[0]), dtype=torch.int32),
            'mask': torch.zeros(1, len(sample[0]), dtype=torch.bool)
        }
        mock_seq2 = {
            'value': torch.zeros(1, len(sample[1]), dtype=torch.int32), 
            'mask': torch.zeros(1, len(sample[1]), dtype=torch.bool)
        }
        mock_tuple = (mock_seq1, mock_seq2)
        
        # Test roundtrip for several indices
        total_length = space.length(mock_tuple)
        test_indices = list(range(min(total_length, 10)))  # Test first 10 indices
        
        for idx in test_indices:
            path = space.ix_to_path(mock_tuple, idx)
            recovered_idx = space.path_to_ix(mock_tuple, path)
            assert recovered_idx == idx, f"Roundtrip failed: {idx} -> {path} -> {recovered_idx}"
    
    print("TupleConcatSpace test passed")


def test_dict_space_roundtrip():
    """Test DictSpace for roundtrip correctness - this is the critical test."""
    print("Testing DictSpace roundtrip...")
    
    # Create a DictSpace similar to the one used in the actual code but with TestEnum
    spaces = {
        'cards': SequenceSpace(TupleAddSpace(EnumSpace(TestEnum), IntSpace(21))),
        'relics': SequenceSpace(EnumSpace(TestEnum)),
        'potions': SequenceSpace(EnumSpace(TestEnum)),
        'fixed': SequenceSpace(EnumSpace(TestEnum))
    }
    space = DictSpace(spaces)
    rng = np.random.default_rng(42)
    
    # Test with multiple samples
    for i in range(10):
        sample = space.sample(rng)
        print(f"\n  Sample {i}:")
        
        # Create mock tensor representations for each component
        mock_dict = {}
        total_expected_length = 0
        
        for key, seq in sample.items():
            mock_dict[key] = {
                'value': torch.zeros(1, len(seq), 2 if key == 'cards' else 1, dtype=torch.int32),
                'mask': torch.zeros(1, len(seq), dtype=torch.bool)
            }
            seq_length = len(seq)
            total_expected_length += seq_length
            print(f"    {key}: length={seq_length}")
        
        # Verify total length calculation
        calculated_length = space.length(mock_dict)
        print(f"    Total length: expected={total_expected_length}, calculated={calculated_length}")
        assert calculated_length == total_expected_length, f"Length mismatch"
        
        # Test roundtrip for several indices
        test_indices = list(range(min(calculated_length, 20)))  # Test up to 20 indices
        
        for idx in test_indices:
            try:
                path = space.ix_to_path(mock_dict, idx)
                recovered_idx = space.path_to_ix(mock_dict, path)
                print(f"    Index {idx}: {path} -> {recovered_idx}")
                assert recovered_idx == idx, f"Roundtrip failed: {idx} -> {path} -> {recovered_idx}"
            except Exception as e:
                print(f"    ERROR at index {idx}: {e}")
                raise
    
    print("DictSpace test passed")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("Testing edge cases...")
    
    # Test empty sequences
    seq_space = SequenceSpace(EnumSpace(TestEnum))
    empty_seq = {'value': torch.zeros(1, 0, dtype=torch.int32), 'mask': torch.ones(1, 0, dtype=torch.bool)}
    assert seq_space.length(empty_seq) == 0, "Empty sequence should have length 0"
    
    # Test single element sequences
    single_seq = {'value': torch.zeros(1, 1, dtype=torch.int32), 'mask': torch.zeros(1, 1, dtype=torch.bool)}
    assert seq_space.length(single_seq) == 1, "Single element sequence should have length 1"
    path = seq_space.ix_to_path(single_seq, 0)
    assert path == [0], f"Single element path should be [0], got {path}"
    
    print("Edge cases test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING COMPREHENSIVE INPUTS.PY TESTS")
    print("=" * 60)
    
    try:
        test_enum_space_roundtrip()
        test_int_space_roundtrip()
        test_sequence_space_roundtrip()
        test_fixed_vec_space_roundtrip()
        test_tuple_add_space_roundtrip()
        test_tuple_concat_space_roundtrip()
        test_dict_space_roundtrip()  # This is the critical test
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()