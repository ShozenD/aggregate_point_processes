import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from aggpp.models._utils import create_interval_dict, find_overlap_intervals

def test_find_overlap_intervals_no_overlap():
  """Test intervals with no overlaps"""
  interval_series = pd.Series([pd.Interval(1, 2, closed='left'), pd.Interval(3, 4, closed='left'), 
                              pd.Interval(5, 6, closed='left'), pd.Interval(7, 8, closed='left')], 
                             dtype='category')
  intervals = create_interval_dict(interval_series)
  result = find_overlap_intervals(intervals)
  assert len(result) == 0  # No overlaps expected

def test_find_overlap_intervals_single_overlap():
  """Test intervals with single overlap"""
  interval_series = pd.Series([pd.Interval(1, 4, closed='left'), pd.Interval(3, 5, closed='left')], 
                             dtype='category')
  intervals = create_interval_dict(interval_series)
  result = find_overlap_intervals(intervals)
  assert len(result) == 1
  assert (0, 1) in result  # These intervals overlap
  
def test_find_overlap_intervals_multiple_overlaps():
  """Test intervals with multiple overlaps"""
  interval_series = pd.Series([pd.Interval(1, 3, closed='left'), pd.Interval(2, 4, closed='left'), 
                              pd.Interval(3, 5, closed='left')], dtype='category')
  intervals = create_interval_dict(interval_series)
  result = find_overlap_intervals(intervals)
  assert len(result) >= 1  # At least one overlap expected
  
def test_find_overlap_intervals_nested():
  """Test nested intervals"""
  interval_series = pd.Series([pd.Interval(1, 5, closed='left'), pd.Interval(2, 3, closed='left')], 
                             dtype='category')
  intervals = create_interval_dict(interval_series)
  result = find_overlap_intervals(intervals)
  assert len(result) == 1  # Nested intervals should overlap
  
def test_find_overlap_intervals_identical():
  """Test identical intervals"""
  interval_series = pd.Series([pd.Interval(1, 3, closed='left'), pd.Interval(1, 3, closed='left')], 
                             dtype='category')
  intervals = create_interval_dict(interval_series)
  result = find_overlap_intervals(intervals)
  # Note: identical intervals in pandas categorical will be deduplicated, so no overlaps expected
  assert len(result) == 0
  
def test_find_overlap_intervals_empty():
  """Test empty intervals"""
  interval_series = pd.Series([], dtype='category')
  intervals = create_interval_dict(interval_series)
  result = find_overlap_intervals(intervals)
  assert len(result) == 0
  
def test_find_overlap_intervals_single_interval():
  """Test single interval"""
  interval_series = pd.Series([pd.Interval(1, 3, closed='left')], dtype='category')
  intervals = create_interval_dict(interval_series)
  result = find_overlap_intervals(intervals)
  assert len(result) == 0

def test_find_overlap_intervals_touching_boundaries():
  """Test intervals that touch at boundaries"""
  interval_series = pd.Series([pd.Interval(1, 3, closed='left'), pd.Interval(3, 5, closed='left')], 
                             dtype='category')
  intervals = create_interval_dict(interval_series)
  result = find_overlap_intervals(intervals)
  # Intervals [1,3) and [3,5) don't overlap since intervals are left-closed
  assert len(result) == 0

def test_create_interval_dict_with_lists():
  """Test create_interval_dict with list inputs"""
  intervals = create_interval_dict([1, 3], [3, 5])
  result = find_overlap_intervals(intervals)
  # These intervals [1,3) and [3,5) should not overlap
  assert len(result) == 0

def test_create_interval_dict_with_overlapping_lists():
  """Test create_interval_dict with overlapping list inputs"""
  intervals = create_interval_dict([1, 3], [4, 5])
  result = find_overlap_intervals(intervals)
  # These intervals [1,4) and [3,5) should overlap
  assert len(result) == 1
