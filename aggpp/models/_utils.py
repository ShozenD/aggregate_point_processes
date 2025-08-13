import numpy as np
import pandas as pd

def create_interval_index_array(intervals_series, grid_start=0, grid_end=None, grid_step=1):
		"""
		Create an index array that maps intervals to points on a fine integer grid.
		
		Parameters:
		-----------
		intervals_series : pandas.Series
				A pandas Series of categorical dtype containing pd.Interval objects
				that are closed on the left and open on the right [left, right).
		grid_start : int, default 0
				The starting point of the integer grid.
		grid_end : int, optional
				The ending point of the integer grid. If None, inferred from intervals.
		grid_step : int, default 1
				The step size for the integer grid.
		
		Returns:
		--------
		list
				A list where each element corresponds to a grid point and
				contains a list of interval indices that contain that grid point.
				Points not in any interval get an empty list.
		"""
		
		# Get unique intervals and their categorical codes
		unique_intervals = intervals_series.cat.categories
		
		# Determine grid bounds if not provided
		if grid_end is None:
				max_right = max(interval.right for interval in unique_intervals)
				grid_end = int(np.floor(max_right))

		# Create the fine integer grid
		grid_points = np.arange(grid_start, grid_end + grid_step, grid_step)
		
		# Initialize index array with empty lists for each grid point
		index_array = [[] for _ in range(len(grid_points))]
		
		# For each unique interval, find which grid points it contains
		for i, interval in enumerate(unique_intervals):
				# Find grid points that fall within this interval [left, right)
				mask = (grid_points >= interval.left) & (grid_points < interval.right)
				
				# Add the interval index to all grid points that fall within this interval
				for j, in_interval in enumerate(mask):
						if in_interval:
								index_array[j].append(i)
		
		return index_array

def create_interval_grid_index_map(intervals_series, grid_start=0, grid_end=None, grid_step=1):
		"""
		Create an index array that maps intervals to points on a fine integer grid.
		
		Parameters:
		-----------
		intervals_series : pandas.Series
				A pandas Series of categorical dtype containing pd.Interval objects
				that are closed on the left and open on the right [left, right).
		grid_start : int, default 0
				The starting point of the integer grid.
		grid_end : int, optional
				The ending point of the integer grid. If None, inferred from intervals.
		grid_step : int, default 1
				The step size for the integer grid.
		
		Returns:
		--------
		list
				A list where each element corresponds to a grid point and
				contains a list of interval indices that contain that grid point.
				Points not in any interval get an empty list.
		"""
		
		# Get unique intervals and their categorical codes
		unique_intervals = intervals_series.cat.categories
		unique_interval_codes = intervals_series.cat.codes.sort_values().unique()
		
		# Determine grid bounds if not provided
		if grid_end is None:
				max_right = max(interval.right for interval in unique_intervals)
				grid_end = int(np.floor(max_right))

		# Create the fine integer grid
		grid_points = np.arange(grid_start, grid_end + grid_step, grid_step)
		grid_index = np.arange(len(grid_points))
		
		# Initialize index array with empty lists for each grid point
		interval_grid_index_map = dict()
		for code, interval in zip(unique_interval_codes, unique_intervals):
			mask = (grid_points >= interval.left) & (grid_points < interval.right)
			interval_grid_index_map[code] = np.array(grid_index[mask])

		return interval_grid_index_map

def create_interval_dict(x: pd.Series):
	"""
	Create a dictionary mapping interval codes to their corresponding intervals.
	
	Parameters
	----------
	x : pd.Series
		A pandas Series containing interval codes.
	
	Returns
	-------
	dict: A dictionary where the keys are the interval codes and
				the values are the corresponding intervals.
	"""
	codes = x.cat.codes.sort_values().unique()
	intervals = x.cat.categories
	return {code: interval for code, interval in zip(codes, intervals)}

def find_overlapping_intervals(interval_dict: dict):
	"""
	Find overlapping intervals in a sorted list of intervals.

	Returns
	-------
	dict: A dictionary where the keys are the interval codes and
				the values are lists of interval codes that overlap with the key interval.
	"""
	overlap_record = {}
	for i, interval in interval_dict.items():
		for j, other_interval in interval_dict.items():
			if i != j and interval.overlaps(other_interval):
				overlap_record.setdefault(i, []).append(j)
	 
	return overlap_record

def create_overlap_weights(interval_dict: dict,
							 overlap_dict: dict,
							 sampling_effort_dict: dict):
	"""
	Create overlap weights for intervals based on overlapping regions and sampling effort.
	
	Parameters
	----------
	interval_dict : dict
		Dictionary mapping interval codes to their corresponding intervals.
	overlap_dict : dict
		Dictionary mapping interval codes to lists of overlapping interval codes.
	sampling_effort_dict : dict
		Dictionary mapping interval codes to their sampling effort values.
	
	Returns
	-------
	dict
		Dictionary mapping interval codes to their overlap weight arrays.
	"""
	overlap_weights = {}
	
	for code, interval in interval_dict.items():
		left, right = interval.left, interval.right
		sampling_effort = sampling_effort_dict[code]
		
		# Initialize total effort array for this interval
		total_effort = np.full(right - left, sampling_effort, dtype=float)
		
		# Add effort from overlapping intervals
		if code in overlap_dict:
			for overlap_code in overlap_dict[code]:
				other_interval = interval_dict[overlap_code]
				overlap_start = max(left, other_interval.left) - left
				overlap_end = min(right, other_interval.right) - left
				
				if overlap_start < overlap_end:
					total_effort[overlap_start:overlap_end] += sampling_effort_dict[overlap_code]
		
		# Calculate weights as sampling effort divided by total effort
		overlap_weights[code] = sampling_effort / total_effort
	
	return overlap_weights