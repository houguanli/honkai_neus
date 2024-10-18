"""
Fast Robust ICP
"""
from __future__ import annotations
import numpy
__all__ = ['PY_FRICPd']
class PY_FRICPd:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def run_icp(self, method: int = 3) -> numpy.ndarray[numpy.float64]:
        """
        Run ICP
        """
    def set_init_matrix(self, init_matrix: numpy.ndarray[numpy.float64]) -> None:
        """
        Set initial transformation matrix
        """
    def set_points(self, source_point: numpy.ndarray[numpy.float64], target_point: numpy.ndarray[numpy.float64]) -> None:
        """
        Set source and target points
        """
    def set_points_from_file(self, file_source: str, file_target: str) -> None:
        """
        Set source and target points from file
        """
    def set_source_from_file(self, file_source: str) -> None:
        """
        Set source points from file
        """
    def set_source_points(self, source_point: numpy.ndarray[numpy.float64]) -> None:
        """
        Set source points.
        """
    def set_target_from_file(self, file_target: str) -> None:
        """
        Set target points from file
        """
    def set_target_points(self, target_point: numpy.ndarray[numpy.float64]) -> None:
        """
        Set target points
        """
