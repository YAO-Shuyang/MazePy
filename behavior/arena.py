import enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional
from shapely import boundary, linestrings
from shapely.geometry import Point, Polygon, LineString
import warnings
import networkx as nx

from mazepy.exceptions import DimensionError, ShapeError
from sympy import per

def is_concave(polygon: np.ndarray) -> np.ndarray:
    """
    Determine if the vertices of a polygon are concave or convex.
    
    Parameters
    ----------
    polygon : np.ndarray
        An array of shape (n, 2) representing the polygon vertices in order.
        
    Returns
    -------
    np.ndarray
        A boolean array of shape (n,), where True indicates the vertex is concave.
    """
    n = polygon.shape[0]
    is_concave = np.zeros(n, dtype=bool)
    
    # Loop through each vertex
    for i in range(n):
        # Get previous, current, and next vertices (cyclic indexing)
        p_prev = polygon[i - 1]
        p_curr = polygon[i]
        p_next = polygon[(i + 1) % n]
        
        # Compute cross product
        cross_product = (
            (p_next[0] - p_curr[0]) * (p_prev[1] - p_curr[1]) -
            (p_prev[0] - p_curr[0]) * (p_next[1] - p_curr[1])
        )
        
        # Determine if concave
        is_concave[i] = cross_product < 0  # Adjust based on orientation if needed
    
    return is_concave

class Arena2D:
    """
    Arena is a class that defines the arena of the environment.
    We assume that the arena you define are polygons (either convex or concave).
    
    Note that the boundary of the arena should be predetermined to initialize 
    the arena., while the inner walls can be later added one by one using the
    'add_wall' method.
    
    Parameters
    ----------
    vertexes: np.ndarray | list[tuple]
        The vertexes of the arena.
    inner_walls: np.ndarray | list[tuple], optional
        The inner walls of the arena.
        
    Attributes
    ----------
    vertexes: np.ndarray | list[tuple]
        The vertexes of the arena.
    inner_walls: np.ndarray | list[tuple], optional
        The inner walls of the arena.
        
    Methods
    -------
    add_wall(self, wall: tuple | list | np.ndarray) -> None
        Add an innate wall to the arena.
    remove_wall(self, wall_index: int) -> None
        Remove a wall from the arena.
    is_inarena(self, points: np.ndarray) -> np.ndarray
        Check if the points are in the arena.
    is_onwall(self, points: np.ndarray) -> np.ndarray
        Check if the points are on the inner walls.
    is_passable(self, points: np.ndarray) -> np.ndarray
        Check if the points cross the walls.

    Raises
    ------
    DimensionError, ShapeError
        If the vertexes are not 2D array with shape (n, 2).
    """
    
    def __init__(
        self, 
        vertexes: np.ndarray | list[tuple], 
        inner_walls: Optional[np.ndarray | list[tuple]] = None,
    ) -> None:
        self._vertexes = np.asarray(vertexes, np.float64)
        self.auto_walls = None
        
        if self._vertexes.ndim != 2:
            raise DimensionError(
                2,
                f"Vertexes should be 2D array, but got {self._vertexes.ndim}"
                f"D array instead."
            )
            
        if self._vertexes.shape[1] != 2:
            raise ShapeError(
                f"Vertexes should be 2D array with shape (n, 2), but got"
                f" {self._vertexes.shape} instead."
            )
            
        if self._vertexes.shape[0] < 3:
            raise ValueError(
                f"Vertexes should have at least 3 points, but got"
                f" {self._vertexes.shape[0]} points instead."
            )
            
        self._inner_walls = None
        
        # Initialize inner walls
        if inner_walls is not None:
            inner_walls = np.asarray(inner_walls, np.float64)
            
            # Check if inner walls are 2D array with shape (n, 4)
            if inner_walls.ndim != 2:
                raise DimensionError(
                    2,
                    f"Inner walls should be 2D array, but got"
                    f" {inner_walls.ndim}D array instead."
                )
                
            if inner_walls.shape[1] != 4:
                raise ShapeError(
                    f"Inner walls should be 2D array with shape (n, 4), but got"
                    f" {inner_walls.shape} instead."
                )
                
            # Add walls one by one to check if they were within the arena
            for i in range(inner_walls.shape[0]):
                self.add_wall(inner_walls[i, :])
        
    @property
    def vertexes(self):
        return self._vertexes
    
    @property
    def inner_walls(self):
        return self._inner_walls
    
    def add_wall(self, wall: tuple[float] | list[float] | np.ndarray) -> None:
        """
        Add an innate wall to the arena.
        
        Parameters
        ----------
        wall: tuple | list | np.ndarray
            The wall to be added. It should be a 1D array with shape (4, ).
            Four values are required: (x1, y1, x2, y2).
            x1, y1: The coordinates of the first point.
            x2, y2: The coordinates of the second point.
            
        Raises
        ------
        DimensionError
            If the input wall information is not 1D array.
        ShapeError
            If the input wall information is not 1D array with shape (4, ).
        """
        wall = np.asarray(wall, np.float64)
        
        if wall.ndim != 1:
            raise DimensionError(
                1,
                f"Wall should be 1D array, but got {wall.ndim}"
                f"D array instead."
            )
        
        if wall.shape[0] != 4:
            raise ShapeError(
                f"Wall should be 1D array with shape (4, ), but got"
                f" {wall.shape} instead."
            )
        
        points = np.reshape(wall, (2, 2))
        # Check if all ends of the wall are within the arena
        if not self.is_inarena(points).all():
            raise ValueError(
                f"All ends of the wall should be within the arena."
                f" Got {points} instead."
            )
                        
        if self.inner_walls is None:
            self._inner_walls = wall[np.newaxis, :]
        else:
            self._inner_walls = np.vstack([self._inner_walls, wall])            

    def enforce_partitioning(
        self, 
        min_pass_width: float = 0, 
        tol: float = 1e-6
    ) -> None:
        """
        In some cases, while two walls are not installed together, they are
        too close for the robot or animals to pass through. This situation
        is equivalent to the situation that there's a wall to be added between
        the two walls. 
        
        If this is the case, we need to add a new wall to enforce partitioning,
        elsewise, the computed shortest path will not match the real shortest
        path. This function is to handle this situation.
        
        Parameters
        ----------
        min_pass_width: float, optional
            The minimum passable width. If either endpoint of a wall is 
            closer than this value to any other wall or the boundary, 
            additional walls will be inserted to enforce partitioning.
        tol : float, optional
            Tolerance for considering a point as being on a wall, by default 
            1e-6.
        """
        min_pass_width = max(min_pass_width, 0)
        tol = max(tol, 0)
        
        perpendicular_walls = []
        if min_pass_width > 0:
            boundary: list[LineString] = [
                LineString([
                    (self._vertexes[i, 0], self._vertexes[i, 1]),
                    (self._vertexes[i+1, 0], self._vertexes[i+1, 1])
                ]) for i in range(self._vertexes.shape[0]-1)
            ] + [
                LineString([
                    (self._vertexes[-1, 0], self._vertexes[-1, 1]),
                    (self._vertexes[0, 0], self._vertexes[0, 1])
                ])
            ]
            
            if self.inner_walls is not None:
                wall_segments: list[LineString] = [
                    LineString([
                        (self.inner_walls[i, 0], self.inner_walls[i, 1]), 
                        (self.inner_walls[i, 2], self.inner_walls[i, 3])
                    ]) for i in range(self.inner_walls.shape[0])
                ]
            else:
                wall_segments: list[LineString] = []
            
            lines: list[LineString] =  wall_segments + boundary
            
            for i in range(self.inner_walls.shape[0]):
                for j, against_line in enumerate(lines):
                    for pt in [
                        Point(self.inner_walls[i, 0], self.inner_walls[i, 1]),
                        Point(self.inner_walls[i, 2], self.inner_walls[i, 3])
                    ]:
                        if i == j:
                            continue
                        
                        dist = against_line.distance(pt)
                        if dist < min_pass_width and dist > tol:
                            nearest_pt = against_line.interpolate(
                                against_line.project(pt)
                            )
                            perpendicular_walls.append(np.array([
                                pt.x, pt.y,
                                nearest_pt.x, nearest_pt.y
                            ]))
        
            for i in range(self.inner_walls.shape[0]):
                for j in range(len(perpendicular_walls)):
                    line = LineString([
                        (perpendicular_walls[j][0], perpendicular_walls[j][1]),
                        (perpendicular_walls[j][2], perpendicular_walls[j][3])
                    ])
                    
                    if wall_segments[i].contains(line):
                        perpendicular_walls.pop(j)
                    if line.contains(wall_segments[i]):
                        self._inner_walls[i, :] = perpendicular_walls[j]
                        perpendicular_walls.pop(j)
                    
            if len(perpendicular_walls) > 0:
                self.auto_walls = np.vstack(perpendicular_walls)
            
    def remove_wall(self, wall_index: int) -> None:
        """
        Remove a wall from the arena.
        
        Parameters
        ----------
        wall_index: int
            The index of the wall to be removed.
        """        
        if self.inner_walls is None:
            warnings.warn(
                "There is no inner wall to be removed."
            )
            return
        
        if self.inner_walls.shape[0] <= wall_index:
            warnings.warn(
                f"The wall index {wall_index} is out of the range of inner walls"
                f" {self.inner_walls.shape[0]}."
            )
            return
            
        self.inner_walls = np.delete(self.inner_walls, wall_index, axis=0)
        
    def is_inarena(self, points: np.ndarray) -> np.ndarray:
        """
        Check if given points are within the polygon defined by a set of vertices.

        arameters
        ----------
        points : np.ndarray
            An array of shape (N, 2) representing the coordinates of the points.
            
        Notes
        -----
        If you input only 1 point, please reshape them into (1, 2)
        e.g., for an array a = [x, y], you can use a[np.newaxis, :] to reshape 
        them.

        Returns
        -------
        np.ndarray
            A boolean array of shape (N,), where each element indicates whether
            the corresponding point is within the polygon.
            The result of the check.
            
        Raises
        ------
        DimensionError, ShapeError
            If the input points are not 2D array or 2D array with shape (n, 2).
        """
        if points.ndim != 2:
            raise DimensionError(
                f"Points should be 2D array, but got {points.ndim}"
                f"D array instead."
            )
        
        if points.shape[1] != 2:
            raise ShapeError(
                f"Points should be 2D array with shape (n, 2), but got"
                f" {points.shape} instead."
            )
        
        # Create a Shapely Polygon object
        poly = Polygon(self._vertexes)
        
        # Check for each point if it lies inside or on the boundary
        return np.array([poly.intersects(Point(p)) for p in points])
    
    def is_onwall(
        self, 
        points: np.ndarray | list[tuple],
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        Check if a set of points are on the inner walls of the arena.

        Parameters
        ----------
        points : np.ndarray | list[tuple]
            A 2D array of shape (M, 2), where each row represents the 
            coordinates of a point [x, y]. If the input is a list of tuples,
            it will be converted to a numpy array via numpy.asarray(). 
            Please make sure that all tuples within the list are of length 2.
        tol : float, optional
            Tolerance for considering a point as being on a wall, by default 
            1e-6.

        Returns
        -------
        np.ndarray
            A 1D boolean array of length M, where True indicates that the 
            corresponding point lies on one of the inner walls, and False 
            otherwise.
        """
        if self.inner_walls is None:
            warnings.warn("There is no inner wall to check.")
            return np.zeros(points.shape[0], dtype=bool)
        
        points = np.asarray(points, np.float64)
        if points.ndim != 2:
            raise DimensionError(
                f"Points should be 2D array, but got {points.ndim}"
                f"D array instead."
            )
        
        if points.shape[1] != 2:
            raise ShapeError(
                f"Points should be 2D array with shape (n, 2), but got"
                f" {points.shape} instead."
            )
        
        results = np.zeros(len(points), dtype=bool)
        
        # Iterate through walls and check if any point is on them
        wall_segments: list[LineString] = [
            LineString([
                (self.inner_walls[i, 0], self.inner_walls[i, 1]), 
                (self.inner_walls[i, 2], self.inner_walls[i, 3])
            ]) for i in range(self.inner_walls.shape[0])
        ]
        
        for i, point in enumerate(points):
            pt = Point(point)
            # Check if the point is within `tol` distance to any wall
            if any(
                wall_segment.distance(pt) <= tol for wall_segment in wall_segments
            ):
                results[i] = True
        
        return results
    
    def is_passable(self, points: np.ndarray | list[tuple]) -> np.ndarray:
        """
        Check if a set of points are passable in the arena.

        Parameters
        ----------
        points : np.ndarray | list[tuple]
            A 2D array of shape (M, 4), where each row represents the 
            coordinates of a point [x1, y1, x2, y2]. If the input is a list of 
            tuples, it will be converted to a numpy array via numpy.asarray(). 
            Please make sure that all tuples within the list are of length 4.

        Returns
        -------
        np.ndarray
            A 1D boolean array of length M, where True indicates that the 
            corresponding point is passable, and False otherwise.
        
        Raises
        ------
        DimensionError, ShapeError
            If the input points are not 2D array or 2D array with shape (n, 4).
        """
        
        points = np.asarray(points, np.float64)
        if points.ndim != 2:
            raise DimensionError(
                f"Points should be 2D array, but got {points.ndim}"
                f"D array instead."
            )
        
        if points.shape[1] != 4:
            raise ShapeError(
                f"Points should be 2D array with shape (n, 4), but got"
                f" {points.shape} instead."
            )

        if self.inner_walls is not None: 
            wall_segments: list[LineString] = [
                LineString([
                    (self.inner_walls[i, 0], self.inner_walls[i, 1]), 
                    (self.inner_walls[i, 2], self.inner_walls[i, 3])
                ]) for i in range(self.inner_walls.shape[0])
            ]
        else:
            wall_segments: list[LineString] = []
        
        boundary = [
            LineString([
                (self._vertexes[i, 0], self._vertexes[i, 1]), 
                (self._vertexes[i+1, 0], self._vertexes[i+1, 1])
            ]) for i in range(self._vertexes.shape[0]-1)
        ] + [
            LineString([
                (self._vertexes[-1, 0], self._vertexes[-1, 1]), 
                (self._vertexes[0, 0], self._vertexes[0, 1])
            ])
        ]
        
        return np.array([
            not any(
                wall_segments[i].intersects(
                    LineString([
                        (points[j, 0], points[j, 1]), 
                        (points[j, 2], points[j, 3])
                    ])
                ) for i in range(len(wall_segments))
            ) for j in range(points.shape[0])
        ])
        
    def visualize(self, ax: Optional[Axes] = None) -> Axes:
        """
        Visualize the arena.
        
        Parameters
        ----------
        ax : Optional[Axes], optional
            The axis to plot on. If not provided, a new figure and axis will 
            be created.
        
        Returns
        -------
        Axes
            The axis with the arena plotted on it.
        """
        if ax is None:
            fig = plt.figure(figsize = (6, 6))
            ax = plt.axes()
        
        # Plot the arena boundary
        for i in range(self._vertexes.shape[0]-1):
            ax.plot(
                [self._vertexes[i, 0], self._vertexes[i+1, 0]], 
                [self._vertexes[i, 1], self._vertexes[i+1, 1]], 
                'k-'
            )
            
        ax.plot(
            [self._vertexes[-1, 0], self._vertexes[0, 0]], 
            [self._vertexes[-1, 1], self._vertexes[0, 1]], 
            'k-'
        )
        
        # Plot the inner walls
        if self._inner_walls is not None:
            for i in range(self._inner_walls.shape[0]):
                ax.plot(
                    [self._inner_walls[i, 0], self._inner_walls[i, 2]], 
                    [self._inner_walls[i, 1], self._inner_walls[i, 3]], 
                    'k-'
                )
                
        if self.auto_walls is not None:
            for i in range(self.auto_walls.shape[0]):
                ax.plot(
                    [self.auto_walls[i, 0], self.auto_walls[i, 2]], 
                    [self.auto_walls[i, 1], self.auto_walls[i, 3]], 
                    'k:'
                )
        
        ax.set_aspect('equal')
        return ax
    
    def _wall_ends_graph(self):
        """
        To compute the shortest path within the maze, we need to construct a
        graph where the nodes are the wall ends and the edges are the distances
        between the wall ends. These edges should not cross the inner walls.
        """
        connectome = np.zeros((self.in))
        