from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: int = None
    mse: float = None
    left: Node = None
    right: Node = None


@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse criterion for a given set of target values."""
        mid = np.mean(y)
        return np.mean((y - mid) ** 2)

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weighted mse criterion for a two given sets of target values"""
        left_mse = self._mse(y_left)
        right_mse = self._mse(y_right)
        return (left_mse * y_left.size + right_mse * y_right.size) / (y_left.size + y_right.size)

    def _split(self, X: np.ndarray, y: np.ndarray, feature: int) -> tuple[float, float]:
        if y.size < 2:
            return None, None

        my_col = X[:, feature]
        iterate = np.unique(my_col)

        best = self._mse(y)
        best_threshold = None

        for i in iterate:
            mask = (my_col <= i)
            left_side = y[mask]
            right_side = y[~mask]
            if left_side.size == 0 or right_side.size == 0:
                continue
            q = self._weighted_mse(left_side, right_side)
            if q < best:
                best = q
                best_threshold = i
        return best_threshold, best

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        best_idx = None
        best_thr = None
        best_mse = None

        for i in range(X[0].size):
            current_threshold, current_mse = self._split(X, y, i)

            if best_thr is None:
                best_thr = current_threshold
                best_idx = i
                best_mse = current_mse
            else:
                if current_mse < best_mse:
                    best_thr = current_threshold
                    best_idx = i
                    best_mse = current_mse
        return best_idx, best_thr

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        feature, threshold = self._best_split(X, y)

        if depth >= self.max_depth:
            return Node(feature=feature, threshold=threshold, n_samples=y.size,
                        value=round(np.mean(y)), mse=self._mse(y), left=None, right=None)

        if feature is not None:
            current_Node = Node(feature=feature, threshold=threshold, n_samples=y.size,
                                value=round(np.mean(y)), mse=self._mse(y))
            depth += 1
            left_mask = X[:, feature] <= threshold

            left_main = X[left_mask]
            left_target = y[left_mask]
            right_main = X[~left_mask]
            right_target = y[~left_mask]
            current_Node.left = self._split_node(left_main, left_target, depth)
            current_Node.right = self._split_node(right_main, right_target, depth)
        else:
            current_Node = Node(n_samples=y.size, value=round(np.mean(y)), mse=self._mse(y))

        return current_Node

    def as_json(self) -> str:
        """Return the decision tree as a JSON string."""
        return self._as_json(self.tree_)

    def _as_json(self, node: Node) -> str:
        """Return the decision tree as a JSON string. Execute recursively."""
        if node is None:
            return None
        if node.left is None and node.right is None:
            return f'{{"value": {node.value}, "n_samples": {node.n_samples}, "mse": {round(node.mse, 2)}}}'
        else:
            left_json = self._as_json(node.left)
            right_json = self._as_json(node.right)
            construct = (f'{{"feature": {node.feature}, "threshold": {node.threshold}, '
                         f'"n_samples": {node.n_samples}, "mse": {round(node.mse, 2)}, '
                         f'"left": {left_json}, '
                         f'"right": {right_json}}}')

        return construct

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        # YOUR CODE HERE
        results = np.empty(X.shape[0])
        for i in range(results.size):
            results[i] = self._predict_one_sample(X[i])
        return results

    def _predict_one_sample(self, features: np.ndarray) -> int:
        """Predict the target value of a single sample."""
        root = self.tree_
        cur_node = root
        while cur_node.left is not None and cur_node.right is not None:

            if cur_node.left is None and cur_node.right is not None:
                cur_node = cur_node.right
            if cur_node.left is not None and cur_node.right is None:
                cur_node = cur_node.left

            if features[cur_node.feature] <= cur_node.threshold:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right

        return cur_node.value
