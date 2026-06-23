import numpy as np

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature        # index of the feature to split on
        self.threshold = threshold    # threshold value for split
        self.left = left              # left subtree
        self.right = right            # right subtree
        self.value = value            # value for leaf node (class)

    def is_leaf(self):
        return self.value is not None

class DecisionTreeClassifier():
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.max_features = max_features  # used in RandomForestClassifier

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return None  # metrics tracking skipped

    def __call__(self, X):
        # Reshape to (N, D) if X is higher-dimensional
        if X.ndim > 2:
            original_shape = X.shape[:-1]
            X_flat = X.reshape(-1, X.shape[-1])
            preds = np.array([self._predict(x, self.root) for x in X_flat])
            return preds.reshape(original_shape)
        else:
            return np.array([self._predict(x, self.root) for x in X])
        
    def _predict(self, x, node):
        while not node.is_leaf():
            if x[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if (depth >= self.max_depth or
            num_labels == 1 or
            num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        # Find best split
        best_feat, best_thresh = self._best_split(X, y, num_features)
        if best_feat is None:
            return DecisionTreeNode(value=self._most_common_label(y))

        # Split dataset
        left_idx = X[:, best_feat] < best_thresh
        right_idx = ~left_idx
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return DecisionTreeNode(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, num_features):
        best_gini = float('inf')
        best_feat, best_thresh = None, None

        feature_indices = np.arange(num_features)
        # Randomly select features if max_features is set (for RandomForest)
        if self.max_features is not None:
            feature_indices = np.random.choice(feature_indices, self.max_features, replace=False)

        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idx = X[:, feature_idx] < threshold
                right_idx = ~left_idx
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                gini = self._gini_index(y[left_idx], y[right_idx])
                if gini < best_gini:
                    best_gini = gini
                    best_feat = feature_idx
                    best_thresh = threshold

        return best_feat, best_thresh

    def _gini_index(self, left, right):
        def gini(y):
            classes = np.unique(y)
            return 1.0 - sum((np.sum(y == cls) / len(y)) ** 2 for cls in classes)

        total = len(left) + len(right)
        return (len(left) / total) * gini(left) + (len(right) / total) * gini(right)

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
