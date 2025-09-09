---
layout: default
title: "Agentic Refactoring and Package Development for Task-Pipeline-Kernel to PocketFlow"
---

# Agentic Refactoring: Task-Pipeline-Kernel to PocketFlow Architecture

> If you are an AI agent helping develop or refactor the task-pipeline-kernel architecture to PocketFlow primitives, read this guide **VERY, VERY** carefully! This refactoring maintains scikit-learn compatibility while leveraging PocketFlow's flexibility and modularity.
{: .warning }

## Core Architecture Mapping

### Foundational Components (New Additions)

The refactoring introduces three foundational components that provide the core functionality:

| Foundational Component | Purpose | Key Features |
|:----------------------|:--------|:-------------|
| **EnhancedSharedStore** | Data management and configuration | YAML configs, dataset storage, metadata tracking, file paths |
| **BaseDataNode** | Base class for dataset operations | Dataset input/output handling, shared store integration |
| **SklearnCompatibleNode** | ML compatibility foundation | scikit-learn interface (fit/transform), PyTorch interface (train/predict), type safety, logging |

### Architecture Mapping (Building on Foundation)

| Core-Tools Concept | Maps to Foundation + PocketFlow | Refactoring Strategy |
|:-------------------|:--------------------------------|:---------------------|
| [`BaseTask`](../../../../../../../c:/Users/ssainis/OneDrive - Intel Corporation/Desktop/python_scripts/applications.manufacturing.intel.quality.tdqr.core-tools/core_tools/core/BasicTaskObjects.py) | **SklearnCompatibleNode** → **Flow** | Enhanced BaseTask inherits from foundation, orchestrates pipeline sub-flows |
| [`BasePipeline`](../../../../../../../c:/Users/ssainis/OneDrive - Intel Corporation/Desktop/python_scripts/applications.manufacturing.intel.quality.tdqr.core-tools/core_tools/core/BasicPipelineObjects.py) | **SklearnCompatibleNode** → **Flow** | Enhanced BasePipeline inherits from foundation, converts to specialized Flows |
| [`BaseKernel`](../../../../../../../c:/Users/ssainis/OneDrive - Intel Corporation/Desktop/python_scripts/applications.manufacturing.intel.quality.tdqr.core-tools/core_tools/core/BasicKernelObjects.py) | **SklearnCompatibleNode** → **Node** | Enhanced BaseKernel inherits from foundation, becomes PocketFlow Node |
| Pipeline abstract methods | **SklearnCompatibleNode** → **Node** | Each abstract method becomes specialized Node inheriting from foundation |
| YAML configuration | **EnhancedSharedStore** | Configuration management through enhanced shared store |
| Dataset management | **EnhancedSharedStore** + **BaseDataNode** | Data flow via shared store, dataset ops via BaseDataNode |

## Refactoring Steps

### Step 1: Repository Structure Enhancement (Maintaining Backward Compatibility)

**IMPORTANT**: Maintain the existing core-tools structure and add PocketFlow capabilities as optional enhancements. This ensures existing applications continue to work without modification.

```
core_tools/                                # Keep existing structure
├── __init__.py                           
├── core/                                 # Existing core classes
│   ├── __init__.py
│   ├── BasicAuthObjects.py              # Keep existing
│   ├── BasicComponentAssemblerObjects.py # Keep existing
│   ├── BasicConnectorObjects.py         # Keep existing
│   ├── BasicKernelObjects.py            # ENHANCE with PocketFlow
│   ├── BasicMeasurementObjects.py       # Keep existing
│   ├── BasicMixinObjects.py             # Keep existing
│   ├── BasicObjectClasses.py            # Keep existing
│   ├── BasicPipelineObjects.py          # ENHANCE with PocketFlow
│   ├── BasicPOTObjects.py               # Keep existing
│   ├── BasicTaskObjects.py              # ENHANCE with PocketFlow
│   └── pocketflow_enhanced/             # NEW: PocketFlow enhancements
│       ├── __init__.py
│       ├── base_kernels.py              # PocketFlow kernel classes
│       ├── base_nodes.py                # PocketFlow base classes
│       ├── base_flows.py                # PocketFlow flow classes
│       ├── enhanced_pipelines.py        # PocketFlow-enhanced pipelines
│       ├── enhanced_tasks.py            # PocketFlow-enhanced tasks
│       └── shared_store.py              # Enhanced shared store
├── data/                                 # Keep existing
├── data_extractors/                      # Keep existing
├── datasets/                             # Keep existing
├── generators/                           # Keep existing
├── report_adapters/                      # Keep existing
├── reports/                              # Keep existing
├── tests/                                # Keep existing
└── utils/                                # Keep existing + add new
    ├── __init__.py
    ├── conversion_helpers.py             # Keep existing
    ├── decorators.py                     # Keep existing
    ├── helpers.py                        # Keep existing
    └── pocketflow_utils.py               # NEW: PocketFlow utilities
```

**Migration Strategy**: 
1. **Existing Applications**: Continue to work without any changes
2. **New Applications**: Can opt-in to PocketFlow enhancements
3. **Gradual Migration**: Existing applications can gradually adopt PocketFlow features

### Step 2: Core Abstractions Enhancement (Non-Breaking)

#### 2.1 Foundational Components

**Enhanced Shared Store (New Foundation)**:

```python
# core_tools/core/pocketflow_enhanced/shared_store.py
from typing import Dict, Any, Optional, Union
import pandas as pd
from pathlib import Path

class EnhancedSharedStore:
    """Enhanced shared store for task-pipeline-kernel architecture."""
    
    def __init__(self):
        self.data = {
            "config": {},                    # YAML configurations
            "datasets": {},                  # All datasets by name
            "models": {},                    # Trained models
            "pipelines": {},                 # Pipeline artifacts (encoders, etc.)
            "metadata": {                    # Execution metadata
                "task_id": None,
                "experiment_id": None,
                "pipeline_status": {},
                "execution_history": []
            },
            "paths": {                       # File system paths
                "task_dir": None,
                "experiment_dir": None,
                "output_dir": None
            }
        }
    
    def get_dataset(self, name: str) -> pd.DataFrame:
        """Retrieve dataset by name."""
        return self.data["datasets"].get(name, pd.DataFrame())
    
    def set_dataset(self, name: str, dataset: pd.DataFrame):
        """Store dataset with name."""
        self.data["datasets"][name] = dataset.copy()
    
    def get_config(self, section: str) -> Dict[str, Any]:
        """Retrieve configuration section."""
        return self.data["config"].get(section, {})
    
    def load_config(self, config_path: Union[str, Path]):
        """Load YAML configuration into shared store."""
        import yaml
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.data["config"].update(config)
```

**Base Data Node (Foundation for all data operations)**:

```python
# core_tools/core/pocketflow_enhanced/base_nodes.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from pocketflow import Node
import pandas as pd

class BaseDataNode(Node):
    """Base class for nodes that work with datasets."""
    
    def __init__(self, dataset_input: str = "input_dataset", 
                 dataset_output: str = "output_dataset", **kwargs):
        super().__init__(**kwargs)
        self.dataset_input = dataset_input
        self.dataset_output = dataset_output
    
    def prep(self, shared):
        """Prepare data from shared store."""
        store = shared if hasattr(shared, 'get_dataset') else EnhancedSharedStore()
        store.data.update(shared if isinstance(shared, dict) else {})
        return {
            "dataset": store.get_dataset(self.dataset_input),
            "config": store.get_config(self.__class__.__name__.lower()),
            "store": store
        }
    
    def post(self, shared, prep_res, exec_res):
        """Store results back to shared store."""
        store = prep_res["store"]
        if isinstance(exec_res, pd.DataFrame):
            store.set_dataset(self.dataset_output, exec_res)
        elif isinstance(exec_res, dict) and "dataset" in exec_res:
            store.set_dataset(self.dataset_output, exec_res["dataset"])
            # Store additional artifacts
            for key, value in exec_res.items():
                if key != "dataset":
                    store.data[key] = value
        
        # Update shared reference
        shared.update(store.data)
        return "default"
```

**Scikit-learn Compatible Node (Foundation for ML compatibility)**:

```python
# core_tools/core/pocketflow_enhanced/base_nodes.py (continued)
from typing import Any, Optional, Dict, Union, TypeVar, Generic, Literal
import logging
from sklearn.base import BaseEstimator, TransformerMixin

# Type variables for datasets
T = TypeVar('T')  # Working dataset type
U = TypeVar('U')  # Ancilliary dataset type
V = TypeVar('V')  # Result type

class SklearnCompatibleNode(BaseDataNode, BaseEstimator, TransformerMixin, Generic[T, U, V]):
    """
    Node that maintains scikit-learn compatibility with mode-aware execution.
    
    This serves as the foundation for all ML-compatible kernels and nodes.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, **kwargs):
        super().__init__(**kwargs)
        
        # Set up logger
        if logger is None:
            self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        else:
            self.logger = logger
        
        # Scikit-learn compatibility attributes
        self.is_fitted_ = False
        self.feature_names_in_ = None
        self.n_features_in_ = None
        
        # Initialize sklearn params
        self.set_params(**kwargs)
        
        self.logger.debug(f"Initialized {self.__class__.__name__} with params: {kwargs}")
    
    # === scikit-learn interface ===
    def fit(self, X: T, y: Optional[U] = None) -> 'SklearnCompatibleNode[T, U, V]':
        """Scikit-learn compatible fit method."""
        self.logger.info(f"Fitting {self.__class__.__name__}")
        self.logger.debug(f"Fit input X shape: {getattr(X, 'shape', None)}")
        
        # Store scikit-learn metadata
        self.feature_names_in_ = X.columns.tolist() if hasattr(X, 'columns') else None
        self.n_features_in_ = X.shape[1] if hasattr(X, 'shape') else None
        
        # Create shared store for fit operation
        shared: Dict[str, Any] = {
            "working_dataset": X, 
            "ancilliary_dataset": y,
            "mode": "fit"
        }
        
        # Execute using PocketFlow
        self.logger.debug("Starting PocketFlow node execution in fit mode")
        self.run(shared)
        self.logger.debug("Completed PocketFlow node execution in fit mode")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: T) -> V:
        """Scikit-learn compatible transform method."""
        if not self.is_fitted_:
            raise ValueError("This node has not been fitted yet.")
            
        self.logger.info(f"Transforming with {self.__class__.__name__}")
        self.logger.debug(f"Transform input X shape: {getattr(X, 'shape', None)}")
        
        # Create shared store for transform operation
        shared: Dict[str, Any] = {
            "working_dataset": X, 
            "ancilliary_dataset": getattr(self, "_fitted_data", None),
            "mode": "transform"
        }
        
        # Execute using PocketFlow
        self.logger.debug("Starting PocketFlow node execution in transform mode")
        self.run(shared)
        self.logger.debug("Completed PocketFlow node execution in transform mode")
        
        result = shared["working_dataset"]
        self.logger.debug(f"Transform output shape: {getattr(result, 'shape', None)}")
        return result
    
    def fit_transform(self, X: T, y: Optional[U] = None) -> V:
        """Scikit-learn compatible fit_transform method."""
        return self.fit(X, y).transform(X)
    
    # === PyTorch-style interface ===
    def train(self, X: T, y: Optional[U] = None) -> 'SklearnCompatibleNode[T, U, V]':
        """PyTorch-style alias for fit() method."""
        self.logger.info(f"Training {self.__class__.__name__} (alias for fit)")
        return self.fit(X, y)
    
    def predict(self, X: T) -> V:
        """PyTorch-style alias for transform() method."""
        self.logger.info(f"Predicting with {self.__class__.__name__} (alias for transform)")
        return self.transform(X)
    
    # === PocketFlow Node interface (mode-aware) ===
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Mode-aware preparation step."""
        mode = shared.get("mode", "transform")
        self.logger.debug(f"Preparing in {mode} mode")
        
        if mode == "fit":
            return self._prep_fit(shared)
        else:
            return self._prep_transform(shared)
    
    def _prep_fit(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Preparation for fit mode (training). Override in subclasses."""
        return {
            "X": shared.get("working_dataset"),
            "y": shared.get("ancilliary_dataset")
        }
    
    def _prep_transform(self, shared: Dict[str, Any]) -> Any:
        """Preparation for transform mode (inference). Override in subclasses."""
        return shared.get("working_dataset")
    
    def exec(self, prep_res: Any) -> Any:
        """Mode-aware execution."""
        # Determine current mode from prep_res structure
        if isinstance(prep_res, dict) and "X" in prep_res and "y" in prep_res:
            mode = "fit"
        else:
            mode = "transform"
            
        self.logger.debug(f"Executing in {mode} mode")
        
        if mode == "fit":
            return self._exec_fit(prep_res)
        else:
            return self._exec_transform(prep_res)
    
    def _exec_fit(self, prep_res: Dict[str, Any]) -> V:
        """Execution for fit mode (training). Override in subclasses."""
        # Default implementation: store the training data
        self._fitted_data = prep_res
        self.logger.debug("Default fit implementation (storing training data)")
        return prep_res["X"]
    
    def _exec_transform(self, X: T) -> V:
        """Execution for transform mode (inference). Override in subclasses."""
        # Default implementation: identity transformation
        self.logger.debug("Default transform implementation (identity function)")
        return X
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Any) -> Literal["default"]:
        """Mode-aware post-processing."""
        mode = shared.get("mode", "transform")
        self.logger.debug(f"Post-processing in {mode} mode")
        
        if mode == "fit":
            return self._post_fit(shared, prep_res, exec_res)
        else:
            return self._post_transform(shared, prep_res, exec_res)
    
    def _post_fit(self, shared: Dict[str, Any], prep_res: Any, exec_res: Any) -> Literal["default"]:
        """Post-processing for fit mode (training). Override in subclasses."""
        shared["working_dataset"] = exec_res
        self.logger.debug("Updated working_dataset in shared store after fit")
        return "default"
    
    def _post_transform(self, shared: Dict[str, Any], prep_res: Any, exec_res: Any) -> Literal["default"]:
        """Post-processing for transform mode (inference). Override in subclasses."""
        shared["working_dataset"] = exec_res
        self.logger.debug("Updated working_dataset in shared store after transform")
        return "default"
```

#### 2.2 Enhanced Existing Classes (Building on Foundation)

Now that we have the foundational components, we can enhance the existing classes by inheriting from them:

**Enhanced Kernel Objects**:

```python
# core_tools/core/BasicKernelObjects.py (ENHANCED)
from abc import ABC, abstractmethod

# Keep all existing kernel classes exactly as they are for backward compatibility
class BaseKernel(ABC):
    """Original BaseKernel - UNCHANGED for backward compatibility"""
    # ... keep all existing methods exactly as they are
    pass

# Add NEW PocketFlow-enhanced version that inherits from SklearnCompatibleNode
class BaseKernelWithPocketFlow(SklearnCompatibleNode[T, U, V]):
    """
    PocketFlow-enhanced kernel that inherits from SklearnCompatibleNode.
    
    This provides all the scikit-learn compatibility, PyTorch-style interface,
    and PocketFlow execution capabilities from the base class.
    
    Type Parameters:
        T: Type of working dataset
        U: Type of ancilliary dataset  
        V: Type of transformed/result data
    """
    
    def __init__(self, **kwargs):
        """Initialize the enhanced kernel."""
        super().__init__(**kwargs)
        self.logger.info(f"Initialized {self.__class__.__name__} as PocketFlow-enhanced kernel")
    
    # The fit, transform, train, predict methods are inherited from SklearnCompatibleNode
    # The prep, exec, post methods are inherited and can be overridden for custom behavior
    
    # Override these methods in concrete implementations:
    def _exec_fit(self, prep_res: Dict[str, Any]) -> V:
        """Override this for custom fit logic."""
        return super()._exec_fit(prep_res)
    
    def _exec_transform(self, X: T) -> V:
        """Override this for custom transform logic."""
        return super()._exec_transform(X)
```

**Enhanced Pipeline Objects**:

```python
# core_tools/core/BasicPipelineObjects.py (ENHANCED)
from typing import List, Optional
from pocketflow import Flow

# Keep all existing classes exactly as they are
class BasePipeline(ABC):
    """Original BasePipeline - UNCHANGED for backward compatibility"""
    # ... keep all existing methods exactly as they are
    pass

# Add NEW PocketFlow-enhanced version that combines Flow with sklearn compatibility
class BasePipelineWithPocketFlow(Flow, BaseEstimator, TransformerMixin, Generic[T, V]):
    """
    PocketFlow-enhanced pipeline that combines Flow with scikit-learn compatibility.
    
    This class orchestrates multiple BaseKernelWithPocketFlow instances.
    """
    
    def __init__(self, kernels: Optional[List[BaseKernelWithPocketFlow]] = None, 
                 logger: Optional[logging.Logger] = None, **kwargs):
        """Initialize with a sequence of kernels."""
        # Set up logger
        if logger is None:
            self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        else:
            self.logger = logger
            
        # Store kernels
        self.kernels = kernels or []
        
        # Connect kernels in sequence using PocketFlow
        for i in range(len(self.kernels) - 1):
            self.kernels[i] >> self.kernels[i+1]
        
        # Initialize flow with first kernel as start if available
        if self.kernels:
            self.logger.info(f"Initializing pipeline with {len(self.kernels)} kernels")
            Flow.__init__(self, start=self.kernels[0])
        else:
            self.logger.warning("Initializing empty pipeline with no kernels")
            Flow.__init__(self)
            
        # Store sklearn params
        self.set_params(**kwargs)
    
    # === scikit-learn interface ===
    def fit(self, X: T, y: Optional[Any] = None) -> 'BasePipelineWithPocketFlow[T, V]':
        """Fit the pipeline using PocketFlow's execution model."""
        self.logger.info(f"Fitting {self.__class__.__name__} with {len(self.kernels)} kernels")
        
        shared: Dict[str, Any] = {
            "working_dataset": X, 
            "ancilliary_dataset": y,
            "mode": "fit"
        }
        
        self.run(shared)
        return self
        
    def transform(self, X: T) -> V:
        """Transform using PocketFlow's execution model."""
        self.logger.info(f"Transforming with {self.__class__.__name__}")
        
        shared: Dict[str, Any] = {
            "working_dataset": X,
            "mode": "transform"
        }
        
        self.run(shared)
        return shared["working_dataset"]
    
    # === PyTorch-style interface ===  
    def train(self, X: T, y: Optional[Any] = None) -> 'BasePipelineWithPocketFlow[T, V]':
        """PyTorch-style alias for fit() method."""
        return self.fit(X, y)
    
    def predict(self, X: T) -> V:
        """PyTorch-style alias for transform() method."""
        return self.transform(X)
```

**Enhanced Task Objects**:

```python
# core_tools/core/BasicTaskObjects.py (ENHANCED)

# Keep all existing classes exactly as they are
class BaseTask(ABC):
    """Original BaseTask - UNCHANGED for backward compatibility"""
    # ... keep all existing methods exactly as they are
    pass

# Add NEW PocketFlow-enhanced version
class BaseTaskWithPocketFlow(BaseEstimator, Generic[T, V]):
    """
    PocketFlow-enhanced task that orchestrates multiple pipelines.
    
    This class manages the high-level workflow coordination.
    """
    
    def __init__(self, pipelines: Optional[List[BasePipelineWithPocketFlow]] = None, 
                 logger: Optional[logging.Logger] = None, 
                 config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the task with pipelines and configuration."""
        # Set up logger
        if logger is None:
            self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        else:
            self.logger = logger
            
        # Store pipelines and config
        self.pipelines = pipelines or []
        self.config = config or {}
        
        # Initialize parameters
        self.set_params(**kwargs)
        
        self.logger.info(f"Initialized {self.__class__.__name__} with {len(self.pipelines)} pipelines")
    
    # === scikit-learn interface ===
    def fit(self, X: T, y: Optional[Any] = None) -> 'BaseTaskWithPocketFlow[T, V]':
        """Fit all pipelines in sequence."""
        self.logger.info(f"Fitting {self.__class__.__name__}")
        
        current_X = X
        for i, pipeline in enumerate(self.pipelines):
            self.logger.info(f"Fitting pipeline {i+1}/{len(self.pipelines)}: {pipeline.__class__.__name__}")
            pipeline.fit(current_X, y)
            current_X = pipeline.transform(current_X)
        
        return self
    
    def transform(self, X: T) -> V:
        """Transform data through all pipelines in sequence."""
        self.logger.info(f"Transforming with {self.__class__.__name__}")
        
        current_X = X
        for i, pipeline in enumerate(self.pipelines):
            self.logger.info(f"Running pipeline {i+1}/{len(self.pipelines)}: {pipeline.__class__.__name__}")
            current_X = pipeline.transform(current_X)
        
        return current_X
    
    # === PyTorch-style interface ===
    def train(self, X: T, y: Optional[Any] = None) -> 'BaseTaskWithPocketFlow[T, V]':
        """PyTorch-style alias for fit() method."""
        return self.fit(X, y)
    
    def predict(self, X: T) -> V:
        """PyTorch-style alias for transform() method."""
        return self.transform(X)
    
    def add_pipeline(self, pipeline: BasePipelineWithPocketFlow) -> None:
        """Add a pipeline to the task."""
        self.pipelines.append(pipeline)
        self.logger.info(f"Added pipeline: {pipeline.__class__.__name__}")
```
```

#### 2.3 Usage Examples

**Example: Creating a Custom Kernel**:

```python
# Example: Enhanced Kernel Implementation
import pandas as pd
import numpy as np
from core_tools.core.pocketflow_enhanced.base_nodes import SklearnCompatibleNode

class StandardizationKernel(SklearnCompatibleNode[pd.DataFrame, None, pd.DataFrame]):
    """A kernel that standardizes data (zero mean, unit variance)."""
    
    def _exec_fit(self, prep_res: Dict[str, Any]) -> pd.DataFrame:
        """Learn mean and std from training data."""
        X = prep_res["X"]
        self.logger.info("Calculating mean and std for standardization")
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        
        # Return standardized training data
        return (X - self.mean_) / self.std_
    
    def _exec_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply standardization to new data."""
        self.logger.debug(f"Standardizing data with mean={self.mean_.iloc[0]:.4f} and std={self.std_.iloc[0]:.4f}")
        return (X - self.mean_) / self.std_

# Usage Examples:
# 1. Use as individual components
standardizer = StandardizationKernel()
X_train = pd.DataFrame(np.random.randn(100, 5))
X_test = pd.DataFrame(np.random.randn(20, 5))

# Fit and transform (scikit-learn style)
standardizer.fit(X_train)
X_train_scaled = standardizer.transform(X_train)
X_test_scaled = standardizer.transform(X_test)

# 2. PyTorch-style interface
standardizer.train(X_train)
predictions = standardizer.predict(X_test)

# 3. Use enhanced versions in task/pipeline/kernel hierarchy
preprocessing_pipeline = BasePipelineWithPocketFlow(kernels=[standardizer])
ml_task = BaseTaskWithPocketFlow(pipelines=[preprocessing_pipeline])

# 4. Integration with scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Can be used in scikit-learn pipelines
sklearn_pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('classifier', RandomForestClassifier())
])

# All sklearn functionality works
from sklearn.model_selection import cross_val_score, GridSearchCV
scores = cross_val_score(sklearn_pipeline, X_train, y_train, cv=5)
```
```

### Step 3: Pipeline-Specific Flows

*Note: This section remains unchanged from the original document as it represents concrete implementation details for specific use cases.*

```python
# core_tools/core/pocketflow_enhanced/base_flows.py
from pocketflow import Flow, Node
from typing import Any, Dict, Optional
import pandas as pd

class PocketFlowPipelineAdapter:
    """Adapter to bridge existing pipelines with PocketFlow execution."""
    
    def __init__(self, original_pipeline):
        self.original_pipeline = original_pipeline
        self.shared_store = EnhancedSharedStore()
        self._create_pocketflow_nodes()
    
    def _create_pocketflow_nodes(self):
        """Create PocketFlow nodes from original pipeline methods."""
        # Create nodes for each abstract method in the original pipeline
        self.nodes = []
        
        # Example for BaseDataWranglingPipeline
        if hasattr(self.original_pipeline, '_clean_data'):
            self.nodes.append(CleanDataNode(self.original_pipeline))
        if hasattr(self.original_pipeline, '_feature_engineering'):
            self.nodes.append(FeatureEngineeringNode(self.original_pipeline))
        if hasattr(self.original_pipeline, '_populate_ancilliary'):
            self.nodes.append(PopulateAncilliaryNode(self.original_pipeline))
        
        # Connect nodes in sequence
        for i in range(len(self.nodes) - 1):
            self.nodes[i] >> self.nodes[i+1]
        
        # Create flow
        if self.nodes:
            self.flow = Flow(start=self.nodes[0])
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit using PocketFlow execution."""
        self.shared_store.set_dataset("input_dataset", X)
        if y is not None:
            self.shared_store.set_dataset("target", y)
        
        self.shared_store.data["mode"] = "fit"
        self.flow.run(self.shared_store.data)
        return self.original_pipeline
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using PocketFlow execution."""
        self.shared_store.set_dataset("input_dataset", X)
        self.shared_store.data["mode"] = "transform"
        self.flow.run(self.shared_store.data)
        return self.shared_store.get_dataset("output_dataset")
    
    def run(self):
        """Execute the flow."""
        return self.flow.run(self.shared_store.data)

class PipelineMethodNode(Node):
    """Base node that wraps existing pipeline methods."""
    
    def __init__(self, original_pipeline, method_name: str, **kwargs):
        super().__init__(**kwargs)
        self.original_pipeline = original_pipeline
        self.method_name = method_name
    
    def prep(self, shared):
        """Extract data for the method."""
        return {
            "dataset": shared["datasets"].get("input_dataset", pd.DataFrame()),
            "config": shared.get("config", {}),
            "original_pipeline": self.original_pipeline
        }
    
    def exec(self, prep_res):
        """Execute the original pipeline method."""
        method = getattr(prep_res["original_pipeline"], self.method_name)
        return method(prep_res["dataset"])
    
    def post(self, shared, prep_res, exec_res):
        """Update shared store."""
        shared["datasets"]["output_dataset"] = exec_res
        return "default"
```

```python
# src/pocketflow_core_tools/core/base_nodes.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from pocketflow import Node
import pandas as pd

class BaseDataNode(Node):
    """Base class for nodes that work with datasets."""
    
    def __init__(self, dataset_input: str = "input_dataset", 
                 dataset_output: str = "output_dataset", **kwargs):
        super().__init__(**kwargs)
        self.dataset_input = dataset_input
        self.dataset_output = dataset_output
    
    def prep(self, shared):
        """Prepare data from shared store."""
        store = shared if hasattr(shared, 'get_dataset') else EnhancedSharedStore()
        store.data.update(shared if isinstance(shared, dict) else {})
        return {
            "dataset": store.get_dataset(self.dataset_input),
            "config": store.get_config(self.__class__.__name__.lower()),
            "store": store
        }
    
    def post(self, shared, prep_res, exec_res):
        """Store results back to shared store."""
        store = prep_res["store"]
        if isinstance(exec_res, pd.DataFrame):
            store.set_dataset(self.dataset_output, exec_res)
        elif isinstance(exec_res, dict) and "dataset" in exec_res:
            store.set_dataset(self.dataset_output, exec_res["dataset"])
            # Store additional artifacts
            for key, value in exec_res.items():
                if key != "dataset":
                    store.data[key] = value
        
        # Update shared reference
        shared.update(store.data)
        return "default"

class SklearnCompatibleNode(BaseDataNode):
    """Node that maintains scikit-learn compatibility."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_fitted_ = False
        self.feature_names_in_ = None
        self.n_features_in_ = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Scikit-learn compatible fit method."""
        # Store scikit-learn metadata
        self.feature_names_in_ = X.columns.tolist() if hasattr(X, 'columns') else None
        self.n_features_in_ = X.shape[1] if hasattr(X, 'shape') else None
        
        # Call internal fit logic
        self._fit_impl(X, y)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scikit-learn compatible transform method."""
        if not self.is_fitted_:
            raise ValueError("This node has not been fitted yet.")
        return self._transform_impl(X)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Scikit-learn compatible fit_transform method."""
        return self.fit(X, y).transform(X)
    
    @abstractmethod
    def _fit_impl(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Internal fit implementation."""
        pass
    
    @abstractmethod
    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        """Internal transform implementation."""
        pass
    
    def exec(self, prep_res):
        """PocketFlow exec method integrates with sklearn methods."""
        dataset = prep_res["dataset"]
        config = prep_res["config"]
        
        # Determine operation mode
        if config.get("mode", "transform") == "fit_transform":
            y = prep_res.get("target", None)
            return self.fit_transform(dataset, y)
        elif hasattr(self, 'is_fitted_') and self.is_fitted_:
            return self.transform(dataset)
        else:
            return self.fit_transform(dataset)
```

### Step 3: Pipeline-Specific Flows

#### 3.1 Data Wrangling Pipeline

```python
# src/pocketflow_core_tools/pipelines/data_wrangling/flows.py
from pocketflow import Flow
from .nodes import CleanDataNode, FeatureEngineeringNode, PopulateAncilliaryNode

class DataWranglingFlow(Flow):
    """PocketFlow implementation of BaseDataWranglingPipeline."""
    
    def __init__(self, config: dict):
        # Create nodes
        clean_node = CleanDataNode()
        feature_node = FeatureEngineeringNode()
        populate_node = PopulateAncilliaryNode()
        
        # Connect nodes
        clean_node >> feature_node >> populate_node
        
        # Initialize flow
        super().__init__(start=clean_node)
        self.config = config
    
    def prep(self, shared):
        """Flow-level preparation."""
        # Load configuration into shared store
        shared["config"]["data_wrangling"] = self.config
        return None
    
    def post(self, shared, prep_res, exec_res):
        """Flow-level post-processing."""
        shared["metadata"]["pipeline_status"]["data_wrangling"] = "completed"
        return "default"

# src/pocketflow_core_tools/pipelines/data_wrangling/nodes.py
from ...core.base_nodes import BaseDataNode

class CleanDataNode(BaseDataNode):
    """Node for data cleaning operations."""
    
    def exec(self, prep_res):
        dataset = prep_res["dataset"]
        config = prep_res["config"]
        
        # Implement cleaning logic
        cleaned_dataset = self._clean_data(dataset, config)
        return cleaned_dataset
    
    def _clean_data(self, dataset, config):
        """Implement specific cleaning logic."""
        # Remove duplicates, handle missing values, etc.
        cleaned = dataset.dropna()
        return cleaned

class FeatureEngineeringNode(BaseDataNode):
    """Node for feature engineering operations."""
    
    def __init__(self, **kwargs):
        super().__init__(dataset_input="cleaned_dataset", 
                        dataset_output="engineered_dataset", **kwargs)
    
    def exec(self, prep_res):
        dataset = prep_res["dataset"]
        config = prep_res["config"]
        
        # Implement feature engineering logic
        engineered_dataset = self._feature_engineering(dataset, config)
        return engineered_dataset
    
    def _feature_engineering(self, dataset, config):
        """Implement specific feature engineering logic."""
        # Create new features, transform existing ones, etc.
        return dataset  # Placeholder

class PopulateAncilliaryNode(BaseDataNode):
    """Node for populating ancilliary data."""
    
    def __init__(self, **kwargs):
        super().__init__(dataset_input="engineered_dataset", 
                        dataset_output="final_dataset", **kwargs)
    
    def exec(self, prep_res):
        dataset = prep_res["dataset"]
        config = prep_res["config"]
        store = prep_res["store"]
        
        # Populate ancilliary data
        useful_columns = config.get('useful_columns', dataset.columns.tolist())
        store.data["ancilliary"]["useful_columns"] = useful_columns
        
        return dataset
```

#### 3.2 Preprocessing Pipeline

```python
# src/pocketflow_core_tools/pipelines/preprocessing/flows.py
from pocketflow import Flow
from .nodes import FilterDataNode, SplitDataNode, EncodeDataNode, NormalizeDataNode

class PreProcessingFlow(Flow):
    """PocketFlow implementation of BasePreProcessingPipeline."""
    
    def __init__(self, config: dict):
        # Create nodes
        filter_node = FilterDataNode()
        split_node = SplitDataNode()
        encode_node = EncodeDataNode()
        normalize_node = NormalizeDataNode()
        
        # Connect nodes in sequence
        filter_node >> split_node >> encode_node >> normalize_node
        
        super().__init__(start=filter_node)
        self.config = config

# src/pocketflow_core_tools/pipelines/preprocessing/nodes.py
from ...core.base_nodes import SklearnCompatibleNode
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

class FilterDataNode(SklearnCompatibleNode):
    """Node for data filtering operations."""
    
    def _fit_impl(self, X, y=None):
        """Learn filtering criteria."""
        self.filtering_criteria_ = self._determine_filtering_criteria(X)
    
    def _transform_impl(self, X):
        """Apply filtering."""
        return self._apply_filtering(X, self.filtering_criteria_)
    
    def _determine_filtering_criteria(self, X):
        """Determine what rows/columns to filter."""
        # Implement filtering logic
        return {}
    
    def _apply_filtering(self, X, criteria):
        """Apply filtering criteria."""
        # Implement actual filtering
        return X

class EncodeDataNode(SklearnCompatibleNode):
    """Node for data encoding operations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoders_ = {}
    
    def _fit_impl(self, X, y=None):
        """Fit encoders for categorical variables."""
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            encoder = LabelEncoder()
            encoder.fit(X[col].astype(str))
            self.encoders_[col] = encoder
    
    def _transform_impl(self, X):
        """Apply encoding transformations."""
        X_encoded = X.copy()
        
        for col, encoder in self.encoders_.items():
            if col in X_encoded.columns:
                X_encoded[col] = encoder.transform(X_encoded[col].astype(str))
        
        return X_encoded
    
    def exec(self, prep_res):
        """Override exec to store encoder pipeline."""
        result_dataset = super().exec(prep_res)
        
        # Store encoding pipeline for later use
        prep_res["store"].data["pipelines"]["encoding_pipeline"] = self.encoders_
        
        return result_dataset

class NormalizeDataNode(SklearnCompatibleNode):
    """Node for data normalization operations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scaler_ = StandardScaler()
    
    def _fit_impl(self, X, y=None):
        """Fit normalization scaler."""
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        self.numeric_cols_ = numeric_cols
        
        if len(numeric_cols) > 0:
            self.scaler_.fit(X[numeric_cols])
    
    def _transform_impl(self, X):
        """Apply normalization."""
        X_normalized = X.copy()
        
        if len(self.numeric_cols_) > 0:
            X_normalized[self.numeric_cols_] = self.scaler_.transform(X[self.numeric_cols_])
        
        return X_normalized
    
    def exec(self, prep_res):
        """Override exec to store normalization pipeline."""
        result_dataset = super().exec(prep_res)
        
        # Store normalization pipeline
        prep_res["store"].data["pipelines"]["normalization_pipeline"] = self.scaler_
        
        return result_dataset
```

### Step 4: Task-Level Orchestration

#### 4.1 Experiment Flow (was BaseExperiment)

```python
# src/pocketflow_core_tools/tasks/experiment_flows.py
from pocketflow import Flow, BatchFlow
from typing import Dict, Any
from pathlib import Path
import uuid

class ExperimentFlow(Flow):
    """PocketFlow implementation of BaseExperiment."""
    
    def __init__(self, experiment_config_path: str):
        self.config_path = Path(experiment_config_path)
        self.experiment_id = str(uuid.uuid4())
        
        # Load configuration
        self.config = self._load_config()
        
        # Set up experiment directory
        self.experiment_dir = Path(self.config["experiment_base_dir"]) / self.config["experiment_name"]
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create pipeline flows based on configuration
        self.pipeline_flows = self._create_pipeline_flows()
        
        # Connect flows
        self._connect_pipeline_flows()
        
        # Initialize with first pipeline
        super().__init__(start=list(self.pipeline_flows.values())[0])
    
    def _load_config(self) -> Dict[str, Any]:
        """Load experiment configuration from YAML."""
        import yaml
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _create_pipeline_flows(self) -> Dict[str, Flow]:
        """Create pipeline flows based on configuration."""
        from ..pipelines.data_wrangling.flows import DataWranglingFlow
        from ..pipelines.preprocessing.flows import PreProcessingFlow
        
        flows = {}
        
        for pipeline_name, pipeline_config in self.config.get("pipeline_configs", {}).items():
            pipeline_type = pipeline_config.get("type", "")
            
            if pipeline_type == "data_wrangling":
                flows[pipeline_name] = DataWranglingFlow(pipeline_config)
            elif pipeline_type == "preprocessing":
                flows[pipeline_name] = PreProcessingFlow(pipeline_config)
            # Add other pipeline types as needed
        
        return flows
    
    def _connect_pipeline_flows(self):
        """Connect pipeline flows in sequence."""
        flow_list = list(self.pipeline_flows.values())
        for i in range(len(flow_list) - 1):
            flow_list[i] >> flow_list[i + 1]
    
    def prep(self, shared):
        """Experiment-level preparation."""
        # Initialize enhanced shared store
        if not hasattr(shared, 'data'):
            from ..core.shared_store import EnhancedSharedStore
            store = EnhancedSharedStore()
            shared.update(store.data)
        
        # Set experiment metadata
        shared["metadata"]["experiment_id"] = self.experiment_id
        shared["paths"]["experiment_dir"] = str(self.experiment_dir)
        
        # Load experiment configuration
        shared["config"]["experiment"] = self.config
        
        return None
    
    def post(self, shared, prep_res, exec_res):
        """Experiment-level post-processing."""
        # Save results
        self._save_results(shared)
        return "default"
    
    def _save_results(self, shared):
        """Save experiment results to disk."""
        import yaml
        
        results = {
            "experiment_id": shared["metadata"]["experiment_id"],
            "config": shared["config"],
            "pipeline_status": shared["metadata"]["pipeline_status"],
            "datasets": {name: df.shape for name, df in shared["datasets"].items()}
        }
        
        results_path = self.experiment_dir / "experiment_results.yaml"
        with open(results_path, 'w') as file:
            yaml.dump(results, file)

class BatchExperimentFlow(BatchFlow):
    """Run multiple experiments in batch mode."""
    
    def prep(self, shared):
        """Prepare multiple experiment configurations."""
        experiment_configs = shared.get("experiment_configs", [])
        return [{"experiment_config": config} for config in experiment_configs]
```

### Step 5: Scikit-learn Compatibility Layer

```python
# src/pocketflow_core_tools/utils/sklearn_compatibility.py
from typing import Optional, Union
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ..core.shared_store import EnhancedSharedStore

class PocketFlowSklearnWrapper(BaseEstimator, TransformerMixin):
    """Wrapper to make PocketFlow flows compatible with scikit-learn pipelines."""
    
    def __init__(self, flow_class, flow_config: dict):
        self.flow_class = flow_class
        self.flow_config = flow_config
        self.flow_ = None
        self.shared_store_ = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the PocketFlow pipeline."""
        # Initialize flow and shared store
        self.flow_ = self.flow_class(self.flow_config)
        self.shared_store_ = EnhancedSharedStore()
        
        # Set up shared store with input data
        self.shared_store_.set_dataset("input_dataset", X)
        if y is not None:
            self.shared_store_.set_dataset("target", y)
        
        # Configure for fit mode
        self.shared_store_.data["config"]["mode"] = "fit_transform"
        
        # Run flow
        self.flow_.run(self.shared_store_.data)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted PocketFlow pipeline."""
        if self.flow_ is None:
            raise ValueError("Pipeline has not been fitted yet.")
        
        # Set up shared store with new data
        self.shared_store_.set_dataset("input_dataset", X)
        self.shared_store_.data["config"]["mode"] = "transform"
        
        # Run flow
        self.flow_.run(self.shared_store_.data)
        
        # Return transformed dataset
        return self.shared_store_.get_dataset("output_dataset")
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

# Example usage in scikit-learn pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def create_sklearn_pipeline():
    """Create scikit-learn pipeline with PocketFlow components."""
    from ..pipelines.preprocessing.flows import PreProcessingFlow
    
    return Pipeline([
        ('preprocessing', PocketFlowSklearnWrapper(
            PreProcessingFlow, 
            {'normalization_method': 'standard', 'encoding_method': 'label'}
        )),
        ('classifier', RandomForestClassifier())
    ])
```

### Step 6: Configuration Management

```python
# src/pocketflow_core_tools/utils/config_loader.py
from typing import Dict, Any, Union
from pathlib import Path
import yaml

class ConfigurationManager:
    """Manages YAML configuration loading and validation."""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and validate YAML configuration."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Validate configuration structure
        ConfigurationManager._validate_config(config)
        return config
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]):
        """Validate configuration structure."""
        required_sections = ["experiment_config", "pipeline_configs"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    @staticmethod
    def create_template_config() -> Dict[str, Any]:
        """Create template configuration for new projects."""
        return {
            "experiment_config": {
                "experiment_name": "new_experiment",
                "experiment_base_dir": "./experiments",
                "experiment_description": "Description of the experiment"
            },
            "pipeline_configs": {
                "data_wrangling": {
                    "type": "data_wrangling",
                    "useful_columns": [],
                    "cleaning_params": {}
                },
                "preprocessing": {
                    "type": "preprocessing",
                    "normalization_method": "standard",
                    "encoding_method": "label"
                }
            }
        }
```

## Refactoring Best Practices

### 1. **Start Small and Iterate**
Begin by refactoring one pipeline at a time. Start with [`BaseDataWranglingPipeline`](../../../../../../../c:/Users/ssainis/OneDrive - Intel Corporation/Desktop/python_scripts/applications.manufacturing.intel.quality.tdqr.core-tools/core_tools/core/BasicPipelineObjects.py) as it has the simplest structure.

### 2. **Maintain Backward Compatibility**
All existing classes remain unchanged. New features are added as optional enhancements:

```python
# Example: Existing applications continue to work
from core_tools.core.BasicPipelineObjects import BaseDataWranglingPipeline

# This still works exactly as before
class MyExistingPipeline(BaseDataWranglingPipeline):
    def _clean_data(self, dataset):
        # Original implementation unchanged
        return cleaned_data
    
    def run(self):
        # Original method unchanged
        return super().run()

# NEW: Applications can opt-in to PocketFlow enhancements
class MyEnhancedPipeline(BasePipelineWithPocketFlow):
    def _clean_data(self, dataset):
        # Same implementation, now with PocketFlow benefits
        return cleaned_data
    
    # Now supports scikit-learn interface
    def fit(self, X, y=None):
        return super().fit(X, y)
    
    def transform(self, X):
        return super().transform(X)
```

### 3. **Preserve Scikit-learn Integration**
Ensure all refactored components maintain compatibility with scikit-learn pipelines through the wrapper classes.

### 4. **Test-Driven Refactoring**
Write tests for the new PocketFlow components before refactoring to ensure behavioral consistency:

```python
# tests/unit/test_data_wrangling_flow.py
import unittest
import pandas as pd
from src.pocketflow_core_tools.pipelines.data_wrangling.flows import DataWranglingFlow
from src.pocketflow_core_tools.core.shared_store import EnhancedSharedStore

class TestDataWranglingFlow(unittest.TestCase):
    
    def setUp(self):
        self.config = {"useful_columns": ["col1", "col2"]}
        self.flow = DataWranglingFlow(self.config)
        self.shared_store = EnhancedSharedStore()
        
        # Create test data
        test_data = pd.DataFrame({
            'col1': [1, 2, 3, None],
            'col2': ['a', 'b', 'c', 'd'],
            'col3': [1.1, 2.2, 3.3, 4.4]
        })
        self.shared_store.set_dataset("input_dataset", test_data)
    
    def test_flow_execution(self):
        """Test that the flow executes without errors."""
        result = self.flow.run(self.shared_store.data)
        self.assertIsNotNone(result)
        
        # Check that final dataset exists
        final_dataset = self.shared_store.get_dataset("final_dataset")
        self.assertFalse(final_dataset.empty)
    
    def test_sklearn_compatibility(self):
        """Test scikit-learn compatibility wrapper."""
        from src.pocketflow_core_tools.utils.sklearn_compatibility import PocketFlowSklearnWrapper
        
        wrapper = PocketFlowSklearnWrapper(DataWranglingFlow, self.config)
        test_data = self.shared_store.get_dataset("input_dataset")
        
        # Test fit_transform
        result = wrapper.fit_transform(test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
```

### 5. **Migration Strategy**

1. **Phase 1**: Add PocketFlow enhancements to existing classes (no breaking changes)
2. **Phase 2**: Create opt-in enhanced versions of pipelines
3. **Phase 3**: Add scikit-learn compatibility layer
4. **Phase 4**: Gradually migrate individual applications to use enhanced versions
5. **Phase 5**: Add advanced PocketFlow patterns (async, batch processing)

**Application Migration Example**:
```python
# Step 1: Existing application (no changes needed)
from core_tools.core.BasicTaskObjects import BaseExperiment

class MyExperiment(BaseExperiment):
    # Existing code works unchanged
    pass

# Step 2: Opt-in to enhancements when ready
from core_tools.core.BasicTaskObjects import BaseExperimentWithPocketFlow

class MyEnhancedExperiment(BaseExperimentWithPocketFlow):
    # Same interface, enhanced capabilities
    pass
```

## Expected Benefits

### 1. **Enhanced Modularity**
Each pipeline step becomes a reusable Node that can be recombined in different ways.

### 2. **Better Error Handling**
PocketFlow's built-in retry and fallback mechanisms improve pipeline reliability.

### 3. **Improved Testability**
Individual nodes can be tested in isolation, making debugging easier.

### 4. **Flexible Orchestration**
Flows can be nested and combined to create complex workflows.

### 5. **Scikit-learn Integration**
Maintains full compatibility with scikit-learn while adding PocketFlow's benefits.

This refactoring transforms your task-pipeline-kernel architecture into a more flexible, testable, and maintainable system while preserving all existing functionality and scikit-learn compatibility.