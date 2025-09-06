---
layout: default
title: "Agentic Coding for Python Package Development"
---

# Agentic Coding for Python Packages: Humans Design, Agents code!

> If you are an AI agent involved in building Python packages with PocketFlow, read this guide **VERY, VERY** carefully! This is the most important chapter in the entire document. Throughout development, you should always (1) start with a small and simple solution, (2) design at a high level (`docs/design.md`) before implementation, and (3) frequently ask humans for feedback and clarification.
{: .warning }

## Agentic Coding Steps for Python Package Development

Agentic Coding should be a collaboration between Human System Design and Agent Implementation:

| Steps                  | Human      | AI        | Comment                                                                 |
|:-----------------------|:----------:|:---------:|:------------------------------------------------------------------------|
| 1. Package Requirements | ★★★ High  | ★☆☆ Low   | Humans understand the requirements and context.                    |
| 2. Package Architecture | ★★☆ Medium | ★★☆ Medium |  Humans specify the high-level design, and the AI fills in the details. |
| 3. Core Primitives      | ★★☆ Medium | ★★☆ Medium | Humans outline the core abstractions, and the AI helps with implementation. |
| 4. Extension Mechanisms | ★☆☆ Low    | ★★★ High   | AI designs the plugin/extension system, and humans verify.             |
| 5. Module Structure    | ★☆☆ Low   | ★★★ High  | The AI helps design the package module structure based on requirements.  |
| 6. Implementation      | ★☆☆ Low   | ★★★ High  |  The AI implements the package based on the design. |
| 7. Documentation       | ★★☆ Medium | ★★☆ Medium | Humans specify doc requirements, and the AI helps create API docs and examples. |
| 8. Testing & CI/CD     | ★☆☆ Low   | ★★★ High  |  The AI writes test cases and sets up continuous integration.     |

1. **Package Requirements**: Define the scope, purpose, and requirements for your Python package.
    - Understand use cases and target audience:
      - **Who will use it?**: Internal developers, data scientists, AI engineers, etc.
      - **How will they use it?**: As a library, through a CLI, in notebooks, etc.
      - **What problems does it solve?**: Simplify LLM workflow creation, standardize deployment, etc.
    - **API Design First:** Consider API ergonomics and user experience before implementation details.
    - **Balance scope vs. maintainability**: Focus on core functionality that provides maximum value.

2. **Package Architecture**: Design the high-level structure of your package.
    - Identify the core components and their relationships:
      - **Core Primitives**: What are the fundamental building blocks? (e.g., enhanced nodes, flows, patterns)
      - **Extension Points**: Where can users customize or extend functionality?
      - **Integration Interfaces**: How will it connect with other systems and services?
    - Create a package architecture diagram. For example:
      ```mermaid
      flowchart TD
          core[Core Package] --> primitives[PocketFlow Primitives]
          core --> extensions[Extension Mechanisms]
          extensions --> plugins[Plugin System]
          extensions --> hooks[Hook Points]
          
          primitives --> nodes[Enhanced Nodes]
          primitives --> flows[Enhanced Flows]
          primitives --> patterns[Pattern Libraries]
          
          core --> utils[Utility Functions]
      ```
    - > **Design for extensibility from the start!** A well-designed package allows users to easily extend functionality without modifying the core code.
      {: .best-practice }

3. **Core Primitives**: Identify and design the foundational components of your package.
    - Extend PocketFlow's basic primitives for specific use cases:
        - **Enhanced Node Types**: Specialized nodes with additional functionality
        - **Flow Extensions**: Utilities for creating and managing complex flows
        - **Pattern Libraries**: Pre-built templates for common design patterns
    - For each primitive, document its purpose, interface, and example usage.
    - Example core primitive design:
      ```python
      # Enhanced node with validation and typing
      class TypedNode(Node):
          """A node with input/output type validation."""
          
          def __init__(self, input_type=None, output_type=None, **kwargs):
              super().__init__(**kwargs)
              self.input_type = input_type
              self.output_type = output_type
              
          def exec(self, prep_res):
              # Validate input type if specified
              if self.input_type and not isinstance(prep_res, self.input_type):
                  raise TypeError(f"Expected input of type {self.input_type.__name__}, got {type(prep_res).__name__}")
              
              # Call the implementation
              result = self._impl(prep_res)
              
              # Validate output type if specified
              if self.output_type and not isinstance(result, self.output_type):
                  raise TypeError(f"Expected output of type {self.output_type.__name__}, got {type(result).__name__}")
              
              return result
              
          def _impl(self, prep_res):
              """Implementation to be overridden by subclasses."""
              raise NotImplementedError("Subclasses must implement _impl")
      ```
    - > **Maintain backward compatibility**: Ensure your enhanced primitives remain compatible with base PocketFlow when possible.
      {: .best-practice }

4. **Extension Mechanisms**: Design how users will extend and customize your package.
    - Provide clear extension points for common customization needs:
      - **Plugin System**: For registering new node types, flow patterns, or utilities
      - **Hook Points**: Allow insertion of custom logic at specific points in the execution
      - **Configuration System**: Enable customization of behavior without code changes
    - Example extension mechanism:
      ```python
      # Plugin registry system
      class PluginRegistry:
          """Registry for package plugins."""
          
          _plugins = {}
          
          @classmethod
          def register(cls, name=None):
              """Decorator to register a plugin class."""
              def decorator(plugin_class):
                  plugin_name = name or plugin_class.__name__
                  cls._plugins[plugin_name] = plugin_class
                  return plugin_class
              return decorator
          
          @classmethod
          def get_plugin(cls, name):
              """Get a plugin by name."""
              if name not in cls._plugins:
                  raise ValueError(f"Plugin '{name}' not registered")
              return cls._plugins[name]
          
          @classmethod
          def list_plugins(cls):
              """List all registered plugins."""
              return list(cls._plugins.keys())
      
      # Usage example
      @PluginRegistry.register()
      class MyCustomNode(TypedNode):
          """A custom node implementation."""
          pass
      ```

5. **Module Structure**: Plan the organization of your package modules.
   - Design a clean, intuitive module structure:
     - **Core Module**: Contains the fundamental classes and functions
     - **Extensions**: Houses the extension mechanisms and built-in plugins
     - **Utilities**: Common helper functions and tools
     - **CLI**: Command-line interface for the package
   - Example module structure:
     ```
     mypackage/
     ├── __init__.py            # Package initialization, version, imports
     ├── core/                  # Core functionality
     │   ├── __init__.py
     │   ├── nodes.py           # Enhanced node types
     │   ├── flows.py           # Flow extensions
     │   └── patterns.py        # Pattern implementations
     ├── extensions/            # Extension mechanisms
     │   ├── __init__.py
     │   ├── plugins.py         # Plugin system
     │   └── hooks.py           # Hook points
     ├── utils/                 # Utility functions
     │   ├── __init__.py
     │   ├── llm.py             # LLM utilities
     │   └── io.py              # I/O utilities
     ├── cli/                   # Command-line interface
     │   ├── __init__.py
     │   └── commands.py        # CLI commands
     └── examples/              # Example usage
         ├── __init__.py
         └── basic_flow.py      # Basic example
     ```

6. **Implementation**: Develop the package based on the design.
   - 🎉 If you've reached this step, humans have finished the design. Now *Agentic Coding* begins!
   - **Start with packaging structure**:
     - Create `pyproject.toml` and/or `setup.py` to define package metadata
     - Set up proper imports and package initialization
   - **Implement core functionality first**:
     - Focus on the core primitives that extend PocketFlow
     - Ensure they're well-tested and documented
   - **Add extension mechanisms**:
     - Implement the plugin system and hook points
     - Create examples of how to use them
   - **Build utility functions**:
     - Implement common helpers needed by package users

7. **Documentation**:
   - **API Documentation**: Document all public classes, methods, and functions.
   - **Usage Examples**: Provide clear examples for common use cases.
   - **Extension Guide**: Show how to extend and customize the package.
   - **README and Guides**: Create user-friendly documentation for getting started.
   - Example documentation structure:
     ```
     docs/
     ├── api/                   # API reference documentation
     ├── examples/              # Example usage notebooks/scripts
     ├── guides/                # User guides for specific topics
     ├── index.md               # Main documentation page
     └── installation.md        # Installation instructions
     ```

8. **Testing & CI/CD**:
   - **Unit Tests**: Test individual components in isolation.
   - **Integration Tests**: Test how components work together.
   - **End-to-End Tests**: Test complete workflows.
   - **CI/CD Pipeline**: Set up automated testing and deployment.
   - Example test structure:
     ```
     tests/
     ├── unit/                  # Unit tests
     │   ├── test_nodes.py
     │   ├── test_flows.py
     │   └── test_patterns.py
     ├── integration/           # Integration tests
     │   └── test_workflows.py
     └── e2e/                   # End-to-end tests
         └── test_examples.py
     ```

## Python Package Structure

```
mypackage/
├── pyproject.toml            # Modern package definition
├── setup.py                  # Traditional package setup (optional with pyproject.toml)
├── setup.cfg                 # Package metadata and config
├── README.md                 # Package documentation
├── LICENSE                   # License information
├── MANIFEST.in               # Additional files to include
├── src/                      # Source code directory
│   └── mypackage/            # Package directory
│       ├── __init__.py       # Package initialization
│       ├── core/             # Core functionality
│       ├── extensions/       # Extension mechanisms
│       ├── utils/            # Utility functions
│       └── cli/              # Command-line interface
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── e2e/                  # End-to-end tests
├── docs/                     # Documentation
│   ├── api/                  # API reference
│   ├── examples/             # Example usage
│   └── guides/               # User guides
└── examples/                 # Example scripts
    ├── basic_usage.py
    └── advanced_usage.py
```

- **`pyproject.toml`**: Defines the package build system and dependencies.
  ```toml
  [build-system]
  requires = ["setuptools>=42", "wheel"]
  build-backend = "setuptools.build_meta"
  
  [project]
  name = "mypackage"
  version = "0.1.0"
  description = "A package built on PocketFlow primitives"
  readme = "README.md"
  requires-python = ">=3.8"
  license = {text = "MIT"}
  dependencies = [
      "pocketflow>=1.0.0",
      "pyyaml>=6.0",
  ]
  
  [project.optional-dependencies]
  dev = [
      "pytest>=7.0.0",
      "black>=22.0.0",
      "isort>=5.0.0",
  ]
  
  [project.scripts]
  mypackage = "mypackage.cli:main"
  ```

- **`src/mypackage/__init__.py`**: Package initialization and version.
  ```python
  """A package built on PocketFlow primitives."""

  __version__ = "0.1.0"
  
  # Import core components for easy access
  from mypackage.core.nodes import TypedNode
  from mypackage.core.flows import EnhancedFlow
  from mypackage.extensions.plugins import PluginRegistry
  
  # Define what's available via `from mypackage import *`
  __all__ = [
      "TypedNode",
      "EnhancedFlow",
      "PluginRegistry",
  ]
  ```

- **`docs/design.md`**: Contains project design documentation, similar to the existing template but focused on package design.
  ```markdown
  # Design Doc: Your Package Name

  > Please DON'T remove notes for AI

  ## Package Requirements

  > Notes for AI: Keep it simple and clear.
  > Define the purpose, scope, and requirements of the package.

  ## Package Architecture

  > Notes for AI:
  > 1. Design the high-level structure of the package.
  > 2. Identify core components and their relationships.

  ### Core Components:

  1. Enhanced PocketFlow Primitives:
     - TypedNode: Node with input/output type validation
     - EnhancedFlow: Flow with additional capabilities
  
  2. Extension Mechanisms:
     - PluginRegistry: System for registering custom components
     - HookSystem: Points for inserting custom logic

  ```mermaid
  flowchart TD
      core[Core Package] --> primitives[PocketFlow Primitives]
      primitives --> nodes[Enhanced Nodes]
      primitives --> flows[Enhanced Flows]
      core --> extensions[Extension Mechanisms]
  ```
  
  ## Core Primitives

  > Notes for AI:
  > 1. Define the core primitives that extend PocketFlow.
  > 2. Document their purpose, interface, and example usage.

  1. **TypedNode** (`core/nodes.py`)
    - *Purpose*: Provide type validation for node inputs/outputs
    - *Interface*: 
      ```python
      TypedNode(input_type=dict, output_type=str, **kwargs)
      ```
    - *Example*: 
      ```python
      class SummarizeNode(TypedNode):
          def __init__(self):
              super().__init__(input_type=str, output_type=str)
          
          def _impl(self, text):
              return call_llm(f"Summarize: {text}")
      ```

  ## Extension Mechanisms

  > Notes for AI: Design how users will extend and customize the package.

  1. **PluginRegistry** (`extensions/plugins.py`)
    - *Purpose*: Registry for package plugins
    - *Interface*: 
      ```python
      @PluginRegistry.register(name=None)
      class MyPlugin: ...
      
      plugin_class = PluginRegistry.get_plugin("name")
      all_plugins = PluginRegistry.list_plugins()
      ```

  ## Module Structure

  > Notes for AI: Plan the organization of the package modules.

  The package module structure is organized as follows:

  ```
  mypackage/
  ├── __init__.py
  ├── core/
  │   ├── __init__.py
  │   ├── nodes.py
  │   └── flows.py
  ├── extensions/
  │   ├── __init__.py
  │   └── plugins.py
  └── utils/
      ├── __init__.py
      └── helpers.py
  ```
  ```

- **`src/mypackage/core/nodes.py`**: Contains enhanced node implementations.
  ```python
  """Enhanced node implementations that extend PocketFlow's Node."""

  from typing import Any, Callable, Optional, Type
  from pocketflow import Node
  
  class TypedNode(Node):
      """A node with input/output type validation."""
      
      def __init__(
          self, 
          input_type: Optional[Type] = None, 
          output_type: Optional[Type] = None,
          **kwargs
      ):
          """Initialize the typed node.
          
          Args:
              input_type: Expected type of prep_res input
              output_type: Expected type of exec result
              **kwargs: Additional arguments passed to Node
          """
          super().__init__(**kwargs)
          self.input_type = input_type
          self.output_type = output_type
          
      def exec(self, prep_res: Any) -> Any:
          """Execute with type validation.
          
          Args:
              prep_res: Result from prep() method
              
          Returns:
              Result of _impl() method
              
          Raises:
              TypeError: If input or output type validation fails
          """
          # Validate input type if specified
          if self.input_type and not isinstance(prep_res, self.input_type):
              raise TypeError(
                  f"Expected input of type {self.input_type.__name__}, "
                  f"got {type(prep_res).__name__}"
              )
          
          # Call the implementation
          result = self._impl(prep_res)
          
          # Validate output type if specified
          if self.output_type and not isinstance(result, self.output_type):
              raise TypeError(
                  f"Expected output of type {self.output_type.__name__}, "
                  f"got {type(result).__name__}"
              )
          
          return result
          
      def _impl(self, prep_res: Any) -> Any:
          """Implementation to be overridden by subclasses.
          
          Args:
              prep_res: Result from prep() method
              
          Returns:
              Result to be passed to post() method
              
          Raises:
              NotImplementedError: If not overridden by subclass
          """
          raise NotImplementedError("Subclasses must implement _impl")
  
  
  class FunctionNode(TypedNode):
      """A node that wraps a function."""
      
      def __init__(
          self, 
          func: Callable[[Any], Any],
          input_type: Optional[Type] = None,
          output_type: Optional[Type] = None,
          **kwargs
      ):
          """Initialize with a function.
          
          Args:
              func: Function to call in _impl
              input_type: Expected type of prep_res input
              output_type: Expected type of exec result
              **kwargs: Additional arguments passed to TypedNode
          """
          super().__init__(input_type=input_type, output_type=output_type, **kwargs)
          self.func = func
          
      def _impl(self, prep_res: Any) -> Any:
          """Call the wrapped function.
          
          Args:
              prep_res: Result from prep() method
              
          Returns:
              Result of calling self.func with prep_res
          """
          return self.func(prep_res)
  
  
  # Example of how to use FunctionNode
  if __name__ == "__main__":
      # Create a node that doubles a number
      double_node = FunctionNode(
          func=lambda x: x * 2,
          input_type=int,
          output_type=int
      )
      
      # Use the node
      result = double_node.exec(5)
      print(f"Result: {result}")  # Result: 10
      
      try:
          double_node.exec("not a number")  # This will raise TypeError
      except TypeError as e:
          print(f"Error: {e}")
  ```

- **`tests/unit/test_nodes.py`**: Unit tests for node implementations.
  ```python
  """Unit tests for enhanced node implementations."""

  import pytest
  from mypackage.core.nodes import TypedNode, FunctionNode
  
  class TestTypedNode:
      """Tests for TypedNode class."""
      
      class SampleNode(TypedNode):
          """Sample implementation for testing."""
          
          def _impl(self, prep_res):
              return str(prep_res)
      
      def test_valid_types(self):
          """Test with valid input and output types."""
          node = self.SampleNode(input_type=int, output_type=str)
          result = node.exec(42)
          assert result == "42"
          assert isinstance(result, str)
      
      def test_invalid_input_type(self):
          """Test with invalid input type."""
          node = self.SampleNode(input_type=int, output_type=str)
          with pytest.raises(TypeError):
              node.exec("not an int")
      
      def test_invalid_output_type(self):
          """Test with invalid output type."""
          class BadNode(TypedNode):
              def _impl(self, prep_res):
                  return 42  # Always returns int
                  
          node = BadNode(input_type=str, output_type=str)
          with pytest.raises(TypeError):
              node.exec("input")
  
  class TestFunctionNode:
      """Tests for FunctionNode class."""
      
      def test_function_call(self):
          """Test that the function is called correctly."""
          node = FunctionNode(
              func=lambda x: x.upper(),
              input_type=str,
              output_type=str
          )
          result = node.exec("hello")
          assert result == "HELLO"
      
      def test_with_type_validation(self):
          """Test with type validation."""
          node = FunctionNode(
              func=len,
              input_type=list,
              output_type=int
          )
          result = node.exec([1, 2, 3])
          assert result == 3
          
          with pytest.raises(TypeError):
              node.exec("not a list")
  ```

## Usage Examples

### Creating a New Python Package

1. **Set up the package structure**:
   ```python
   # Create basic structure
   import os
   
   def create_package_structure(name):
       """Create the basic package structure."""
       directories = [
           f"src/{name}",
           f"src/{name}/core",
           f"src/{name}/extensions",
           f"src/{name}/utils",
           "tests/unit",
           "tests/integration",
           "docs",
           "examples"
       ]
       
       for directory in directories:
           os.makedirs(directory, exist_ok=True)
       
       # Create __init__.py files
       init_files = [
           f"src/{name}/__init__.py",
           f"src/{name}/core/__init__.py",
           f"src/{name}/extensions/__init__.py",
           f"src/{name}/utils/__init__.py",
       ]
       
       for init_file in init_files:
           with open(init_file, "w") as f:
               f.write(f'"""Package {name}."""\n\n__version__ = "0.1.0"\n')
       
       # Create pyproject.toml
       with open("pyproject.toml", "w") as f:
           f.write(f'''[build-system]
   requires = ["setuptools>=42", "wheel"]
   build-backend = "setuptools.build_meta"
   
   [project]
   name = "{name}"
   version = "0.1.0"
   description = "A package built on PocketFlow primitives"
   readme = "README.md"
   requires-python = ">=3.8"
   license = {{text = "MIT"}}
   dependencies = [
       "pocketflow>=1.0.0",
       "pyyaml>=6.0",
   ]
   ''')
       
       # Create README.md
       with open("README.md", "w") as f:
           f.write(f"# {name}\n\nA package built on PocketFlow primitives.\n")
   
   # Example usage
   create_package_structure("mypocketpackage")
   ```

2. **Extending PocketFlow primitives**:
   ```python
   # src/mypocketpackage/core/nodes.py
   from pocketflow import Node
   
   class EnhancedNode(Node):
       """An enhanced version of PocketFlow's Node."""
       
       def __init__(self, description=None, **kwargs):
           super().__init__(**kwargs)
           self.description = description
           
       def exec(self, prep_res):
           """Add logging before calling the implementation."""
           print(f"Executing node: {self.description or self.__class__.__name__}")
           return super().exec(prep_res)
   ```

3. **Creating a plugin system**:
   ```python
   # src/mypocketpackage/extensions/plugins.py
   class PluginRegistry:
       """Registry for package plugins."""
       
       _plugins = {}
       
       @classmethod
       def register(cls, name=None):
           """Decorator to register a plugin class."""
           def decorator(plugin_class):
               plugin_name = name or plugin_class.__name__
               cls._plugins[plugin_name] = plugin_class
               return plugin_class
           return decorator
       
       @classmethod
       def get_plugin(cls, name):
           """Get a plugin by name."""
           if name not in cls._plugins:
               raise ValueError(f"Plugin '{name}' not registered")
           return cls._plugins[name]
       
       @classmethod
       def list_plugins(cls):
           """List all registered plugins."""
           return list(cls._plugins.keys())
   ```

4. **Creating a command-line interface**:
   ```python
   # src/mypocketpackage/cli/__init__.py
   import argparse
   import sys
   
   def main():
       """Main entry point for the CLI."""
       parser = argparse.ArgumentParser(description="MyPocketPackage CLI")
       parser.add_argument("--version", action="store_true", help="Show version information")
       
       args = parser.parse_args()
       
       if args.version:
           from mypocketpackage import __version__
           print(f"MyPocketPackage version {__version__}")
           return 0
       
       parser.print_help()
       return 0
       
   if __name__ == "__main__":
       sys.exit(main())
   ```

Remember that creating a good Python package requires attention to detail and a focus on user experience. The package should be easy to use, well-documented, and follow Python best practices.

> **You'll likely iterate a lot!** Expect to refine your package structure and API design multiple times based on user feedback and real-world usage.
{: .best-practice }
