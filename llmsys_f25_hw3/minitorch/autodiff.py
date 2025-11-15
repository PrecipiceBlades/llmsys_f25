from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Set, List, Dict

from typing_extensions import Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """
        Accumulates the derivative (gradient) for this Variable.

        Args:
            x (Any): The gradient value to be accumulated.
        """
        pass

    @property
    def unique_id(self) -> int:
        """
        Returns:
            int: The unique identifier of this Variable.
        """
        pass

    def is_leaf(self) -> bool:
        """
        Returns whether this Variable is a leaf node in the computation graph.

        Returns:
            bool: True if this Variable is a leaf node, False otherwise.
        """
        pass

    def is_constant(self) -> bool:
        """
        Returns whether this Variable represents a constant value.

        Returns:
            bool: True if this Variable is constant, False otherwise.
        """
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """
        Returns the parent Variables of this Variable in the computation graph.

        Returns:
            Iterable[Variable]: The parent Variables of this Variable.
        """
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """
        Implements the chain rule to compute the gradient contributions of this Variable.

        Args:
            d_output (Any): The gradient of the output with respect to the Variable.

        Returns:
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple
                contains a parent Variable and the corresponding gradient contribution.
        """
        pass


def topological_sort_dfs(variable: Variable, visited: Set[int], result: List[Variable]) -> None:
    """
    Performs a post-order depth-first search to compute the topological order of the computation graph.
    
    Args:
        variable: The current variable to process
        visited: A set of visited variable IDs to avoid cycles
        result: A list to store the topological order
    """
    if variable.is_constant() or variable.unique_id in visited:
        return
    visited.add(variable.unique_id)
    for parent in variable.parents:
        topological_sort_dfs(parent, visited, result)
    result.append(variable)


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # BEGIN ASSIGN1_1
    visited: Set[int] = set()
    result: List[Variable] = []
    topological_sort_dfs(variable, visited, result)
    return list(reversed(result))
    # END ASSIGN1_1


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # BEGIN ASSIGN1_1
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[int, List[Variable]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[variable.unique_id] = [deriv]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = topological_sort(variable)
    
    # Reset gradients for all nodes in the computation graph
    # This is **very important** to ensure that we are not accumulating gradients from multiple backward passes!!!!
    for node in reverse_topo_order:
        node.zero_grad_()
    
    for node in reverse_topo_order:
        # Sum all adjoints into grad (for both leaf and non-leaf nodes)
        if node.unique_id in node_to_output_grads_list:
            for adjoint in node_to_output_grads_list[node.unique_id]:
               node.accumulate_derivative(adjoint)
            # For non-leaf nodes, compute gradients for inputs
            if not node.is_leaf():
                all_partial_adjoints = node.chain_rule(node.grad)
                for parent, partial_adjoint in all_partial_adjoints:
                    if parent.unique_id not in node_to_output_grads_list:
                        node_to_output_grads_list[parent.unique_id] = []
                    node_to_output_grads_list[parent.unique_id].append(partial_adjoint)
    # END ASSIGN1_1

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
