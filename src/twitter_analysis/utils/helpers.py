from typing import Callable, TypeVar, List, Any, Optional, Iterator
from functools import wraps, reduce
import time
import logging
from pathlib import Path

T = TypeVar('T')
U = TypeVar('U')


def compose(*functions: Callable) -> Callable:
    """
    Compose multiple functions into a single function.
    
    Args:
        *functions: Functions to compose (applied right to left)
        
    Returns:
        Composed function
    """
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def pipe(data: T, *functions: Callable[[T], T]) -> T:
    """
    Pipe data through a series of functions (left to right).
    
    Args:
        data: Initial data
        *functions: Functions to apply in sequence
        
    Returns:
        Transformed data
    """
    return reduce(lambda acc, func: func(acc), functions, data)


def curry(func: Callable) -> Callable:
    """
    Convert a function to curried form.
    
    Args:
        func: Function to curry
        
    Returns:
        Curried function
    """
    @wraps(func)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return lambda *more_args, **more_kwargs: curried(*(args + more_args), **{**kwargs, **more_kwargs})
    return curried


def memoize(func: Callable) -> Callable:
    """
    Memoization decorator for caching function results.
    
    Args:
        func: Function to memoize
        
    Returns:
        Memoized function
    """
    cache = {}
    
    @wraps(func)
    def memoized(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoized




def safe_execute(func: Callable, default: Any = None) -> Callable:
    """
    Create a safe version of a function that returns default on exception.
    
    Args:
        func: Function to make safe
        default: Default value to return on exception
        
    Returns:
        Safe function
    """
    @wraps(func)
    def safe(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Function {func.__name__} failed with error: {e}")
            return default
    return safe


def chunk_list(lst: List[T], chunk_size: int) -> Iterator[List[T]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of the list
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def flatten(nested_list: List[List[T]]) -> List[T]:
    """
    Flatten a nested list.
    
    Args:
        nested_list: List of lists
        
    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def filter_none(lst: List[Optional[T]]) -> List[T]:
    """
    Filter out None values from a list.
    
    Args:
        lst: List that may contain None values
        
    Returns:
        List with None values removed
    """
    return [item for item in lst if item is not None]


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional log file path
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
