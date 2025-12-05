import torch
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


typed = jaxtyped(typechecker=typechecker)
typed = lambda f: f
