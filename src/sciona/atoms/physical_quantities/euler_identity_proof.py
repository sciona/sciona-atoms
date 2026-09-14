"""Exact symbolic realization of Euler's identity at pi, with proof steps."""
import sympy as sp
from sciona.ghost.registry import register_atom


def witness_euler_identity_proof() -> dict:
    return {'schema':'sciona.euler-proof.v1','step_count':4,'output_kind':'exact symbolic proof certificate'}


@register_atom(witness_euler_identity_proof)
def euler_identity_proof() -> dict:
    """Compute and verify the fixed identity exp(I*pi)+1=0 symbolically.

    Takes no inputs. Returns a JSON-safe certificate with an explicit initial
    real-angle Euler identity, four computed transformation steps and a final
    exact equality. I denotes the imaginary unit and pi the exact mathematical
    constant; no floating-point approximation or user symbol inference is used.
    Each intermediate is an unevaluated Equality serialized with SymPy srepr,
    preserving both sides even when equal. Transformations use SymPy's exact
    algebra; validation checks the root identity and every computed equality.

    This is a fixed mathematical proof, not a generic theorem prover, numerical
    trigonometric solver or dynamical simulation. The initial Euler identity is
    checked using SymPy's complex-exponential expansion, not derived from first
    principles. The certificate records that verification scope explicitly.
    """
    theta=sp.Symbol('theta',real=True)
    eq=lambda left,right:sp.Eq(left,right,evaluate=False)
    initial=eq(sp.exp(sp.I*theta),sp.cos(theta)+sp.I*sp.sin(theta))
    if sp.simplify(sp.expand_complex(initial.lhs-initial.rhs))!=0:
        raise ValueError('Initial Euler identity verification failed')
    specialized=eq(initial.lhs.subs(theta,sp.pi),initial.rhs.subs(theta,sp.pi))
    simplified=eq(sp.simplify(specialized.lhs),sp.simplify(specialized.rhs))
    reaffirmed=eq(sp.simplify(simplified.lhs),sp.simplify(simplified.rhs))
    final=eq(reaffirmed.lhs+1,reaffirmed.rhs+1)
    steps=[specialized,simplified,reaffirmed,final]
    if any(sp.simplify(step.lhs-step.rhs)!=0 for step in steps):
        raise ValueError('Computed Euler proof step verification failed')
    if final.lhs!=0 or final.rhs!=0:raise ValueError('Exact terminal equality differs')
    return dict(schema='sciona.euler-proof.v1',initial_equation=sp.srepr(initial),
        steps=[dict(operation=operation,equation=sp.srepr(step)) for operation,step in zip(
            ['substitute theta=pi','simplify','simplify','add one to both sides'],steps)],
        terminal_equation=sp.srepr(final),
        verification_scope='Exact symbolic identities under I as imaginary unit, pi exact and initial angle real; root checked by complex-exponential expansion.')
