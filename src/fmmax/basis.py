"""Functions related to vectors and field expansion in the FMM scheme.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import dataclasses
import enum
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as onp

from fmmax import utils

# Officially defines the x- and y- directions. By convention, the x-axis preceeds
# the y-axis in our array indexing scheme.
X: jnp.ndarray = jnp.array([1.0, 0.0])
Y: jnp.ndarray = jnp.array([0.0, 1.0])


@dataclasses.dataclass
class LatticeVectors:
    """Stores a pair of lattice vectors.

    Note that this is just a pair of 2-dimensional vectors, which may be either for
    the real-space lattice or the reciprocal space lattice, depending on usage.

    Attributes:
        u: The first primitive lattice vector.
        v: The second primitive lattice vector, with identical shape.
    """

    u: jnp.ndarray
    v: jnp.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.u, jnp.ndarray):
            if self.u.shape[-1] != 2 or self.v.shape[-1] != 2:
                raise ValueError(
                    f"`u` and `v` must have a trailing length of 2, but got shapes "
                    f"{self.u.shape} and {self.v.shape}."
                )

    @property
    def reciprocal(self) -> "LatticeVectors":
        """Returns the corresponding vectors on the reciprocal lattice."""
        return reciprocal(self)


@dataclasses.dataclass
class Expansion:
    """Stores the expansion.

    The expansion consists of the integer coefficients of the reciprocal lattice vectors
    used in the Fourier expansion of fields, permittivities, etc. in the FMM scheme.

    Attributes:
        basis_coefficients: The integer coefficients of the primitive reciprocal lattice
            vectors, which generate the full set of reciprocal-space vectors in the expansion.
        num_terms: The number of terms in the expansion.
    """

    basis_coefficients: onp.ndarray

    def __post_init__(self) -> None:
        if self.basis_coefficients.ndim != 2 or self.basis_coefficients.shape[-1] != 2:
            raise ValueError(
                f"`basis_coefficients` must have shape `(num, 2)` but got "
                f"{self.basis_coefficients.shape}."
            )

    @property
    def num_terms(self) -> int:
        return self.basis_coefficients.shape[-2]


@enum.unique
class Truncation(enum.Enum):
    """Enumerates truncation modes."""

    CIRCULAR: str = "circular"
    PARALLELOGRAMIC: str = "parallelogramic"
    ANNULUS: str = "annulus"
    CUSTOM: str = "custom"
    TOPM: str = "topM"


def generate_expansion(
    primitive_lattice_vectors: LatticeVectors,
    approximate_num_terms: int,
    truncation: Truncation,
    factor: float | None = None,
    kmin_abs: float | None = None,
    keep_zero_order: bool = True,
    custom_basis_coefficients: onp.ndarray | None = None,
    mask: jnp.ndarray | None = None,
) -> Expansion:
    """Generates the expansion for the specified real-space basis.

    Args:
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        approximate_num_terms: The approximate number of terms in the expansion. To
            maintain a symmetric expansion, the total number of terms may differ from
            this value.
        truncation: The Truncation object to be used for the expansion: "CIRCULAR", "PARALLELOGRAMIC", "ANNULUS", "CUSTOM", or "TOPM".
        factor: For ANNULUS truncation, the inner radius factor (0-1). For TOPM truncation, the factor to multiply `approximate_num_terms` to get `N`. Ignored for other modes.
        kmin_abs: For ANNULUS truncation, the absolute inner radius. Ignored for other modes.
        keep_zero_order: For ANNULUS truncation, whether to include the zero-order term. Ignored for other modes.
        custom_basis_coefficients: For CUSTOM truncation, an integer-valued array
            with shape `(num_terms, 2)` containing the basis coefficients. Ignored
            for other modes.

    Returns:
        The `Expansion`. The basis coefficients of the expansion are sorted so that
        the zeroth-order term is first (for circular and parallelogramic) or by magnitude 
        with zero-order first (for annulus).
    """
    reciprocal_vectors = primitive_lattice_vectors.reciprocal
    if truncation == Truncation.CIRCULAR:
        basis_coefficients = _basis_coefficients_circular(
            reciprocal_vectors, approximate_num_terms
        )
    elif truncation == Truncation.PARALLELOGRAMIC:
        basis_coefficients = _basis_coefficients_parallelogramic(
            reciprocal_vectors, approximate_num_terms
        )
    elif truncation == Truncation.ANNULUS:
        basis_coefficients = _basis_coefficients_annulus(
            reciprocal_vectors,
            approximate_num_terms,
            factor=factor,
            kmin_abs=kmin_abs,
            keep_zero_order=keep_zero_order,
        )
    elif truncation == Truncation.CUSTOM:
        basis_coefficients = _basis_coefficients_custom(custom_basis_coefficients)
    elif truncation == Truncation.TOPM:
        basis_coefficients = _basis_coefficients_topM(
            primitive_lattice_vectors=reciprocal_vectors,
            M=approximate_num_terms,
            x=mask,
        )  
    else:
        raise ValueError(f"Unknown `truncation`, got {truncation}.")
    return Expansion(basis_coefficients)


def reciprocal(lattice_vectors: LatticeVectors) -> LatticeVectors:
    """Computes the reciprocal vectors for the `basis`."""
    cross_product = _cross_product(lattice_vectors.u, lattice_vectors.v)
    uprime = (
        jnp.stack([lattice_vectors.v[..., 1], -lattice_vectors.v[..., 0]], axis=-1)
        / cross_product
    )
    vprime = (
        jnp.stack([-lattice_vectors.u[..., 1], lattice_vectors.u[..., 0]], axis=-1)
        / cross_product
    )
    return LatticeVectors(u=uprime, v=vprime)


def unit_cell_coordinates(
    primitive_lattice_vectors: LatticeVectors,
    shape: Tuple[int, int],
    num_unit_cells: Tuple[int, int],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes coordinates within unit cells, given the shape and number of unit cells."""
    i_stop = num_unit_cells[0] * shape[0]
    j_stop = num_unit_cells[1] * shape[1]
    i, j = tuple(
        jnp.meshgrid(
            jnp.arange(0, i_stop) / shape[0],
            jnp.arange(0, j_stop) / shape[1],
            indexing="ij",
        )
    )
    x = (
        i * primitive_lattice_vectors.u[..., jnp.newaxis, jnp.newaxis, 0]
        + j * primitive_lattice_vectors.v[..., jnp.newaxis, jnp.newaxis, 0]
    )
    y = (
        i * primitive_lattice_vectors.u[..., jnp.newaxis, jnp.newaxis, 1]
        + j * primitive_lattice_vectors.v[..., jnp.newaxis, jnp.newaxis, 1]
    )
    return x, y


# -----------------------------------------------------------------------------
# Functions related to transverse wavevectors.
# -----------------------------------------------------------------------------


def plane_wave_in_plane_wavevector(
    wavelength: jnp.ndarray,
    polar_angle: jnp.ndarray,
    azimuthal_angle: jnp.ndarray,
    permittivity: jnp.ndarray,
) -> jnp.ndarray:
    """Computes the in-plane wavevector for a plane-wave excitation.

    Args:
        wavelength: The free-space wavelength of the plane-wave excitation.
        polar_angle: Polar angle of the plane-wave excitation.
        azimuthal_angle: Azimuthal angle of plane-wave the excitation.
        permittivity: Scalar permittivity of the medium in which the polar angle
            and azimuthal angle are specified.

    Returns:
        The fundamental transverse wavevector, i.e. `(kx0, ky0)`.
    """
    angular_frequency = utils.angular_frequency_for_wavelength(wavelength)
    kx0 = (
        angular_frequency
        * jnp.sin(polar_angle)
        * jnp.cos(azimuthal_angle)
        * jnp.sqrt(permittivity.real)
    )
    ky0 = (
        angular_frequency
        * jnp.sin(polar_angle)
        * jnp.sin(azimuthal_angle)
        * jnp.sqrt(permittivity.real)
    )
    return jnp.stack([kx0, ky0], axis=-1)


def brillouin_zone_in_plane_wavevector(
    brillouin_grid_shape: Tuple[int, int],
    primitive_lattice_vectors: LatticeVectors,
) -> jnp.ndarray:
    """Computes in-plane wavevectors on a regular grid in the first Brillouin zone.

    The wavevectors are found by dividing the Brillouin zone into a grid with the
    specified shape; the wavevectors are at the centers of the grid voxels.

    Args:
        brillouin_grid_shape: The shape of the wavevector grid.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.

    Returns:
        The in-plane wavevectors, with shape `brillouin_grid_shape + (2,)`.
    """
    if len(brillouin_grid_shape) != 2 or brillouin_grid_shape < (1, 1):
        raise ValueError(
            f"`brillouin_grid_shape` must be length-2 with positive values, "
            f"but got {brillouin_grid_shape}."
        )

    spacing_i = 1 / brillouin_grid_shape[0]
    spacing_j = 1 / brillouin_grid_shape[1]
    i, j = jnp.meshgrid(
        jnp.arange(-0.5 + spacing_i / 2, 0.5, spacing_i),
        jnp.arange(-0.5 + spacing_j / 2, 0.5, spacing_j),
        indexing="ij",
    )
    reciprocal_vectors = primitive_lattice_vectors.reciprocal
    return jnp.stack(
        [
            2
            * jnp.pi
            * (i * reciprocal_vectors.u[..., 0] + j * reciprocal_vectors.v[..., 0]),
            2
            * jnp.pi
            * (i * reciprocal_vectors.u[..., 1] + j * reciprocal_vectors.v[..., 1]),
        ],
        axis=-1,
    )


def transverse_wavevectors(
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
) -> jnp.ndarray:
    """Obtains all the relevant transverse wavevectors given the expansion.

    Args:
        in_plane_wavevector: The zeroth-order transverse wavevector.
        primitive_lattice_vectors: The primitive vectors for the real-space lattice.
        expansion: The expansion for which the set of transverse wavevectors are sought.

    Returns:
        The transverse wavevectors.
    """
    reciprocal_vectors = primitive_lattice_vectors.reciprocal
    kx = in_plane_wavevector[..., 0, jnp.newaxis] + 2 * jnp.pi * (
        expansion.basis_coefficients[:, 0] * reciprocal_vectors.u[..., 0]
        + expansion.basis_coefficients[:, 1] * reciprocal_vectors.v[..., 0]
    )
    ky = in_plane_wavevector[..., 1, jnp.newaxis] + 2 * jnp.pi * (
        expansion.basis_coefficients[:, 0] * reciprocal_vectors.u[..., 1]
        + expansion.basis_coefficients[:, 1] * reciprocal_vectors.v[..., 1]
    )
    return jnp.stack([kx, ky], axis=-1)


# -----------------------------------------------------------------------------
# Functions to generate basis coefficients with various truncations.
# -----------------------------------------------------------------------------


def _basis_coefficients_circular(
    primitive_lattice_vectors: LatticeVectors,
    approximate_num_terms: int,
) -> onp.ndarray:
    """Computes the basis coefficients of lattice vectors lying within a circle.

    The coefficients generate a set of lattice vectors from a basis using a circular
    truncation, so that all lattice vectors lying within a circular region with area
    given by `num * u ⨉ v` are included, where `⨉` is the cross product. The size of
    the set will generally be close to `approximate_num_terms`.

    Args:
        primitive_lattice_vectors: Primitive vectors for the reciprocal-space lattice.
        approximate_num_terms: The approximate number of terms in the expansion. To
            maintain a symmetric expansion, the total number of terms may differ from
            this value.

    Returns:
        The coefficients, with shape `(num_vectors, 2)`. The final axis gives the
        coefficient for the first and second vector in the basis.
    """
    # Generate candidate coefficients. These will include more coefficients than needed;
    # subsequently we will filter based on magnitude.
    g = onp.arange(-approximate_num_terms // 2, approximate_num_terms // 2 + 1)
    G1, G2 = onp.meshgrid(g, g, indexing="ij")
    G1 = G1.flatten()
    G2 = G2.flatten()

    # Generate the actual vectors and compute their magnitude.
    vectors = (
        G1[..., onp.newaxis] * primitive_lattice_vectors.u[..., onp.newaxis, :]
        + G2[..., onp.newaxis] * primitive_lattice_vectors.v[..., onp.newaxis, :]
    )
    magnitude = onp.linalg.norm(vectors, axis=-1)

    # Include all vectors lying within the circle with area equal to cross product
    # of `u` and `v` scaled by `num_terms`.
    max_magnitude = onp.sqrt(
        approximate_num_terms
        * onp.abs(
            _cross_product(primitive_lattice_vectors.u, primitive_lattice_vectors.v)
        )
        / onp.pi
    )
    mask = magnitude < max_magnitude

    G1 = G1[mask]
    G2 = G2[mask]
    magnitude = magnitude[mask]
    G = onp.stack([G1, G2], axis=-1)

    order = onp.argsort(magnitude)
    return G[..., order, :]


def _basis_coefficients_parallelogramic(
    primitive_lattice_vectors: LatticeVectors,
    approximate_num_terms: int,
) -> onp.ndarray:
    """Computes the basis coefficients of lattice vectors lying within a parallelogram.

    The coefficients generate a set of lattice vectors from a basis using a
    parallelogramic truncation. The size of the set will generally be close to
    `approximate_num_terms`.

    Args:
        primitive_lattice_vectors: Primitive vectors for the reciprocal-space lattice.
        approximate_num_terms: The approximate number of terms in the expansion. To
            maintain a full parallelogram expansion, the total number of terms may
            differ from this value.

    Returns:
        The coefficients, with shape `(num_vectors, 2)`. The final axis gives the
        coefficient for the first and second vector in the basis.
    """

    ku_spacing = onp.sqrt(onp.sum(onp.abs(primitive_lattice_vectors.u) ** 2))
    kv_spacing = onp.sqrt(onp.sum(onp.abs(primitive_lattice_vectors.v) ** 2))

    # Solve for `(nu, nv)` such that we approximately satisfy
    #     (2 * nu + 1) * (2 * nv + 1) = approximate_num_terms
    # and
    #     nu / nv = kv_spacing / ku_spacing
    # Since `ku_spacing` and `kv_spacing` give the spacing of points in reciprocal
    # space along the ku and kv directions, respectively, this second equation ensures
    # that we are chosing points in a parallelogram with equal-length sides in k-space.
    # (Note that while the sides of the parallelogram have equal length, the number of
    # points along each direction will differ, as these are spaced by `ku_spacing` and
    # `kv_spacing`.)

    def _solve_quadratic(ratio):
        a = 4 * ratio
        b = 2 * (ratio + 1)
        c = 1 - approximate_num_terms
        nu = (-b + onp.sqrt(b**2 - 4 * a * c)) / (2 * a)
        return int(onp.around(nu))

    nu = _solve_quadratic(ku_spacing / kv_spacing)
    nv = _solve_quadratic(kv_spacing / ku_spacing)

    G1, G2 = onp.meshgrid(
        onp.arange(-nu, nu + 1),
        onp.arange(-nv, nv + 1),
        indexing="ij",
    )
    G1 = G1.flatten()
    G2 = G2.flatten()
    G = onp.stack([G1, G2], axis=-1)
    # Generate the actual vectors and compute their magnitude.
    vectors = (
        G1[..., onp.newaxis] * primitive_lattice_vectors.u[..., onp.newaxis, :]
        + G2[..., onp.newaxis] * primitive_lattice_vectors.v[..., onp.newaxis, :]
    )
    magnitude = onp.linalg.norm(vectors, axis=-1)

    order = onp.argsort(magnitude)
    return G[..., order, :]


def _cross_product(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Computes the cross product of 2D vectors `x` and `y`."""
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def _basis_coefficients_custom(
    custom_basis_coefficients: onp.ndarray | None,
) -> onp.ndarray:
    """Validates and returns user-provided custom basis coefficients."""
    if custom_basis_coefficients is None:
        raise ValueError(
            "`custom_basis_coefficients` must be provided for CUSTOM truncation."
        )

    coeffs = onp.asarray(custom_basis_coefficients)
    if coeffs.ndim != 2 or coeffs.shape[-1] != 2 or coeffs.shape[0] == 0:
        raise ValueError(
            "`custom_basis_coefficients` must have shape `(num_terms>0, 2)` but got "
            f"{coeffs.shape}."
        )
    if not onp.issubdtype(coeffs.dtype, onp.integer):
        raise ValueError(
            "`custom_basis_coefficients` must have an integer dtype, but got "
            f"{coeffs.dtype}."
        )


    _, unique_indices = onp.unique(coeffs, axis=0, return_index=True)
    if len(unique_indices) != len(coeffs):
        raise ValueError("`custom_basis_coefficients` contains duplicate rows.")
    unique_coeffs = coeffs[onp.sort(unique_indices)]

    zero_mask = onp.all(unique_coeffs == onp.array([0, 0]), axis=1)
    if onp.any(zero_mask):
        unique_coeffs = onp.vstack([unique_coeffs[zero_mask], unique_coeffs[~zero_mask]])

    return unique_coeffs

def _basis_coefficients_topM(
    primitive_lattice_vectors: LatticeVectors,
    M: int,
    x: jnp.ndarray,
    axes: Tuple[int, int] = (-2, -1),
) -> onp.ndarray:
    if x is None:
        raise ValueError("`x` must be provided for TOPM truncation.")
    axes: Tuple[int, int] = utils.absolute_axes(axes, x.ndim)  
    x_fft = jnp.fft.fft2(x, axes=axes, norm="forward")
    leading_dims = len(x.shape[: axes[0]])
    trailing_dims = len(x.shape[axes[1] + 1 :])

    candidate_coeffs = _basis_coefficients_parallelogramic(primitive_lattice_vectors=primitive_lattice_vectors,
                                                           approximate_num_terms=x.shape[axes[0]] * x.shape[axes[1]])
    
    
    
    slices = (
        [slice(None)] * leading_dims
        + [candidate_coeffs[:, 0], candidate_coeffs[:, 1]]
        + [slice(None)] * trailing_dims
    )
    
    x_fft = x_fft[tuple(slices)]

    magnitude = onp.abs(x_fft)
    order = onp.argsort(-magnitude)
    sorted_coeffs = candidate_coeffs[..., order, :]
    return sorted_coeffs[..., :M, :]

def _basis_coefficients_annulus(
    primitive_lattice_vectors: LatticeVectors,
    approximate_num_terms: int,
    kmin_factor: float | None = None,
    kmin_abs: float | None = None,
    keep_zero_order: bool = True,
) -> onp.ndarray:
    """Computes the basis coefficients of lattice vectors lying within an annulus.

    The coefficients generate a set of lattice vectors from a basis using an annular
    truncation, so that all lattice vectors lying within an annular region with area
    given by `num * u ⨉ v` are included, where `⨉` is the cross product. The size of
    the set will generally be close to `approximate_num_terms`.

    Args:
        primitive_lattice_vectors: Primitive vectors for the reciprocal-space lattice.
        approximate_num_terms: The approximate number of terms in the expansion. To
            maintain a symmetric expansion, the total number of terms may differ from
            this value.
        kmin_factor: The inner radius of the annulus is given by `kmin_factor` times the
            outer radius. Must be between 0 and 1. Outer radius is determined by `approximate_num_terms` as described above.
        kmin_abs: The inner radius of the annulus is at least `kmin_abs`. Must be non-negative.
        keep_zero_order: Whether to include the zero-order term (i.e. the term with zero coefficients) in the expansion, even if it lies outside the annulus.


    Returns:
        The coefficients, with shape `(num_vectors, 2)`. The final axis gives the
        coefficient for the first and second vector in the basis.
    """
    
    if kmin_factor is not None and (kmin_factor < 0 or kmin_factor > 1):
        raise ValueError(f"`kmin_factor` must be between 0 and 1, but got {kmin_factor}.")
    if kmin_abs is not None and kmin_abs < 0:
        raise ValueError(f"`kmin_abs` must be non-negative, but got {kmin_abs}.")
    if kmin_factor is not None and kmin_abs is not None:
        raise ValueError("Cannot specify both `kmin_factor` and `kmin_abs`.")
    
    if kmin_factor is None and kmin_abs is None:
        return _basis_coefficients_circular(primitive_lattice_vectors, approximate_num_terms)

    FBZ_area = onp.abs(
        _cross_product(primitive_lattice_vectors.u, primitive_lattice_vectors.v)
    )
    if kmin_factor is not None:
        if kmin_factor == 0:
            return _basis_coefficients_circular(primitive_lattice_vectors, approximate_num_terms)
        max_magnitude = onp.sqrt(
            approximate_num_terms
            * FBZ_area / onp.pi / (1 - kmin_factor**2)
        )
        min_magnitude = kmin_factor * max_magnitude
    else:  # kmin_abs is not None
        min_magnitude = kmin_abs
        max_magnitude = onp.sqrt(
            approximate_num_terms
            * FBZ_area
            / onp.pi + min_magnitude**2
        )
    
    # Generate candidate coefficients. These will include more coefficients than needed;
    # subsequently we will filter based on magnitude.
    Nmin = int(onp.ceil(onp.pi *min_magnitude**2 / FBZ_area))
    Nmax = Nmin + approximate_num_terms 
    exp_max = _basis_coefficients_circular(primitive_lattice_vectors, Nmax)
    exp_min = _basis_coefficients_circular(primitive_lattice_vectors, Nmin)
    
    # Find rows in exp_max that are not in exp_min
    exp_min_set = set(map(tuple, exp_min))
    mask = onp.array([tuple(row) not in exp_min_set for row in exp_max])
    exp = exp_max[mask]

    if keep_zero_order and not onp.any(onp.all(exp == onp.array([0, 0]), axis=1)):
        exp = onp.vstack([exp, onp.array([[0, 0]])])

    vectors = (
        exp[..., 0, onp.newaxis] * primitive_lattice_vectors.u[..., onp.newaxis, :]
        + exp[..., 1, onp.newaxis] * primitive_lattice_vectors.v[..., onp.newaxis, :]
    )
    magnitude = onp.linalg.norm(vectors, axis=-1)
    order = onp.argsort(magnitude)
    exp = exp[..., order, :]

    if keep_zero_order:
        zero_mask = onp.all(exp == onp.array([0, 0]), axis=1)
        if onp.any(zero_mask):
            exp = onp.vstack([exp[zero_mask], exp[~zero_mask]])

    return exp



            
# -----------------------------------------------------------------------------
# Register custom objects in this module with jax to enable `jit`.
# -----------------------------------------------------------------------------


jax.tree_util.register_pytree_node(
    LatticeVectors,
    lambda lv: ((lv.u, lv.v), None),
    lambda _, uv: LatticeVectors(*uv),
)


jax.tree_util.register_pytree_node(
    Truncation,
    lambda x: ((), x.value),
    lambda value, _: Truncation(value),
)


def unflatten_expansion(
    aux_tuple: Tuple["_HashableArray"],
    leaves: Tuple,
) -> Expansion:
    """Unflattens the `Expansion`."""
    del leaves
    wrapped: "_HashableArray" = aux_tuple[0]
    coeffs: onp.ndarray = wrapped.value
    return Expansion(basis_coefficients=coeffs)


jax.tree_util.register_pytree_node(
    Expansion,
    lambda e: ((), (_HashableArray(e.basis_coefficients),)),
    unflatten_func=unflatten_expansion,
)


class _HashableArray:
    """Hashable wrapper for numpy arrays."""

    def __init__(self, value: onp.ndarray):
        self.value: onp.ndarray = value

    def __hash__(self):
        return hash((self.value.shape, self.value.dtype, self.value.tobytes()))

    def __eq__(self, other):
        if not isinstance(other, _HashableArray):
            return False
        return (
            self.value.shape == other.value.shape
            and self.value.dtype == other.value.dtype
            and onp.all(self.value == other.value)
        )
