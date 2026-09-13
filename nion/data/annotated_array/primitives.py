"""Annotated-array processing primitives.

This module is the home for low-level processing operations on annotated arrays.
Operation implementations can be added here and re-exported from
`nion.data.annotated_array` as they are introduced.
"""

from __future__ import annotations

import math
import typing

import numpy
import numpy.typing
import scipy.fft
import scipy.signal.windows

from nion.data.annotated_array._implementation import (
    AffineCalibration,
    AnnotatedArray,
    ArrayDescriptor,
    AxisGroup,
    Calibration,
    CoordinateCalibration,
    ValueType,
)


# ---------------------------------------------------------------------------
# Internal calibration helpers
# ---------------------------------------------------------------------------

def _spatial_to_frequency_axis_group(axis_group: AxisGroup) -> AxisGroup:
    """Return a new AxisGroup with all calibrations transformed to frequency domain.

    For each axis of size *N* with spatial calibration ``(scale=s, unit=u)``
    the frequency calibration is::

        scale_freq   = 1 / (s * N)
        offset_freq  = (-0.5 - N // 2) / (s * N)   # DC at centre after fftshift
        unit_freq    = "1/" + u                      # reciprocal unit

    All calibration keys are preserved (same keys in, same keys out), each
    transformed independently under its original key; the primary key is unchanged.
    """
    def transform_coord_calibration(coord_cal: CoordinateCalibration) -> CoordinateCalibration:
        freq_calibrations: list[AffineCalibration] = []
        for axis_index, axis in enumerate(axis_group.axes):
            n = axis.size
            spatial_cal = coord_cal.calibrations[axis_index]
            s = spatial_cal.scale if isinstance(spatial_cal, AffineCalibration) else 1.0
            u = spatial_cal.unit  if isinstance(spatial_cal, AffineCalibration) else ""
            freq_calibrations.append(AffineCalibration(
                scale=1.0 / (s * n),
                offset=(-0.5 - n // 2) / (s * n),
                unit=("1/" + u) if u else "",
            ))
        return CoordinateCalibration(calibrations=tuple(freq_calibrations))

    new_calibrations = {
        key: transform_coord_calibration(coord_cal)
        for key, coord_cal in axis_group.coordinate_calibrations.items()
    }

    return AxisGroup(
        axes=axis_group.axes,
        coordinate_system_id=axis_group.coordinate_system_id,
        coordinate_calibrations=new_calibrations,
        primary_calibration_key=axis_group.primary_calibration_key,
    )


def _frequency_to_spatial_axis_group(axis_group: AxisGroup) -> AxisGroup:
    """Return a new AxisGroup with all calibrations transformed back to spatial domain.

    For each frequency calibration ``(scale=s_freq, unit=u_freq)``::

        scale_spatial   = 1 / (s_freq * N)
        offset_spatial  = 0
        unit_spatial    = u_freq[2:] if u_freq.startswith("1/") else ""

    All calibration keys are preserved (same keys in, same keys out); the primary
    key is unchanged.
    """
    def transform_coord_calibration(coord_cal: CoordinateCalibration) -> CoordinateCalibration:
        spatial_calibrations: list[AffineCalibration] = []
        for axis_index, axis in enumerate(axis_group.axes):
            n = axis.size
            freq_cal = coord_cal.calibrations[axis_index]
            if isinstance(freq_cal, AffineCalibration) and freq_cal.scale != 0.0:
                s_freq = freq_cal.scale
                u_freq = freq_cal.unit
                s_spatial = 1.0 / (s_freq * n)
                u_spatial = u_freq[2:] if u_freq.startswith("1/") else ""
            else:
                s_spatial, u_spatial = 1.0, ""
            spatial_calibrations.append(AffineCalibration(scale=s_spatial, offset=0.0, unit=u_spatial))
        return CoordinateCalibration(calibrations=tuple(spatial_calibrations))

    new_calibrations = {
        key: transform_coord_calibration(coord_cal)
        for key, coord_cal in axis_group.coordinate_calibrations.items()
    }

    return AxisGroup(
        axes=axis_group.axes,
        coordinate_system_id=axis_group.coordinate_system_id,
        coordinate_calibrations=new_calibrations,
        primary_calibration_key=axis_group.primary_calibration_key,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fft(array: AnnotatedArray) -> AnnotatedArray:
    """Compute the forward FFT of an :class:`AnnotatedArray`.

    The transform is applied to the array's only axis group.

    Energy normalisation
    ~~~~~~~~~~~~~~~~~~~~
    The scaling factor ``1 / sqrt(N)`` (1-D) or ``1 / sqrt(N * M)`` (2-D) is
    applied so that the RMS value is preserved:

    .. code-block:: python

        numpy.sqrt(numpy.mean(numpy.abs(data)**2))
        == numpy.sqrt(numpy.mean(numpy.abs(fft(xdata).data)**2))

    DC at centre
    ~~~~~~~~~~~~
    The result is passed through :func:`scipy.fft.fftshift` along the signal axes
    so that the zero-frequency component lies at the array centre.

    Calibration
    ~~~~~~~~~~~
    All calibrations in the signal :class:`AxisGroup` are transformed to the
    frequency domain; keys (including the primary key) are preserved. For example,
    an input with calibrations keyed ``"spatial"`` and ``"angular"`` produces
    output calibrations under the same two keys, with scales/offsets/units
    transformed to frequency.

    Args:
        array: Input :class:`AnnotatedArray` with a scalar or complex datum.
               RGB and RGBA value types are not supported.

    Returns:
        :class:`AnnotatedArray` with complex datum (``complex128``) and
        frequency-domain calibrations on the signal :class:`AxisGroup`.

    Raises:
        ValueError: If there is not exactly one axis group, if that group rank
                    is not 1 or 2, or if the value type is not ``SCALAR`` or
                    ``COMPLEX``.
    """
    axis_groups = array.descriptor.axis_groups
    if len(axis_groups) != 1:
        raise ValueError(f"fft: expected exactly one axis group, got {len(axis_groups)}")

    signal_group = axis_groups[-1]
    rank = signal_group.rank

    if rank not in (1, 2):
        raise ValueError(f"fft: signal rank must be 1 or 2, got {rank}")

    value_type = array.descriptor.value_type
    if value_type not in (ValueType.SCALAR, ValueType.COMPLEX):
        raise ValueError(
            f"fft: unsupported value type {value_type!r}; "
            "only SCALAR and COMPLEX are supported"
        )

    data = numpy.asarray(array.data)
    signal_shape = signal_group.shape          # e.g. (N,) or (N, M)
    signal_axes = tuple(range(-rank, 0))      # e.g. (-1,) or (-2, -1)

    if rank == 1:
        n = signal_shape[0]
        scaling = 1.0 / math.sqrt(n)
        result_data: numpy.typing.NDArray[numpy.complexfloating[typing.Any, typing.Any]] = (
            scipy.fft.fftshift(scipy.fft.fft(data, axis=-1) * scaling, axes=signal_axes)
        )
    else:
        n, m = signal_shape
        scaling = 1.0 / math.sqrt(n * m)
        result_data = scipy.fft.fftshift(
            scipy.fft.fft2(data, axes=signal_axes) * scaling,
            axes=signal_axes,
        )

    new_signal_group = _spatial_to_frequency_axis_group(signal_group)
    new_axis_groups = axis_groups[:-1] + (new_signal_group,)
    new_descriptor = ArrayDescriptor(
        axis_groups=new_axis_groups,
        intensity_calibrations=array.descriptor.intensity_calibrations,
        value_type=ValueType.COMPLEX,
    )
    return AnnotatedArray(data=result_data, descriptor=new_descriptor, metadata=array.metadata)


def ifft(array: AnnotatedArray) -> AnnotatedArray:
    """Compute the inverse FFT of an :class:`AnnotatedArray`.

    The transform is applied to the array's only axis group.

    Energy normalisation
    ~~~~~~~~~~~~~~~~~~~~
    The inverse scaling factor ``sqrt(N)`` (1-D) or ``sqrt(N * M)`` (2-D) is
    applied to be the exact inverse of :func:`fft`.

    DC at centre
    ~~~~~~~~~~~~
    The input is assumed to have its DC component at the array centre
    (produced by :func:`fft`); :func:`scipy.fft.ifftshift` is applied along
    the signal axes before the inverse transform.

    Calibration round-trip
    ~~~~~~~~~~~~~~~~~~~~~~
    All calibrations in the signal :class:`AxisGroup` are transformed back to the
    spatial domain; keys (including the primary key) are preserved. For example,
    a frequency-domain array with calibrations keyed ``"spatial"`` and ``"angular"``
    produces a result with the same two keys, now holding spatial scales/offsets/units.

    Args:
        array: :class:`AnnotatedArray` with a complex datum in frequency
               space (DC at centre).

    Returns:
        :class:`AnnotatedArray` with complex datum and spatial-domain
        calibrations on the signal :class:`AxisGroup`.

    Raises:
        ValueError: If there is not exactly one axis group, or if that group
                    rank is not 1 or 2.
    """
    axis_groups = array.descriptor.axis_groups
    if len(axis_groups) != 1:
        raise ValueError(f"ifft: expected exactly one axis group, got {len(axis_groups)}")

    signal_group = axis_groups[-1]
    rank = signal_group.rank

    if rank not in (1, 2):
        raise ValueError(f"ifft: signal rank must be 1 or 2, got {rank}")

    data = numpy.asarray(array.data)
    signal_shape = signal_group.shape
    signal_axes = tuple(range(-rank, 0))

    if rank == 1:
        n = signal_shape[0]
        scaling = math.sqrt(n)
        result_data = scipy.fft.ifft(
            scipy.fft.ifftshift(data, axes=signal_axes) * scaling,
            axis=-1,
        )
    else:
        n, m = signal_shape
        scaling = math.sqrt(n * m)
        result_data = scipy.fft.ifft2(
            scipy.fft.ifftshift(data, axes=signal_axes) * scaling,
            axes=signal_axes,
        )

    new_signal_group = _frequency_to_spatial_axis_group(signal_group)
    new_axis_groups = axis_groups[:-1] + (new_signal_group,)
    new_descriptor = ArrayDescriptor(
        axis_groups=new_axis_groups,
        intensity_calibrations=array.descriptor.intensity_calibrations,
        value_type=ValueType.COMPLEX,
    )
    return AnnotatedArray(data=result_data, descriptor=new_descriptor, metadata=array.metadata)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def _validate_window_axis_group(name: str, axis_group: AxisGroup) -> int:
    """Validate that ``axis_group`` is a suitable shape for a window generator.

    Returns the axis group's rank.
    """
    rank = axis_group.rank
    if rank not in (1, 2):
        raise ValueError(f"{name}: axis group rank must be 1 or 2, got {rank}")
    return rank


def _window_array(axis_group: AxisGroup, data: numpy.typing.NDArray[numpy.float64]) -> AnnotatedArray:
    """Wrap a real-valued ``data`` array as a standalone window :class:`AnnotatedArray`.

    The window carries ``axis_group`` unchanged, so its coordinate calibration matches
    whatever data it will later be multiplied into (see `Windowing Generators` in the
    processing-operations design document). Window values are dimensionless weights, so
    no intensity calibration is attached.
    """
    descriptor = ArrayDescriptor(axis_groups=(axis_group,), value_type=ValueType.SCALAR)
    return AnnotatedArray(data=data, descriptor=descriptor)


def _calibrated_sigma_to_pixels(name: str, axis_group: AxisGroup, sigma: float) -> tuple[float, ...]:
    """Convert a physical-unit ``sigma`` to one pixel-unit sigma per axis of ``axis_group``.

    Requires a primary coordinate calibration whose axes share one non-empty unit
    (a single physical value can't be meaningfully split across differing units).
    Each axis's factor comes from ``calibration.to_index(sigma) - calibration.to_index(0.0)``,
    which cancels the offset without assuming a concrete implementation such as
    :class:`AffineCalibration`; this is exact only for an affine (constant-scale)
    calibration.

    A non-positive ``sigma`` or a non-positive converted result (e.g. a degenerate
    zero-scale calibration) clamps to ``1.0`` pixel, matching the non-calibrated clamp.
    """
    if axis_group.primary_calibration_key is None:
        raise ValueError(f"{name}: calibrated sigma requires the axis group to have a coordinate calibration")

    calibrations: list[Calibration] = [axis_group.get_calibration(axis_index) for axis_index in range(axis_group.rank)]
    units = {calibration.unit for calibration in calibrations}
    if len(units) > 1:
        raise ValueError(f"{name}: calibrated sigma requires all axes to share the same unit, got {sorted(units)!r}")
    if not next(iter(units)):
        raise ValueError(f"{name}: calibrated sigma requires a non-empty calibration unit")

    if sigma <= 0.0:
        return (1.0,) * axis_group.rank

    pixel_sigmas: list[float] = []
    for calibration in calibrations:
        pixel_sigma = abs(calibration.to_index(sigma) - calibration.to_index(0.0))
        pixel_sigmas.append(pixel_sigma if pixel_sigma > 0.0 else 1.0)
    return tuple(pixel_sigmas)


def gaussian_window(axis_group: AxisGroup, sigma: float, *, calibrated: bool = False) -> AnnotatedArray:
    """Generate a Gaussian window matching the shape of ``axis_group``.

    Returns the window array standalone (see `Windowing Generators` in the
    processing-operations design document); combining it with a target array,
    typically by elementwise multiplication, is up to the caller.

    Relative sigma (``calibrated=False``, the default)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``sigma`` is a relative standard deviation in ``[0.0, 1.0]``; the absolute
    standard deviation used to build the window is ``sigma * min(shape)``, where
    ``shape`` is ``axis_group``'s shape (in pixels). A non-positive ``sigma`` is
    clamped to ``1.0`` pixel to avoid a divide-by-zero window. This mode is
    calibration-agnostic: it depends only on the pixel shape.

    Calibrated sigma (``calibrated=True``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``sigma`` is a physical-unit standard deviation (e.g. nanometers), converted
    to pixels independently per axis since axes may have different scales. This
    generally yields an anisotropic (elliptical) 2D Gaussian rather than the
    circular window used in relative mode. Requires a primary coordinate
    calibration whose axes share one non-empty unit.

    2D construction
    ~~~~~~~~~~~~~~~
    For a 2D axis group, the window is
    ``exp(-0.5 * (y**2 / sigma_y**2 + x**2 / sigma_x**2))``, which reduces to the
    circularly symmetric radial Gaussian when ``sigma_y == sigma_x`` (always true in
    relative mode; not guaranteed in calibrated mode).

    Args:
        axis_group: The 1D or 2D :class:`AxisGroup` whose shape (and, in calibrated
                    mode, calibration) the generated window matches.
        sigma: Relative standard deviation in ``[0.0, 1.0]`` when ``calibrated`` is
               ``False``; physical-unit standard deviation when ``calibrated`` is
               ``True``.
        calibrated: Whether ``sigma`` is expressed in ``axis_group``'s coordinate
                    calibration unit rather than as a relative fraction.

    Returns:
        :class:`AnnotatedArray` with a real scalar datum of ``axis_group``'s shape,
        carrying ``axis_group`` as its only axis group.

    Raises:
        ValueError: If ``axis_group``'s rank is not 1 or 2, or if ``calibrated`` is
                    ``True`` and ``axis_group`` has no calibration or has axes with
                    differing or empty units.
    """
    rank = _validate_window_axis_group("gaussian_window", axis_group)
    shape = axis_group.shape

    if calibrated:
        pixel_sigmas = _calibrated_sigma_to_pixels("gaussian_window", axis_group, sigma)
    else:
        absolute_sigma = sigma * min(shape)
        absolute_sigma = absolute_sigma if absolute_sigma > 0.0 else 1.0
        pixel_sigmas = (absolute_sigma,) * rank

    if rank == 1:
        data = scipy.signal.windows.gaussian(shape[0], std=pixel_sigmas[0])
    else:
        h, w = shape
        sigma_y, sigma_x = pixel_sigmas
        y, x = numpy.meshgrid(
            numpy.arange(0, h) - (h - 1) / 2,
            numpy.arange(0, w) - (w - 1) / 2,
            indexing="ij",
        )
        data = numpy.exp(-0.5 * ((y * y) / (sigma_y * sigma_y) + (x * x) / (sigma_x * sigma_x)))

    return _window_array(axis_group, data)


def hamming_window(axis_group: AxisGroup) -> AnnotatedArray:
    """Generate a Hamming window matching the shape of ``axis_group``.

    Returns the window array standalone (see `Windowing Generators` in the
    processing-operations design document); combining it with a target array,
    typically by elementwise multiplication, is up to the caller.

    2D construction
    ~~~~~~~~~~~~~~~
    For a 2D axis group, the window is the separable outer product of two 1D Hamming
    windows, one per axis.

    Args:
        axis_group: The 1D or 2D :class:`AxisGroup` whose shape the generated window
                    matches.

    Returns:
        :class:`AnnotatedArray` with a real scalar datum of ``axis_group``'s shape,
        carrying ``axis_group`` as its only axis group.

    Raises:
        ValueError: If ``axis_group``'s rank is not 1 or 2.
    """
    rank = _validate_window_axis_group("hamming_window", axis_group)
    shape = axis_group.shape

    if rank == 1:
        data = scipy.signal.windows.hamming(shape[0])
    else:
        h, w = shape
        w0 = numpy.reshape(scipy.signal.windows.hamming(w), (1, w))
        w1 = numpy.reshape(scipy.signal.windows.hamming(h), (h, 1))
        data = w0 * w1

    return _window_array(axis_group, data)


def hann_window(axis_group: AxisGroup) -> AnnotatedArray:
    """Generate a Hann window matching the shape of ``axis_group``.

    Returns the window array standalone (see `Windowing Generators` in the
    processing-operations design document); combining it with a target array,
    typically by elementwise multiplication, is up to the caller.

    2D construction
    ~~~~~~~~~~~~~~~
    For a 2D axis group, the window is the separable outer product of two 1D Hann
    windows, one per axis.

    Args:
        axis_group: The 1D or 2D :class:`AxisGroup` whose shape the generated window
                    matches.

    Returns:
        :class:`AnnotatedArray` with a real scalar datum of ``axis_group``'s shape,
        carrying ``axis_group`` as its only axis group.

    Raises:
        ValueError: If ``axis_group``'s rank is not 1 or 2.
    """
    rank = _validate_window_axis_group("hann_window", axis_group)
    shape = axis_group.shape

    if rank == 1:
        data = scipy.signal.windows.hann(shape[0])
    else:
        h, w = shape
        w0 = numpy.reshape(scipy.signal.windows.hann(w), (1, w))
        w1 = numpy.reshape(scipy.signal.windows.hann(h), (h, 1))
        data = w0 * w1

    return _window_array(axis_group, data)
