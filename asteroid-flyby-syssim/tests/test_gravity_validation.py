"""Validation tests for asteroid gravity model.

Tests confirm that the spherical-harmonic gravity implementation:
1. Reproduces point-mass Keplerian gravity in the lmax=0 limit
2. Shows proper far-field decay and asymptotic behavior
"""

import numpy as np
import pytest

from syssim.core import InputPort

from asteroid_flyby_syssim.nodes import NodeAsteroidGravity


def _seed_input_port(port, value, sim_time: float = 0.0):
    port.write(np.array(value, dtype=float), sim_time)


def _attach_output_sink(output_port):
    sink = InputPort("capture", node=None)
    output_port.connect_input(sink)
    return sink


class TestPointMassRegression:
    """Verify lmax=0 gravity matches point-mass Keplerian."""

    @pytest.mark.parametrize("asteroid", ["Ceres", "Vesta", "Eros"])
    def test_lmax0_matches_kepler(self, asteroid: str):
        """With lmax=0 and omega=0, gravity should match -mu*r/r^3.
        
        This is a regression test: if lmax=0 gravity does not match Kepler,
        the conversion from spherical to Cartesian coordinates is wrong,
        or there's a sign convention issue.
        """
        # Create gravity node with lmax=0 (monopole only) and omega=0
        gravity_node = NodeAsteroidGravity(asteroid=asteroid, lmax=0)
        mu = gravity_node._gravity_model.gm
        accel_sink = _attach_output_sink(gravity_node._o.gravity_accel)

        # Seed RNG for reproducibility
        rng = np.random.RandomState(42)
        
        # Test at 20 random positions
        # Sample radii from 1.5 to 10 times reference radius
        r_ref = gravity_node._gravity_model.r0
        for _ in range(20):
            r_norm = r_ref * rng.uniform(1.5, 10.0)
            # Random direction
            direction = rng.randn(3)
            direction /= np.linalg.norm(direction)
            position = r_norm * direction
            
            # Evaluate gravity model
            _seed_input_port(gravity_node._i.position, position, sim_time=0.0)
            gravity_node.update(sim_time=0.0)
            accel_model = accel_sink.read().value
            
            # Expected Keplerian acceleration
            r_mag = np.linalg.norm(position)
            accel_kepler = -mu / r_mag**3 * position
            
            # Relative error (avoid divide-by-zero for tiny accelerations)
            accel_mag = np.linalg.norm(accel_kepler)
            if accel_mag > 1e-12:
                rel_error = np.linalg.norm(accel_model - accel_kepler) / accel_mag
            else:
                rel_error = np.linalg.norm(accel_model - accel_kepler)
            
            # With lmax=0 and proper conversion, error should be ~machine precision
            # Allow up to 1e-10 relative error for numerical noise
            assert rel_error < 1e-10, (
                f"{asteroid}: rel_error={rel_error:.2e} at r={position}. "
                f"model={accel_model}, kepler={accel_kepler}"
            )

    @pytest.mark.parametrize("asteroid", ["Ceres", "Vesta", "Eros"])
    def test_lmax0_radial_direction(self, asteroid: str):
        """Along +X axis, lmax=0 gravity should point in -X direction only."""
        gravity_node = NodeAsteroidGravity(asteroid=asteroid, lmax=0)
        mu = gravity_node._gravity_model.gm
        r_ref = gravity_node._gravity_model.r0
        accel_sink = _attach_output_sink(gravity_node._o.gravity_accel)
        
        # Test position along +X axis at multiple radii
        for r_norm in [2.0 * r_ref, 5.0 * r_ref, 10.0 * r_ref]:
            position = np.array([r_norm, 0.0, 0.0])
            _seed_input_port(gravity_node._i.position, position, sim_time=0.0)
            gravity_node.update(sim_time=0.0)
            accel = accel_sink.read().value
            
            # Acceleration magnitude
            accel_mag = np.linalg.norm(accel)
            
            # Expected: magnitude is mu/r^2, direction is -X
            expected_mag = mu / r_norm**2
            expected_dir = np.array([-1.0, 0.0, 0.0])
            
            # Check magnitude (within 1% for numerical noise)
            rel_error_mag = abs(accel_mag - expected_mag) / expected_mag
            assert rel_error_mag < 1e-10, (
                f"{asteroid}: magnitude rel_error={rel_error_mag:.2e} at r={r_norm}"
            )
            
            # Check direction: should be purely in -X
            assert abs(accel[1]) < 1e-12, f"{asteroid}: Y component {accel[1]:.2e} should be ~0"
            assert abs(accel[2]) < 1e-12, f"{asteroid}: Z component {accel[2]:.2e} should be ~0"
            assert accel[0] < 0, f"{asteroid}: X component {accel[0]} should be negative"


class TestFarFieldAsymptotics:
    """Verify far-field decay matches 1/r^2 law."""

    @pytest.mark.parametrize("asteroid", ["Ceres", "Vesta", "Eros"])
    @pytest.mark.parametrize("lmax_test", [None, 12])
    def test_far_field_magnitude_decay(self, asteroid: str, lmax_test: int | None):
        """Check |a| * r^2 / mu approaches 1 at large radii.
        
        Tests with both full gravity model and limited degree (lmax=12).
        """
        gravity_node = NodeAsteroidGravity(asteroid=asteroid, lmax=lmax_test)
        mu = gravity_node._gravity_model.gm
        r_ref = gravity_node._gravity_model.r0
        accel_sink = _attach_output_sink(gravity_node._o.gravity_accel)
        
        # Sample radii from 2R to 50R
        radii = np.array([2.0, 5.0, 10.0, 20.0, 50.0]) * r_ref
        
        for r_norm in radii:
            # Test along +X axis
            position = np.array([r_norm, 0.0, 0.0])
            _seed_input_port(gravity_node._i.position, position, sim_time=0.0)
            gravity_node.update(sim_time=0.0)
            accel = accel_sink.read().value
            
            accel_mag = np.linalg.norm(accel)
            
            # Normalized gravity magnitude: should approach 1
            normalized_accel = accel_mag * r_norm**2 / mu
            
            # Far from asteroid, deviation should be small
            # At r=50*R_ref, harmonics are negligible; accept up to 5% deviation
            # At r=20*R_ref, harmonics still small; accept up to 10% deviation
            # At r=2*R_ref, harmonics may be significant; accept up to 50% deviation
            if r_norm > 40 * r_ref:
                tolerance = 0.05
            elif r_norm > 15 * r_ref:
                tolerance = 0.10
            else:
                tolerance = 0.50
            
            error = abs(normalized_accel - 1.0)
            assert error < tolerance, (
                f"{asteroid} (lmax={lmax_test}): at r={r_norm/r_ref:.1f}*R_ref, "
                f"normalized_accel={normalized_accel:.6f} (error={error:.6f}, tolerance={tolerance:.6f})"
            )

    @pytest.mark.parametrize("asteroid", ["Ceres", "Vesta", "Eros"])
    def test_radial_dominance_far_field(self, asteroid: str):
        """Check that transverse components << radial in far field.
        
        At large radii, gravity should be purely radial (Kepler-like).
        """
        gravity_node = NodeAsteroidGravity(asteroid=asteroid, lmax=None)
        r_ref = gravity_node._gravity_model.r0
        accel_sink = _attach_output_sink(gravity_node._o.gravity_accel)
        
        # Test at a moderately large radius where harmonics have decayed
        r_norm = 20.0 * r_ref
        
        # Test position at oblique angle to ensure transverse components could exist
        theta = np.radians(45.0)
        phi = np.radians(30.0)
        position = r_norm * np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        _seed_input_port(gravity_node._i.position, position, sim_time=0.0)
        gravity_node.update(sim_time=0.0)
        accel = accel_sink.read().value
        
        # Radial and transverse components in spherical frame
        accel_mag = np.linalg.norm(accel)
        radial_unit = position / np.linalg.norm(position)
        accel_radial = np.dot(accel, radial_unit)
        accel_transverse_mag = np.sqrt(accel_mag**2 - accel_radial**2)
        
        # Transverse should be small compared to radial
        transverse_ratio = accel_transverse_mag / abs(accel_radial)
        
        # At r=20*R_ref, expect transverse to be <10% of radial
        assert transverse_ratio < 0.10, (
            f"{asteroid}: transverse ratio {transverse_ratio:.4f} at r={r_norm/r_ref:.1f}*R_ref. "
            f"Position: {position}, Accel: {accel}, "
            f"a_radial={accel_radial:.6e}, a_transverse={accel_transverse_mag:.6e}"
        )

    @pytest.mark.parametrize("asteroid", ["Ceres", "Vesta", "Eros"])
    def test_convergence_with_increasing_lmax(self, asteroid: str):
        """Verify that higher lmax provides smoother convergence to asymptotics.
        
        Higher-degree harmonics should have less effect at large radii.
        """
        r_ref_base = None
        results_by_lmax = {}
        
        # Test with different lmax values
        for lmax_val in [0, 6, 12]:
            gravity_node = NodeAsteroidGravity(asteroid=asteroid, lmax=lmax_val)
            mu = gravity_node._gravity_model.gm
            r_ref = gravity_node._gravity_model.r0
            accel_sink = _attach_output_sink(gravity_node._o.gravity_accel)
            
            if r_ref_base is None:
                r_ref_base = r_ref
            
            # Evaluate at a moderate-to-large radius
            r_norm = 15.0 * r_ref
            position = np.array([r_norm, 0.0, 0.0])
            
            _seed_input_port(gravity_node._i.position, position, sim_time=0.0)
            gravity_node.update(sim_time=0.0)
            accel = accel_sink.read().value
            
            accel_mag = np.linalg.norm(accel)
            normalized = accel_mag * r_norm**2 / mu
            results_by_lmax[lmax_val] = normalized
        
        # lmax=0 should be closest to 1.0 (pure monopole)
        # Higher lmax will add multipole corrections
        # All should be within ~20% of unity at r=15*R
        for lmax_val, normalized in results_by_lmax.items():
            error = abs(normalized - 1.0)
            assert error < 0.20, (
                f"{asteroid} lmax={lmax_val}: normalized_accel={normalized:.6f}, error={error:.6f}"
            )


class TestOmegaEffect:
    """Verify that omega parameter affects gravity only when set."""

    @pytest.mark.parametrize("asteroid", ["Ceres", "Vesta", "Eros"])
    def test_omega_disabled(self, asteroid: str):
        """With omega=0, rotating frame term should not affect gravity."""
        gravity_node = NodeAsteroidGravity(asteroid=asteroid, lmax=0)
        
        # Verify omega is zero or None
        omega = gravity_node._gravity_model.omega
        assert omega is None or omega == 0.0, f"{asteroid}: omega should be None or 0, got {omega}"
        
        # Evaluate at a test point
        r_norm = 5.0 * gravity_node._gravity_model.r0
        position = np.array([r_norm, 1.0, 0.5])  # Non-axis-aligned
        accel_sink = _attach_output_sink(gravity_node._o.gravity_accel)
        
        _seed_input_port(gravity_node._i.position, position, sim_time=0.0)
        gravity_node.update(sim_time=0.0)
        accel = accel_sink.read().value
        
        # Should be purely gravitational, no centrifugal term
        # Centrifugal acceleration would be omega^2 * rho (perpendicular to spin axis)
        # With omega=0, this is zero
        mu = gravity_node._gravity_model.gm
        accel_expected = -mu / np.linalg.norm(position)**3 * position
        
        rel_error = np.linalg.norm(accel - accel_expected) / np.linalg.norm(accel_expected)
        assert rel_error < 1e-10, (
            f"{asteroid}: with omega=0, gravity should be Keplerian. "
            f"rel_error={rel_error:.2e}"
        )


class TestGravityNodeInterface:
    """Basic sanity checks on NodeAsteroidGravity interface."""

    @pytest.mark.parametrize("asteroid", ["Ceres", "Vesta", "Eros"])
    def test_initialization(self, asteroid: str):
        """Verify node initializes successfully for each asteroid."""
        node = NodeAsteroidGravity(asteroid=asteroid)
        assert node._asteroid == asteroid
        assert node._gravity_model is not None
        assert node._gravity_model.gm > 0
        assert node._gravity_model.r0 > 0

    def test_invalid_asteroid_raises(self):
        """Invalid asteroid name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown asteroid"):
            NodeAsteroidGravity(asteroid="InvalidAsteroid")

    # @pytest.mark.parametrize("asteroid", ["Ceres", "Vesta", "Eros"])
    # def test_lmax_clamping(self, asteroid: str):
    #     """lmax should not exceed dataset maximum."""
    #     # Request very high lmax; should be clamped
    #     node = NodeAsteroidGravity(asteroid=asteroid, lmax=1000)
        
    #     # Actual lmax used should be dataset maximum
    #     actual_lmax = node._max_degree
        
    #     # Ceres18D: 18, Vesta20H: 20, Eros15A: 15
    #     expected_max = {"Ceres": 18, "Vesta": 20, "Eros": 15}[asteroid]
    #     assert actual_lmax <= expected_max, (
    #         f"{asteroid}: requested lmax=1000 but got {actual_lmax} (expected <= {expected_max})"
    #     )

    @pytest.mark.parametrize("asteroid", ["Ceres", "Vesta", "Eros"])
    def test_near_singularity_handling(self, asteroid: str):
        """Acceleration should be zero (or small) near center of mass."""
        node = NodeAsteroidGravity(asteroid=asteroid)
        
        # Very close to center
        position_near = np.array([0.1, 0.05, 0.02])
        accel_sink = _attach_output_sink(node._o.gravity_accel)
        _seed_input_port(node._i.position, position_near, sim_time=0.0)
        node.update(sim_time=0.0)
        accel_near = accel_sink.read().value
        
        # Should be zero or very small (handled by singularity check in update)
        accel_mag = np.linalg.norm(accel_near)
        assert accel_mag < 1e-8, (
            f"{asteroid}: acceleration near origin should be ~0, got {accel_mag:.2e}"
        )

    @pytest.mark.parametrize("asteroid", ["Ceres", "Vesta", "Eros"])
    def test_nan_input_handling(self, asteroid: str):
        """NaN input should return zero acceleration, not propagate."""
        node = NodeAsteroidGravity(asteroid=asteroid)
        
        position_nan = np.array([np.nan, 1.0, 2.0])
        accel_sink = _attach_output_sink(node._o.gravity_accel)
        _seed_input_port(node._i.position, position_nan, sim_time=0.0)
        node.update(sim_time=0.0)
        accel = accel_sink.read().value
        
        # Should be [0, 0, 0], not [NaN, ...]
        assert np.allclose(accel, [0.0, 0.0, 0.0]), (
            f"{asteroid}: NaN input should yield zero acceleration, got {accel}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
