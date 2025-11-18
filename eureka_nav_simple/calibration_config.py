#!/usr/bin/env python3
"""
Calibration Configuration for Distance Measurement
Parameterized lookup table system
"""

import numpy as np
from typing import List, Tuple

# ═══════════════════════════════════════════════════════════════════
# CALIBRATION PARAMETERS (ADJUST THESE)
# ═══════════════════════════════════════════════════════════════════

# Distance measurement range
MIN_DISTANCE_M = 1.0      # Minimum calibration distance (meters)
MAX_DISTANCE_M = 10.0     # Maximum calibration distance (meters)
DISTANCE_STEP_M = 1.0     # Step between calibration points (meters)

# Generate calibration distances
CALIBRATION_DISTANCES = np.arange(MIN_DISTANCE_M, MAX_DISTANCE_M + DISTANCE_STEP_M, DISTANCE_STEP_M)

# ═══════════════════════════════════════════════════════════════════
# CALIBRATION DATA (TO BE FILLED DURING CALIBRATION)
# ═══════════════════════════════════════════════════════════════════

# Format: (distance_meters, pixel_width, pixel_height)
# EXAMPLE DATA - REPLACE WITH YOUR ACTUAL MEASUREMENTS

# ARROW CALIBRATION TABLE
CALIBRATION_TABLE_ARROWS = [
    # Distance(m), Width(px), Height(px)
    (1.0,  138,  104),   # Arrow at 1 meter
    (2.0,   69,   52),   # Arrow at 2 meters
    (3.0,   46,   35),   # Arrow at 3 meters
    (4.0,   35,   26),   # Arrow at 4 meters
    (5.0,   28,   21),   # Arrow at 5 meters
    (6.0,   23,   17),   # Arrow at 6 meters
    (7.0,   20,   15),   # Arrow at 7 meters
    (8.0,   17,   13),   # Arrow at 8 meters
    (9.0,   15,   11),   # Arrow at 9 meters
    (10.0,  14,   10),   # Arrow at 10 meters
]

# CONE CALIBRATION TABLE
# Cones are typically narrower and taller than arrows
CALIBRATION_TABLE_CONES = [
    # Distance(m), Width(px), Height(px)
    (1.0,  80,  160),   # Cone at 1 meter
    (2.0,  40,   80),   # Cone at 2 meters
    (3.0,  27,   53),   # Cone at 3 meters
    (4.0,  20,   40),   # Cone at 4 meters
    (5.0,  16,   32),   # Cone at 5 meters
    (6.0,  13,   27),   # Cone at 6 meters
    (7.0,  11,   23),   # Cone at 7 meters
    (8.0,  10,   20),   # Cone at 8 meters
    (9.0,   9,   18),   # Cone at 9 meters
    (10.0,  8,   16),   # Cone at 10 meters
]

# Default table for backward compatibility (arrows)
CALIBRATION_TABLE = CALIBRATION_TABLE_ARROWS

# Note: These are EXAMPLE values based on theoretical calculation
# YOU MUST REPLACE with actual measured values from your camera!

# ═══════════════════════════════════════════════════════════════════
# ANGLE CALCULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════

# Focal length in pixels (can also be empirically calibrated)
FX_PIX = 920.0  # Current value, can adjust after testing

# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════

# Display pixel measurements during exhibition
SHOW_PIXEL_WIDTH = True      # Show width in pixels
SHOW_PIXEL_HEIGHT = True     # Show height in pixels
SHOW_CALIBRATION_STATUS = True  # Show if using calibrated or extrapolated value

# Text display settings
TEXT_COLOR_CALIBRATED = (0, 255, 0)      # Green for calibrated range
TEXT_COLOR_EXTRAPOLATED = (0, 255, 255)  # Yellow for extrapolated
TEXT_COLOR_OUT_OF_RANGE = (0, 0, 255)    # Red for out of range

# ═══════════════════════════════════════════════════════════════════
# INTERPOLATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_distance_from_pixels(pixel_width: float, pixel_height: float,
                             use_width: bool = True, object_type: str = "arrow") -> Tuple[float, str]:
    """
    Convert pixel measurement to distance using piecewise linear interpolation.

    Args:
        pixel_width: Width of bounding box in pixels
        pixel_height: Height of bounding box in pixels
        use_width: If True, use width for interpolation; else use height
        object_type: Type of object ("arrow" or "cone") to select calibration table

    Returns:
        (distance_meters, status_string)
        status: "calibrated", "extrapolated_near", "extrapolated_far", "out_of_range"
    """
    # Select appropriate calibration table based on object type
    if object_type == "cone":
        calibration_table = CALIBRATION_TABLE_CONES
        # For cones, prefer height over width
        if use_width is True:  # If not explicitly set, use height for cones
            use_width = False
    else:
        calibration_table = CALIBRATION_TABLE_ARROWS

    if not calibration_table:
        # Fallback to theoretical formula if no calibration data
        return get_distance_theoretical(pixel_width), "uncalibrated"

    # Extract calibration data
    distances = np.array([entry[0] for entry in calibration_table])
    widths = np.array([entry[1] for entry in calibration_table])
    heights = np.array([entry[2] for entry in calibration_table])

    # Choose which measurement to use
    pixels = pixel_width if use_width else pixel_height
    calibration_pixels = widths if use_width else heights

    # Check bounds
    min_pixels = calibration_pixels[-1]  # Smallest pixels = farthest distance
    max_pixels = calibration_pixels[0]   # Largest pixels = nearest distance

    if pixels > max_pixels:
        # Closer than minimum calibration distance - extrapolate
        status = "extrapolated_near"
        # Linear extrapolation from first two points
        if len(calibration_pixels) >= 2:
            slope = (distances[1] - distances[0]) / (calibration_pixels[1] - calibration_pixels[0])
            distance = distances[0] + slope * (pixels - calibration_pixels[0])
        else:
            distance = distances[0] * (calibration_pixels[0] / pixels)

    elif pixels < min_pixels:
        # Farther than maximum calibration distance - extrapolate
        status = "extrapolated_far"
        # Linear extrapolation from last two points
        if len(calibration_pixels) >= 2:
            slope = (distances[-1] - distances[-2]) / (calibration_pixels[-1] - calibration_pixels[-2])
            distance = distances[-1] + slope * (pixels - calibration_pixels[-1])
        else:
            distance = distances[-1] * (calibration_pixels[-1] / pixels)

    else:
        # Within calibration range - interpolate
        status = "calibrated"
        # Piecewise linear interpolation
        distance = np.interp(pixels, calibration_pixels[::-1], distances[::-1])

    # Sanity check
    if distance < 0:
        distance = 0.1
        status = "out_of_range"
    elif distance > MAX_DISTANCE_M * 2:  # Allow 2x extrapolation
        distance = MAX_DISTANCE_M * 2
        status = "out_of_range"

    return distance, status


def get_distance_theoretical(pixel_width: float, arrow_width_m: float = 0.30) -> float:
    """
    Fallback: Calculate distance using pinhole camera model.
    Used when no calibration data available.
    """
    if pixel_width <= 0:
        return float('inf')
    return (FX_PIX * arrow_width_m) / pixel_width


def get_angle_from_position(box_center_x: float, image_center_x: float) -> float:
    """
    Calculate horizontal angle from image center.

    Args:
        box_center_x: X coordinate of box center
        image_center_x: X coordinate of image center

    Returns:
        Angle in degrees (positive = right, negative = left)
    """
    pixel_offset = box_center_x - image_center_x
    angle_rad = np.arctan(pixel_offset / FX_PIX)
    return np.degrees(angle_rad)


# ═══════════════════════════════════════════════════════════════════
# CALIBRATION VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_calibration_table():
    """Validate that calibration table is properly formatted and sorted."""
    if not CALIBRATION_TABLE:
        return False, "Calibration table is empty"

    # Check format
    for i, entry in enumerate(CALIBRATION_TABLE):
        if len(entry) != 3:
            return False, f"Entry {i} has wrong format (need 3 values: distance, width, height)"
        if entry[0] <= 0 or entry[1] <= 0 or entry[2] <= 0:
            return False, f"Entry {i} has invalid values (all must be > 0)"

    # Check if sorted by distance
    distances = [entry[0] for entry in CALIBRATION_TABLE]
    if distances != sorted(distances):
        return False, "Calibration table must be sorted by distance (ascending)"

    # Check if widths and heights decrease with distance
    widths = [entry[1] for entry in CALIBRATION_TABLE]
    heights = [entry[2] for entry in CALIBRATION_TABLE]

    if widths != sorted(widths, reverse=True):
        return False, "Warning: Pixel widths should decrease with distance"

    if heights != sorted(heights, reverse=True):
        return False, "Warning: Pixel heights should decrease with distance"

    # Check coverage
    min_dist = distances[0]
    max_dist = distances[-1]

    if min_dist > MIN_DISTANCE_M:
        return False, f"Calibration starts at {min_dist}m but MIN_DISTANCE_M is {MIN_DISTANCE_M}m"

    if max_dist < MAX_DISTANCE_M:
        return False, f"Calibration ends at {max_dist}m but MAX_DISTANCE_M is {MAX_DISTANCE_M}m"

    return True, "Calibration table is valid"


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def print_calibration_info():
    """Print calibration configuration and status."""
    print("="*70)
    print("DISTANCE CALIBRATION CONFIGURATION")
    print("="*70)
    print(f"\nCalibration Range: {MIN_DISTANCE_M}m to {MAX_DISTANCE_M}m")
    print(f"Step Size: {DISTANCE_STEP_M}m")
    print(f"Expected Calibration Points: {len(CALIBRATION_DISTANCES)}")
    print(f"Actual Calibration Points: {len(CALIBRATION_TABLE)}")

    print("\nCalibration Table:")
    print(f"{'Distance(m)':<12} {'Width(px)':<12} {'Height(px)':<12}")
    print("-"*40)
    for dist, width, height in CALIBRATION_TABLE:
        print(f"{dist:<12.1f} {width:<12} {height:<12}")

    valid, message = validate_calibration_table()
    print(f"\nValidation: {'✓ PASS' if valid else '✗ FAIL'}")
    print(f"Message: {message}")

    print(f"\nVisualization Settings:")
    print(f"  Show pixel width: {SHOW_PIXEL_WIDTH}")
    print(f"  Show pixel height: {SHOW_PIXEL_HEIGHT}")
    print(f"  Show calibration status: {SHOW_CALIBRATION_STATUS}")


if __name__ == "__main__":
    print_calibration_info()

    # Test interpolation
    print("\n" + "="*70)
    print("INTERPOLATION TEST")
    print("="*70)

    test_pixels = [150, 100, 70, 50, 35, 25, 15, 10, 5]
    print(f"\n{'Pixels':<10} {'Distance(m)':<15} {'Status':<20}")
    print("-"*50)
    for px in test_pixels:
        dist, status = get_distance_from_pixels(px, px, use_width=True)
        print(f"{px:<10} {dist:<15.2f} {status:<20}")
