"""
THE GLASS PLANT - PET condensation demonstration
Professional Tkinter dashboard rewrite.

Requirements:
    pip install pillow scipy

Keep pet_condensation.png in the same folder as this script.
"""

import math
import os
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk
from scipy.optimize import brentq, least_squares


# =========================================================
# APPLICATION DEFAULTS
# =========================================================

PTA_DEFAULT_KG_H = 33_204.0
HEAT_DEFAULT_KCAL_H = 8_900_000.0
REFLUX_DEFAULT = 1.74

PTA_TO_EG_MASS_RATIO = 0.7444

PTA_CONVERSION = 0.92
EG_CONVERSION = 0.92

COLUMN_TOP_PRESSURE_BAR = 1.00
COLUMN_STAGES = 18
WATER_DISTILLATE_SPEC = 0.995
WATER_BOTTOM_SPEC = 0.01

TRAY_LIQUID_HEIGHT_M = 0.05
TRAY_LIQUID_DENSITY_KG_M3 = 500.0
GRAVITY_M_S2 = 9.81

TRAY_PRESSURE_DROP_BAR = (
    TRAY_LIQUID_DENSITY_KG_M3
    * GRAVITY_M_S2
    * TRAY_LIQUID_HEIGHT_M
    / 100_000.0
)

REFLUX_SUBCOOLING_C = 5.0
EQUILIBRIUM_MODEL = "NRTL"

MW_WATER = 18.0
MW_EG = 62.0

# Heat properties, kcal/(kg degC) unless noted.
PTA_CP = 1.32 / 4.184
EG_CP = 2.24 / 4.184
EG_LATENT = 573.0 / 4.184
WATER_LATENT = 1439.0 / 4.184
PTA_DISSOLUTION_HEAT = 472.0 / 4.184

# NRTL parameters: water (1), ethylene glycol (2)
NRTL_A12 = -1.54
NRTL_A21 = 1.07
NRTL_B12 = 155.68
NRTL_B21 = -19.50
NRTL_ALPHA = 0.30


# =========================================================
# VISUAL THEME
# =========================================================

COLORS = {
    "app_bg": "#EEF2F6",
    "surface": "#FFFFFF",
    "surface_alt": "#F7F9FC",
    "border": "#D7DEE8",
    "text": "#182433",
    "muted": "#667386",
    "navy": "#17324D",
    "blue": "#2F6B9A",
    "blue_soft": "#E8F1F8",
    "water": "#3A86C8",
    "eg": "#D99032",
    "green": "#2E8B68",
    "green_soft": "#E8F5EF",
    "amber": "#B7791F",
    "amber_soft": "#FFF4D8",
    "red": "#B54747",
    "red_soft": "#FCEAEA",
    "shadow": "#C8D0DA",
}

FONT_FAMILY = "Helvetica Neue"


# =========================================================
# REACTOR MODEL
# =========================================================

def cond_1_reac(
    pta_feed_kg,
    eg_feed_kg,
    heat_sup,
    pta_conv=PTA_CONVERSION,
    eg_conv=EG_CONVERSION,
    t_reac_i=30.0,
    t_reac_o=240.0,
    p_reac=2.0,
    simul_wat=20.63,
):
    """First PET condensation reactor calculation."""

    pta_feed_mol = pta_feed_kg / 166.0
    eg_feed_mol = eg_feed_kg / 62.0

    bhet_reac_mol = pta_feed_mol * pta_conv
    bhet_reac_kg = bhet_reac_mol * 254.0

    wat_reac_mol = bhet_reac_mol * 2.0
    wat_reac_kg = wat_reac_mol * 18.0

    pta_reac_mol = pta_feed_mol - bhet_reac_mol
    pta_reac_kg = pta_reac_mol * 166.0

    eg_consumed_mol = bhet_reac_mol * 2.0
    eg_consumed_kg = eg_consumed_mol * 62.0

    eg_reac_mol = max(0.0, eg_feed_mol - eg_consumed_mol)
    eg_reac_kg = eg_reac_mol * 62.0

    sensible_heat = (
        pta_feed_kg * PTA_CP
        + eg_feed_kg * EG_CP
    ) * (t_reac_o - t_reac_i)

    dissolution_heat = pta_feed_kg * PTA_DISSOLUTION_HEAT
    reaction_heat = bhet_reac_mol * -31.42

    heat_before_evaporation = (
        sensible_heat
        + dissolution_heat
        + reaction_heat
    )

    heat_for_evaporation = max(
        0.0,
        heat_sup - heat_before_evaporation,
    )

    total_liquid_mol = eg_reac_mol + wat_reac_mol + bhet_reac_mol

    y_wat_kg = 0.0
    y_eg_kg = 0.0

    if total_liquid_mol > 0.0:
        x_eg = eg_reac_mol / total_liquid_mol
        x_wat = wat_reac_mol / total_liquid_mol / simul_wat

        pv_eg = 10.0 ** (
            4.9701
            - 1915.0 / (t_reac_o + 273.0 - 85.0)
        )
        pv_wat = (t_reac_o / 100.0) ** 4.0

        y_eg_mol = (pv_eg / p_reac) * x_eg
        y_wat_mol = (pv_wat / p_reac) * x_wat

        vapour_mass = y_wat_mol * 18.0 + y_eg_mol * 62.0

        if vapour_mass > 0.0:
            y_wat_kg = y_wat_mol * 18.0 / vapour_mass
            y_eg_kg = y_eg_mol * 62.0 / vapour_mass

    average_latent_heat = (
        WATER_LATENT * y_wat_kg
        + EG_LATENT * y_eg_kg
    )

    total_evap_kg = (
        heat_for_evaporation / average_latent_heat
        if average_latent_heat > 0.0
        else 0.0
    )

    wat_evap_kg = min(total_evap_kg * y_wat_kg, wat_reac_kg)
    eg_evap_kg = min(total_evap_kg * y_eg_kg, eg_reac_kg)

    return {
        "pta_feed_kg": pta_feed_kg,
        "eg_feed_kg": eg_feed_kg,
        "heat_sup": heat_sup,
        "pta_reac_kg": pta_reac_kg,
        "eg_reac_kg": eg_reac_kg,
        "eg_consumed_kg": eg_consumed_kg,
        "wat_reac_kg": wat_reac_kg,
        "bhet_reac_kg": bhet_reac_kg,
        "wat_evap_kg": wat_evap_kg,
        "eg_evap_kg": eg_evap_kg,
        "total_evap_kg": wat_evap_kg + eg_evap_kg,
        "t_reac_i": t_reac_i,
        "t_reac_o": t_reac_o,
        "p_reac": p_reac,
        "heat_before_evaporation": heat_before_evaporation,
        "heat_for_evaporation": heat_for_evaporation,
    }


# =========================================================
# VLE AND COLUMN MODEL
# =========================================================

def antoine_pressure_bar(temperature_k, a, b, c):
    return 10.0 ** (a - b / (temperature_k + c))


def water_vapor_pressure_bar(temperature_c):
    temperature_k = temperature_c + 273.15

    if 344.0 <= temperature_k < 379.0:
        coefficients = (5.08354, 1663.125, -45.622)
    elif 379.0 <= temperature_k <= 573.0:
        coefficients = (3.55959, 643.748, -198.043)
    else:
        raise ValueError(
            f"No water Antoine coefficients at {temperature_c:.2f} degC."
        )

    return antoine_pressure_bar(temperature_k, *coefficients)


def eg_vapor_pressure_bar(temperature_c):
    temperature_k = temperature_c + 273.15

    if not 323.0 <= temperature_k <= 473.0:
        raise ValueError(
            f"EG Antoine correlation invalid at {temperature_k:.2f} K."
        )

    return antoine_pressure_bar(
        temperature_k,
        4.97012,
        1914.951,
        -84.996,
    )


def mass_to_mole_fraction_water(water_mass_fraction):
    water_basis = water_mass_fraction / MW_WATER
    eg_basis = (1.0 - water_mass_fraction) / MW_EG
    denominator = water_basis + eg_basis

    if denominator <= 0.0:
        raise ValueError("Invalid water/EG composition.")

    return water_basis / denominator


def water_mole_to_mass_fraction(x_water):
    x_water = max(0.0, min(1.0, x_water))
    water_mass = x_water * MW_WATER
    eg_mass = (1.0 - x_water) * MW_EG
    denominator = water_mass + eg_mass
    return water_mass / denominator if denominator > 0.0 else 0.0


def stream_molar_flow(mass_flow_kg_h, water_mass_fraction):
    return (
        mass_flow_kg_h * water_mass_fraction / MW_WATER
        + mass_flow_kg_h * (1.0 - water_mass_fraction) / MW_EG
    )


def calculate_distillate_bottom(
    feed_flow,
    water_feed,
    water_distillate,
    water_bottom,
):
    denominator = water_distillate - water_bottom

    if abs(denominator) < 1e-12:
        raise ValueError(
            "Distillate and bottom compositions cannot be equal."
        )

    distillate_flow = (
        feed_flow
        * (water_feed - water_bottom)
        / denominator
    )
    bottom_flow = feed_flow - distillate_flow

    if distillate_flow < 0.0 or bottom_flow < 0.0:
        raise ValueError(
            "Feed composition is outside the specified product range."
        )

    return distillate_flow, bottom_flow


def nrtl_activity_coefficients(x_water, temperature_c):
    x_water = max(1e-12, min(1.0 - 1e-12, x_water))
    x_eg = 1.0 - x_water
    temperature_k = temperature_c + 273.15

    tau_12 = NRTL_A12 + NRTL_B12 / temperature_k
    tau_21 = NRTL_A21 + NRTL_B21 / temperature_k

    g_12 = math.exp(-NRTL_ALPHA * tau_12)
    g_21 = math.exp(-NRTL_ALPHA * tau_21)

    denominator_21 = x_water + x_eg * g_21
    denominator_12 = x_eg + x_water * g_12

    ln_gamma_water = x_eg ** 2 * (
        tau_21 * (g_21 / denominator_21) ** 2
        + tau_12 * g_12 / denominator_12 ** 2
    )

    ln_gamma_eg = x_water ** 2 * (
        tau_12 * (g_12 / denominator_12) ** 2
        + tau_21 * g_21 / denominator_21 ** 2
    )

    return math.exp(ln_gamma_water), math.exp(ln_gamma_eg)


def ideal_dew_point_residual(temperature_c, y_water, pressure_bar):
    y_eg = 1.0 - y_water
    p_water = water_vapor_pressure_bar(temperature_c)
    p_eg = eg_vapor_pressure_bar(temperature_c)

    return (
        y_water * pressure_bar / p_water
        + y_eg * pressure_bar / p_eg
        - 1.0
    )


def calculate_stage_equilibrium_ideal(
    y_water,
    pressure_bar,
    temperature_low_c=71.0,
    temperature_high_c=199.0,
):
    residual_low = ideal_dew_point_residual(
        temperature_low_c,
        y_water,
        pressure_bar,
    )
    residual_high = ideal_dew_point_residual(
        temperature_high_c,
        y_water,
        pressure_bar,
    )

    if residual_low * residual_high > 0.0:
        raise ValueError(
            f"Ideal dew-point root not bracketed: "
            f"yW={y_water:.6f}, P={pressure_bar:.3f} bar."
        )

    temperature_c = brentq(
        ideal_dew_point_residual,
        temperature_low_c,
        temperature_high_c,
        args=(y_water, pressure_bar),
    )

    p_water = water_vapor_pressure_bar(temperature_c)
    p_eg = eg_vapor_pressure_bar(temperature_c)

    x_water = y_water * pressure_bar / p_water
    x_eg = (1.0 - y_water) * pressure_bar / p_eg
    liquid_sum = x_water + x_eg

    x_water /= liquid_sum
    x_eg /= liquid_sum

    return {
        "temperature_c": temperature_c,
        "x_water": x_water,
        "x_eg": x_eg,
        "y_water": y_water,
        "y_eg": 1.0 - y_water,
        "gamma_water": 1.0,
        "gamma_eg": 1.0,
        "alpha": p_water / p_eg,
        "equilibrium_model": "IDEAL",
        "solver_residual": 0.0,
    }


def nrtl_stage_residuals(unknowns, y_water, pressure_bar):
    x_water, temperature_c = unknowns
    x_eg = 1.0 - x_water
    y_eg = 1.0 - y_water

    gamma_water, gamma_eg = nrtl_activity_coefficients(
        x_water,
        temperature_c,
    )

    p_water = water_vapor_pressure_bar(temperature_c)
    p_eg = eg_vapor_pressure_bar(temperature_c)

    return (
        x_water * gamma_water * p_water / pressure_bar - y_water,
        x_eg * gamma_eg * p_eg / pressure_bar - y_eg,
    )


def calculate_stage_equilibrium_nrtl(
    y_water,
    pressure_bar,
    temperature_low_c=71.0,
    temperature_high_c=199.0,
):
    ideal = calculate_stage_equilibrium_ideal(
        y_water,
        pressure_bar,
        temperature_low_c,
        temperature_high_c,
    )

    solution = least_squares(
        nrtl_stage_residuals,
        x0=(ideal["x_water"], ideal["temperature_c"]),
        bounds=(
            (1e-10, temperature_low_c),
            (1.0 - 1e-10, temperature_high_c),
        ),
        args=(y_water, pressure_bar),
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=500,
    )

    if not solution.success:
        raise ValueError(
            f"NRTL equilibrium solver failed: {solution.message}"
        )

    x_water = float(solution.x[0])
    temperature_c = float(solution.x[1])

    gamma_water, gamma_eg = nrtl_activity_coefficients(
        x_water,
        temperature_c,
    )

    p_water = water_vapor_pressure_bar(temperature_c)
    p_eg = eg_vapor_pressure_bar(temperature_c)

    maximum_residual = max(abs(value) for value in solution.fun)

    if maximum_residual > 1e-7:
        raise ValueError(
            f"NRTL residual too large: {maximum_residual:.3e}"
        )

    return {
        "temperature_c": temperature_c,
        "x_water": x_water,
        "x_eg": 1.0 - x_water,
        "y_water": y_water,
        "y_eg": 1.0 - y_water,
        "gamma_water": gamma_water,
        "gamma_eg": gamma_eg,
        "alpha": gamma_water * p_water / (gamma_eg * p_eg),
        "equilibrium_model": "NRTL",
        "solver_residual": maximum_residual,
    }


def calculate_stage_equilibrium_from_y(y_water, pressure_bar):
    if not 0.0 <= y_water <= 1.0:
        raise ValueError(f"Invalid vapour water fraction: {y_water:.6f}")

    if EQUILIBRIUM_MODEL.upper() == "IDEAL":
        return calculate_stage_equilibrium_ideal(
            y_water,
            pressure_bar,
        )

    return calculate_stage_equilibrium_nrtl(
        y_water,
        pressure_bar,
    )


def calculate_stage_pressure(
    stage,
    number_of_stages,
    top_pressure_bar,
    tray_pressure_drop_bar,
):
    return (
        top_pressure_bar
        + (number_of_stages - stage) * tray_pressure_drop_bar
    )


def calculate_tray_profile(
    reflux_ratio,
    top_pressure_bar,
    tray_pressure_drop_bar,
    number_of_stages,
    xD,
):
    if number_of_stages < 1:
        raise ValueError("Column requires at least one stage.")

    if reflux_ratio <= 0.0:
        raise ValueError("Reflux ratio must be greater than zero.")

    trays = [None] * (number_of_stages + 1)

    for stage in range(number_of_stages, 0, -1):
        if stage == number_of_stages:
            y_stage = xD
        else:
            x_above = trays[stage + 1]["x_water"]
            y_stage = (
                reflux_ratio / (reflux_ratio + 1.0) * x_above
                + xD / (reflux_ratio + 1.0)
            )
            y_stage = max(0.0, min(1.0, y_stage))

        pressure_bar = calculate_stage_pressure(
            stage,
            number_of_stages,
            top_pressure_bar,
            tray_pressure_drop_bar,
        )

        trays[stage] = calculate_stage_equilibrium_from_y(
            y_stage,
            pressure_bar,
        )
        trays[stage]["stage"] = stage
        trays[stage]["pressure_bar"] = pressure_bar

    for stage in range(number_of_stages, 1, -1):
        trays[stage]["composition_change"] = abs(
            trays[stage]["x_water"]
            - trays[stage - 1]["x_water"]
        )

    trays[1]["composition_change"] = 0.0
    return trays


def reflux_ratio_residual(
    reflux_ratio,
    top_pressure_bar,
    tray_pressure_drop_bar,
    number_of_stages,
    xD,
    xB_target,
):
    trays = calculate_tray_profile(
        reflux_ratio,
        top_pressure_bar,
        tray_pressure_drop_bar,
        number_of_stages,
        xD,
    )
    return trays[1]["x_water"] - xB_target


def calculate_required_reflux_ratio(
    top_pressure_bar,
    tray_pressure_drop_bar,
    number_of_stages,
    xD,
    xB_target,
    reflux_low=0.10,
    reflux_high=20.0,
):
    try:
        residual_low = reflux_ratio_residual(
            reflux_low,
            top_pressure_bar,
            tray_pressure_drop_bar,
            number_of_stages,
            xD,
            xB_target,
        )
        residual_high = reflux_ratio_residual(
            reflux_high,
            top_pressure_bar,
            tray_pressure_drop_bar,
            number_of_stages,
            xD,
            xB_target,
        )

        if residual_low * residual_high > 0.0:
            return None

        return brentq(
            reflux_ratio_residual,
            reflux_low,
            reflux_high,
            args=(
                top_pressure_bar,
                tray_pressure_drop_bar,
                number_of_stages,
                xD,
                xB_target,
            ),
        )
    except ValueError:
        return None


def distillation_column(
    water_feed_kg_h,
    eg_feed_kg_h,
    reflux_ratio,
    top_pressure_bar=COLUMN_TOP_PRESSURE_BAR,
    tray_pressure_drop_bar=TRAY_PRESSURE_DROP_BAR,
    number_of_stages=COLUMN_STAGES,
    water_distillate=WATER_DISTILLATE_SPEC,
    water_bottom=WATER_BOTTOM_SPEC,
):
    feed_flow_kg_h = water_feed_kg_h + eg_feed_kg_h

    if feed_flow_kg_h <= 1e-12:
        return {
            "active": False,
            "message": "No vapour feed to the column.",
            "feed_flow_kg_h": 0.0,
            "water_feed_kg_h": 0.0,
            "eg_feed_kg_h": 0.0,
            "distillate_kg_h": 0.0,
            "bottom_kg_h": 0.0,
            "trays": [],
        }

    water_feed_mass_fraction = water_feed_kg_h / feed_flow_kg_h

    distillate_kg_h, bottom_kg_h = calculate_distillate_bottom(
        feed_flow_kg_h,
        water_feed_mass_fraction,
        water_distillate,
        water_bottom,
    )

    zF = mass_to_mole_fraction_water(water_feed_mass_fraction)
    xD = mass_to_mole_fraction_water(water_distillate)
    xB_target = mass_to_mole_fraction_water(water_bottom)

    f_mol = stream_molar_flow(
        feed_flow_kg_h,
        water_feed_mass_fraction,
    )
    d_mol = stream_molar_flow(
        distillate_kg_h,
        water_distillate,
    )
    b_mol = stream_molar_flow(
        bottom_kg_h,
        water_bottom,
    )

    l_mol = reflux_ratio * d_mol
    v_mol = (reflux_ratio + 1.0) * d_mol

    trays = calculate_tray_profile(
        reflux_ratio,
        top_pressure_bar,
        tray_pressure_drop_bar,
        number_of_stages,
        xD,
    )

    calculated_bottom_x_water = trays[1]["x_water"]
    calculated_bottom_water_wt = water_mole_to_mass_fraction(
        calculated_bottom_x_water
    )

    required_reflux_ratio = calculate_required_reflux_ratio(
        top_pressure_bar,
        tray_pressure_drop_bar,
        number_of_stages,
        xD,
        xB_target,
    )

    bottom_pressure_bar = (
        top_pressure_bar
        + (number_of_stages - 1) * tray_pressure_drop_bar
    )

    top_temperature_c = trays[number_of_stages]["temperature_c"]
    bottom_temperature_c = trays[1]["temperature_c"]

    return {
        "active": True,
        "equilibrium_model": EQUILIBRIUM_MODEL.upper(),
        "number_of_stages": number_of_stages,
        "reflux_ratio": reflux_ratio,
        "required_reflux_ratio": required_reflux_ratio,
        "top_pressure_bar": top_pressure_bar,
        "bottom_pressure_bar": bottom_pressure_bar,
        "tray_pressure_drop_bar": tray_pressure_drop_bar,
        "reflux_temperature_c": (
            top_temperature_c - REFLUX_SUBCOOLING_C
        ),
        "feed_flow_kg_h": feed_flow_kg_h,
        "water_feed_kg_h": water_feed_kg_h,
        "eg_feed_kg_h": eg_feed_kg_h,
        "water_feed_mass_fraction": water_feed_mass_fraction,
        "distillate_kg_h": distillate_kg_h,
        "bottom_kg_h": bottom_kg_h,
        "F_mol": f_mol,
        "D_mol": d_mol,
        "B_mol": b_mol,
        "L_mol": l_mol,
        "V_mol": v_mol,
        "zF": zF,
        "xD": xD,
        "xB_target": xB_target,
        "trays": trays,
        "calculated_bottom_x_water": calculated_bottom_x_water,
        "calculated_bottom_water_wt": calculated_bottom_water_wt,
        "target_bottom_water_wt": water_bottom,
        "bottom_composition_error": (
            calculated_bottom_x_water - xB_target
        ),
        "total_balance_residual": (
            f_mol + l_mol - v_mol - b_mol
        ),
        "top_temperature_c": top_temperature_c,
        "bottom_temperature_c": bottom_temperature_c,
    }


# =========================================================
# FORMATTING AND DRAWING HELPERS
# =========================================================

def fmt_flow(value):
    return f"{value:,.0f} kg/h"


def fmt_temperature(value):
    return f"{value:.1f} degC"


def fmt_pressure(value):
    return f"{value:.3f} bar"


def safe_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass

    return ImageFont.load_default()


def blend_rgb(start, end, fraction):
    fraction = max(0.0, min(1.0, fraction))
    return tuple(
        round(start[index] + (end[index] - start[index]) * fraction)
        for index in range(3)
    )


def water_eg_colour(water_mass_fraction):
    eg_rgb = (217, 144, 50)
    water_rgb = (58, 134, 200)
    return blend_rgb(eg_rgb, water_rgb, water_mass_fraction)


def rounded_box(draw, xy, fill, outline, radius=10, width=1):
    draw.rounded_rectangle(
        xy,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def draw_information_card(
    draw,
    x,
    y,
    width,
    title,
    rows,
    accent="#2F6B9A",
):
    row_height = 21
    height = 42 + len(rows) * row_height + 10

    rounded_box(
        draw,
        (x + 3, y + 4, x + width + 3, y + height + 4),
        fill="#DCE2E9",
        outline=None,
        radius=11,
        width=0,
    )
    rounded_box(
        draw,
        (x, y, x + width, y + height),
        fill="#FFFFFF",
        outline="#D7DEE8",
        radius=11,
    )

    draw.rounded_rectangle(
        (x, y, x + 5, y + height),
        radius=4,
        fill=accent,
    )

    title_font = safe_font(13, bold=True)
    row_font = safe_font(12)
    value_font = safe_font(12, bold=True)

    draw.text(
        (x + 16, y + 11),
        title.upper(),
        fill="#17324D",
        font=title_font,
    )

    current_y = y + 38
    for label, value in rows:
        draw.text(
            (x + 16, current_y),
            label,
            fill="#667386",
            font=row_font,
        )

        bbox = draw.textbbox((0, 0), value, font=value_font)
        value_width = bbox[2] - bbox[0]

        draw.text(
            (x + width - value_width - 14, current_y),
            value,
            fill="#182433",
            font=value_font,
        )
        current_y += row_height


def load_process_background(width, height):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "pet_condensation.png")

    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((width, height), Image.Resampling.LANCZOS)

        background = Image.new(
            "RGB",
            (width, height),
            COLORS["surface_alt"],
        )

        x = (width - image.width) // 2
        y = (height - image.height) // 2
        background.paste(image, (x, y))
        return background

    # Clean fallback when the original diagram is absent.
    image = Image.new("RGB", (width, height), COLORS["surface_alt"])
    draw = ImageDraw.Draw(image)
    title_font = safe_font(24, bold=True)
    body_font = safe_font(15)

    draw.text(
        (width // 2, height // 2 - 30),
        "Process image not found",
        anchor="mm",
        fill=COLORS["navy"],
        font=title_font,
    )
    draw.text(
        (width // 2, height // 2 + 10),
        "Place pet_condensation.png beside this Python file.",
        anchor="mm",
        fill=COLORS["muted"],
        font=body_font,
    )
    return image


def draw_column_profile(draw, column, image_width, image_height):
    if not column.get("active"):
        return

    # Existing reference coordinates, scaled from the original image.
    reference_width = 1000.0
    reference_height = 600.0

    sx = image_width / reference_width
    sy = image_height / reference_height

    column_left = int(655 * sx)
    column_right = int(700 * sx)
    column_top = int(94 * sy)
    column_bottom = int(380 * sy)

    number_of_stages = column["number_of_stages"]
    tray_pitch = (column_bottom - column_top) / number_of_stages

    label_font = safe_font(max(9, int(11 * min(sx, sy))))
    value_font = safe_font(max(9, int(11 * min(sx, sy))), bold=True)

    for stage in range(1, number_of_stages + 1):
        tray = column["trays"][stage]
        water_wt = water_mole_to_mass_fraction(tray["x_water"])
        colour = water_eg_colour(water_wt)

        y_bottom = column_bottom - (stage - 1) * tray_pitch
        y_top = column_bottom - stage * tray_pitch
        tray_gap = int(2 * sy)
        draw.rectangle(
            (
                column_left,
                int(y_top) + tray_gap,
                column_right,
                int(y_bottom) - tray_gap,
            ),
            fill=colour,
            outline="#FFFFFF",
            width=1,
        )

        centre_y = int((y_top + y_bottom) / 2)

        draw.text(
            (column_left - int(10 * sx), centre_y),
            f"n {stage}",
            anchor="rm",
            fill="#334155",
            font=label_font,
        )

        draw.text(
            (column_right + int(8 * sx), centre_y),
            f"{tray['temperature_c']:.1f}",
            anchor="lm",
            fill="#334155",
            font=value_font,
        )

        draw.text(
            (column_left - int(42 * sx), centre_y),
            f"{water_wt * 100:.1f}%",
            anchor="rm",
            fill="#2F6B9A",
            font=value_font,
        )

    draw.text(
        (column_left - int(10 * sx), column_top - int(20 * sy)),
        "H2O",
        anchor="rm",
        fill=COLORS["water"],
        font=value_font,
    )
    draw.text(
        (column_right + int(8 * sx), column_top - int(20 * sy)),
        "degC",
        anchor="lm",
        fill="#667386",
        font=value_font,
    )


def create_process_overlay(width, height, reactor, column):
    image = load_process_background(width, height)
    draw = ImageDraw.Draw(image)

    # Coordinates based on a 1000 x 600 process area.
    sx = width / 1000.0
    sy = height / 600.0

    draw_information_card(
        draw,
        int(55 * sx),
        int(75 * sy),
        int(175 * sx),
        "Feed",
        [
            ("PTA", fmt_flow(reactor["pta_feed_kg"])),
            ("EG", fmt_flow(reactor["eg_feed_kg"])),
        ],
        accent=COLORS["blue"],
    )

    draw_information_card(
        draw,
        int(55 * sx),
        int(455 * sy),
        int(165 * sx),
        "Reactor 1",
        [
            ("Temperature", fmt_temperature(reactor["t_reac_o"])),
            ("Pressure", fmt_pressure(reactor["p_reac"])),
            ("BHET", fmt_flow(reactor["bhet_reac_kg"])),
            ("Water", fmt_flow(

            reactor["wat_reac_kg"] - reactor["wat_evap_kg"]

            )),

            ("EG", fmt_flow(

            reactor["eg_reac_kg"] - reactor["eg_evap_kg"]

            )),
        ],
        accent=COLORS["eg"],
    )

    draw_information_card(
        draw,
        int(260 * sx),
        int(75 * sy),
        int(165 * sx),
        "Reactor 1 vapour",
        [
            ("Water", fmt_flow(reactor["wat_evap_kg"])),
            ("EG", fmt_flow(reactor["eg_evap_kg"])),
            ("Total", fmt_flow(reactor["total_evap_kg"])),
        ],
        accent=COLORS["water"],
    )

    if column.get("active"):
        draw_information_card(
            draw,
            int(780 * sx),
            int(185 * sy),
            int(165 * sx),
            "Column products",
            [
                ("Distillate", fmt_flow(column["distillate_kg_h"])),
                ("Bottom", fmt_flow(column["bottom_kg_h"])),
                (
                    "Bottom H2O",
                    f"{column['calculated_bottom_water_wt'] * 100:.2f} wt%",
                ),
            ],
            accent=COLORS["green"],
        )

        draw_information_card(
            draw,
            int(780 * sx),
            int(425 * sy),
            int(165 * sx),
            "Column conditions",
            [
                ("Top", fmt_temperature(column["top_temperature_c"])),
                ("Bottom", fmt_temperature(column["bottom_temperature_c"])),
                (
                    "Pressure",
                    f"{column['top_pressure_bar']:.3f}-"
                    f"{column['bottom_pressure_bar']:.3f} bar",
                ),
            ],
            accent=COLORS["navy"],
        )

    draw_column_profile(draw, column, width, height)
    return image


# =========================================================
# TKINTER WIDGET HELPERS
# =========================================================

class MetricCard(tk.Frame):
    def __init__(self, master, title, value="--", unit="", accent=None):
        super().__init__(
            master,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            bd=0,
        )

        self.accent = accent or COLORS["blue"]

        tk.Frame(
            self,
            bg=self.accent,
            height=4,
        ).pack(fill="x")

        body = tk.Frame(self, bg=COLORS["surface"])
        body.pack(fill="both", expand=True, padx=13, pady=10)

        tk.Label(
            body,
            text=title.upper(),
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=(FONT_FAMILY, 9, "bold"),
            anchor="w",
        ).pack(fill="x")

        value_row = tk.Frame(body, bg=COLORS["surface"])
        value_row.pack(fill="x", pady=(5, 0))

        self.value_label = tk.Label(
            value_row,
            text=value,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=(FONT_FAMILY, 17, "bold"),
            anchor="w",
        )
        self.value_label.pack(side="left")

        self.unit_label = tk.Label(
            value_row,
            text=unit,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=(FONT_FAMILY, 9),
            anchor="w",
        )
        self.unit_label.pack(side="left", padx=(5, 0), pady=(6, 0))

    def set(self, value, unit=None):
        self.value_label.configure(text=value)
        if unit is not None:
            self.unit_label.configure(text=unit)


class SliderCard(tk.Frame):
    def __init__(
        self,
        master,
        title,
        variable,
        from_,
        to,
        resolution,
        command,
        formatter,
    ):
        super().__init__(
            master,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            bd=0,
        )

        header = tk.Frame(self, bg=COLORS["surface"])
        header.pack(fill="x", padx=14, pady=(12, 2))

        tk.Label(
            header,
            text=title,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=(FONT_FAMILY, 11, "bold"),
        ).pack(side="left")

        self.value_label = tk.Label(
            header,
            text=formatter(variable.get()),
            bg=COLORS["blue_soft"],
            fg=COLORS["blue"],
            font=(FONT_FAMILY, 10, "bold"),
            padx=8,
            pady=3,
        )
        self.value_label.pack(side="right")

        self.formatter = formatter
        self.variable = variable
        self.external_command = command

        self.scale = ttk.Scale(
            self,
            variable=variable,
            from_=from_,
            to=to,
            command=self._changed,
        )
        self.scale.pack(fill="x", padx=14, pady=(7, 13))

        self.resolution = resolution

    def _changed(self, value):
        rounded = round(float(value) / self.resolution) * self.resolution
        self.variable.set(rounded)
        self.value_label.configure(text=self.formatter(rounded))
        self.external_command()


# =========================================================
# MAIN APPLICATION
# =========================================================

class GlassPlantApp(tk.Tk):
    PROCESS_WIDTH = 1000
    PROCESS_HEIGHT = 600

    def __init__(self):
        super().__init__()

        self.title("The Glass Plant - PET Condensation")
        self.geometry("1600x920")
        self.minsize(1420, 820)
        self.configure(bg=COLORS["app_bg"])

        self.pta_var = tk.DoubleVar(value=PTA_DEFAULT_KG_H)
        self.heat_var = tk.DoubleVar(value=HEAT_DEFAULT_KCAL_H)
        self.reflux_var = tk.DoubleVar(value=REFLUX_DEFAULT)

        self.reactor = {}
        self.column = {}

        self._update_job = None
        self._process_photo = None

        self._configure_styles()
        self._build_header()
        self._build_body()
        self._build_footer()

        self.update_model()

    def _configure_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "TScale",
            background=COLORS["surface"],
            troughcolor="#DCE5EE",
            bordercolor=COLORS["surface"],
            lightcolor=COLORS["blue"],
            darkcolor=COLORS["blue"],
        )

    def _build_header(self):
        header = tk.Frame(
            self,
            bg=COLORS["navy"],
            height=74,
        )
        header.pack(fill="x")
        header.pack_propagate(False)

        title_block = tk.Frame(header, bg=COLORS["navy"])
        title_block.pack(side="left", padx=24, pady=12)

        tk.Label(
            title_block,
            text="THE GLASS PLANT",
            bg=COLORS["navy"],
            fg="#FFFFFF",
            font=(FONT_FAMILY, 19, "bold"),
            anchor="w",
        ).pack(anchor="w")

        tk.Label(
            title_block,
            text="PET condensation | Process visibility and operator guidance",
            bg=COLORS["navy"],
            fg="#C8D8E8",
            font=(FONT_FAMILY, 10),
            anchor="w",
        ).pack(anchor="w", pady=(2, 0))

        self.status_pill = tk.Label(
            header,
            text="  NORMAL OPERATION  ",
            bg=COLORS["green"],
            fg="#FFFFFF",
            font=(FONT_FAMILY, 10, "bold"),
            padx=10,
            pady=7,
        )
        self.status_pill.pack(side="right", padx=24)

    def _build_body(self):
        body = tk.Frame(self, bg=COLORS["app_bg"])
        body.pack(fill="both", expand=True, padx=16, pady=14)

        body.grid_columnconfigure(0, minsize=270)
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, minsize=290)
        body.grid_rowconfigure(0, weight=1)

        self.left_panel = tk.Frame(body, bg=COLORS["app_bg"])
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        self.centre_panel = tk.Frame(
            body,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        self.centre_panel.grid(row=0, column=1, sticky="nsew")

        self.right_panel = tk.Frame(body, bg=COLORS["app_bg"])
        self.right_panel.grid(row=0, column=2, sticky="nsew", padx=(12, 0))

        self._build_controls()
        self._build_process_panel()
        self._build_insight_panel()

    def _section_title(self, parent, title, subtitle):
        frame = tk.Frame(parent, bg=COLORS["app_bg"])
        frame.pack(fill="x", pady=(2, 10))

        tk.Label(
            frame,
            text=title,
            bg=COLORS["app_bg"],
            fg=COLORS["navy"],
            font=(FONT_FAMILY, 14, "bold"),
            anchor="w",
        ).pack(fill="x")

        tk.Label(
            frame,
            text=subtitle,
            bg=COLORS["app_bg"],
            fg=COLORS["muted"],
            font=(FONT_FAMILY, 9),
            anchor="w",
            justify="left",
            wraplength=255,
        ).pack(fill="x", pady=(3, 0))

    def _build_controls(self):
        self._section_title(
            self.left_panel,
            "Operating inputs",
            "Adjust the main process variables. "
            "The model updates automatically.",
        )

        SliderCard(
            self.left_panel,
            "PTA feed",
            self.pta_var,
            20_000,
            45_000,
            100,
            self.schedule_update,
            lambda value: f"{value:,.0f} kg/h",
        ).pack(fill="x", pady=(0, 10))

        SliderCard(
            self.left_panel,
            "Heat supplied",
            self.heat_var,
            6_000_000,
            12_000_000,
            25_000,
            self.schedule_update,
            lambda value: f"{value / 1_000_000:.2f} MMkcal/h",
        ).pack(fill="x", pady=(0, 10))

        SliderCard(
            self.left_panel,
            "Reflux ratio",
            self.reflux_var,
            0.25,
            5.00,
            0.01,
            self.schedule_update,
            lambda value: f"{value:.2f}",
        ).pack(fill="x", pady=(0, 14))

        tk.Label(
            self.left_panel,
            text="LIVE MODEL",
            bg=COLORS["app_bg"],
            fg=COLORS["muted"],
            font=(FONT_FAMILY, 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(3, 7))

        self.eg_metric = MetricCard(
            self.left_panel,
            "EG feed",
            accent=COLORS["eg"],
        )
        self.eg_metric.pack(fill="x", pady=(0, 8))

        self.vapour_metric = MetricCard(
            self.left_panel,
            "Reactor vapour",
            accent=COLORS["water"],
        )
        self.vapour_metric.pack(fill="x", pady=(0, 8))

        self.distillate_metric = MetricCard(
            self.left_panel,
            "Distillate",
            accent=COLORS["blue"],
        )
        self.distillate_metric.pack(fill="x", pady=(0, 8))

        self.bottom_metric = MetricCard(
            self.left_panel,
            "Bottom",
            accent=COLORS["green"],
        )
        self.bottom_metric.pack(fill="x")

    def _build_process_panel(self):
        title_row = tk.Frame(
            self.centre_panel,
            bg=COLORS["surface"],
            height=52,
        )
        title_row.pack(fill="x")
        title_row.pack_propagate(False)

        tk.Label(
            title_row,
            text="PROCESS OVERVIEW",
            bg=COLORS["surface"],
            fg=COLORS["navy"],
            font=(FONT_FAMILY, 12, "bold"),
        ).pack(side="left", padx=16)

        self.model_badge = tk.Label(
            title_row,
            text=f"VLE: {EQUILIBRIUM_MODEL}",
            bg=COLORS["blue_soft"],
            fg=COLORS["blue"],
            font=(FONT_FAMILY, 9, "bold"),
            padx=9,
            pady=4,
        )
        self.model_badge.pack(side="right", padx=16)

        separator = tk.Frame(
            self.centre_panel,
            bg=COLORS["border"],
            height=1,
        )
        separator.pack(fill="x")

        image_container = tk.Frame(
            self.centre_panel,
            bg=COLORS["surface_alt"],
        )
        image_container.pack(fill="both", expand=True, padx=10, pady=10)

        self.process_label = tk.Label(
            image_container,
            bg=COLORS["surface_alt"],
            bd=0,
        )
        self.process_label.pack(fill="both", expand=True)

    def _build_insight_panel(self):
        self._section_title(
            self.right_panel,
            "Operating intelligence",
            "A concise interpretation of the current model state.",
        )

        self.insight_card = tk.Frame(
            self.right_panel,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        self.insight_card.pack(fill="x", pady=(0, 12))

        self.insight_strip = tk.Frame(
            self.insight_card,
            bg=COLORS["green"],
            height=5,
        )
        self.insight_strip.pack(fill="x")

        insight_body = tk.Frame(
            self.insight_card,
            bg=COLORS["surface"],
        )
        insight_body.pack(fill="x", padx=15, pady=14)

        self.insight_title = tk.Label(
            insight_body,
            text="Model ready",
            bg=COLORS["surface"],
            fg=COLORS["navy"],
            font=(FONT_FAMILY, 14, "bold"),
            anchor="w",
        )
        self.insight_title.pack(fill="x")

        self.insight_text = tk.Label(
            insight_body,
            text="",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=(FONT_FAMILY, 10),
            justify="left",
            anchor="w",
            wraplength=250,
        )
        self.insight_text.pack(fill="x", pady=(8, 0))

        self.action_label = tk.Label(
            insight_body,
            text="",
            bg=COLORS["surface_alt"],
            fg=COLORS["navy"],
            font=(FONT_FAMILY, 10, "bold"),
            justify="left",
            anchor="w",
            wraplength=230,
            padx=10,
            pady=9,
        )
        self.action_label.pack(fill="x", pady=(12, 0))

        tk.Label(
            self.right_panel,
            text="COLUMN PERFORMANCE",
            bg=COLORS["app_bg"],
            fg=COLORS["muted"],
            font=(FONT_FAMILY, 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(2, 7))

        self.reflux_metric = MetricCard(
            self.right_panel,
            "Selected / required reflux",
            accent=COLORS["blue"],
        )
        self.reflux_metric.pack(fill="x", pady=(0, 8))

        self.bottom_water_metric = MetricCard(
            self.right_panel,
            "Bottom water",
            accent=COLORS["green"],
        )
        self.bottom_water_metric.pack(fill="x", pady=(0, 8))

        self.top_temperature_metric = MetricCard(
            self.right_panel,
            "Top temperature",
            accent=COLORS["water"],
        )
        self.top_temperature_metric.pack(fill="x", pady=(0, 8))

        self.bottom_temperature_metric = MetricCard(
            self.right_panel,
            "Bottom temperature",
            accent=COLORS["eg"],
        )
        self.bottom_temperature_metric.pack(fill="x")

    def _build_footer(self):
        footer = tk.Frame(
            self,
            bg=COLORS["surface"],
            height=31,
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        footer.pack(fill="x")
        footer.pack_propagate(False)

        self.footer_label = tk.Label(
            footer,
            text="Model ready",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=(FONT_FAMILY, 9),
            anchor="w",
        )
        self.footer_label.pack(fill="x", padx=18, pady=6)

    def schedule_update(self):
        if self._update_job is not None:
            self.after_cancel(self._update_job)

        self._update_job = self.after(120, self.update_model)

    def update_model(self):
        self._update_job = None

        try:
            pta_feed_kg = float(self.pta_var.get())
            eg_feed_kg = pta_feed_kg * PTA_TO_EG_MASS_RATIO
            heat_sup = float(self.heat_var.get())
            reflux_ratio = float(self.reflux_var.get())

            self.reactor = cond_1_reac(
                pta_feed_kg=pta_feed_kg,
                eg_feed_kg=eg_feed_kg,
                heat_sup=heat_sup,
            )

            self.column = distillation_column(
                water_feed_kg_h=self.reactor["wat_evap_kg"],
                eg_feed_kg_h=self.reactor["eg_evap_kg"],
                reflux_ratio=reflux_ratio,
            )

            self._update_dashboard()
            self._draw_process()
            self.footer_label.configure(
                text=(
                    "Last calculation completed | "
                    f"{COLUMN_STAGES} theoretical stages | "
                    f"tray dP {TRAY_PRESSURE_DROP_BAR:.5f} bar"
                ),
                fg=COLORS["muted"],
            )

        except (ValueError, ArithmeticError) as error:
            self.status_pill.configure(
                text="  MODEL ERROR  ",
                bg=COLORS["red"],
            )
            self.insight_strip.configure(bg=COLORS["red"])
            self.insight_title.configure(text="Calculation error")
            self.insight_text.configure(text=str(error))
            self.action_label.configure(
                text="Check the selected operating inputs."
            )
            self.footer_label.configure(
                text=f"Model error: {error}",
                fg=COLORS["red"],
            )

    def _update_dashboard(self):
        reactor = self.reactor
        column = self.column

        self.eg_metric.set(
            f"{reactor['eg_feed_kg']:,.0f}",
            "kg/h",
        )
        self.vapour_metric.set(
            f"{reactor['total_evap_kg']:,.0f}",
            "kg/h",
        )

        if not column.get("active"):
            self.distillate_metric.set("--", "kg/h")
            self.bottom_metric.set("--", "kg/h")
            self._set_insight(
                "warning",
                "No column feed",
                "The reactor is not generating a vapour feed "
                "for the distillation column.",
                "Increase heat input or review the reactor assumptions.",
            )
            return

        self.distillate_metric.set(
            f"{column['distillate_kg_h']:,.0f}",
            "kg/h",
        )
        self.bottom_metric.set(
            f"{column['bottom_kg_h']:,.0f}",
            "kg/h",
        )

        required = column["required_reflux_ratio"]
        required_text = "--" if required is None else f"{required:.2f}"

        self.reflux_metric.set(
            f"{column['reflux_ratio']:.2f} / {required_text}",
            "",
        )
        self.bottom_water_metric.set(
            f"{column['calculated_bottom_water_wt'] * 100:.2f}",
            "wt%",
        )
        self.top_temperature_metric.set(
            f"{column['top_temperature_c']:.1f}",
            "degC",
        )
        self.bottom_temperature_metric.set(
            f"{column['bottom_temperature_c']:.1f}",
            "degC",
        )

        calculated = column["calculated_bottom_water_wt"]
        target = column["target_bottom_water_wt"]
        difference = calculated - target

        if required is not None and column["reflux_ratio"] < required:
            self._set_insight(
                "warning",
                "Reflux below calculated requirement",
                (
                    f"The selected reflux ratio is "
                    f"{column['reflux_ratio']:.2f}, while the model "
                    f"estimates about {required:.2f} is required."
                ),
                "Increase reflux gradually and verify condenser duty.",
            )
        elif difference > 0.002:
            self._set_insight(
                "warning",
                "Bottom water above target",
                (
                    f"Calculated bottom water is "
                    f"{calculated * 100:.2f} wt% versus a "
                    f"{target * 100:.2f} wt% target."
                ),
                "Review reflux, stage efficiency and bottom-feed assumptions.",
            )
        else:
            self._set_insight(
                "normal",
                "Separation is close to target",
                (
                    f"Bottom water is {calculated * 100:.2f} wt%. "
                    f"Top and bottom temperatures are "
                    f"{column['top_temperature_c']:.1f} and "
                    f"{column['bottom_temperature_c']:.1f} degC."
                ),
                "Maintain conditions and monitor the composition trend.",
            )

    def _set_insight(self, level, title, text, action):
        if level == "normal":
            colour = COLORS["green"]
            status = "  NORMAL OPERATION  "
        elif level == "warning":
            colour = COLORS["amber"]
            status = "  ATTENTION  "
        else:
            colour = COLORS["red"]
            status = "  MODEL ERROR  "

        self.status_pill.configure(text=status, bg=colour)
        self.insight_strip.configure(bg=colour)
        self.insight_title.configure(text=title)
        self.insight_text.configure(text=text)
        self.action_label.configure(text=f"Suggested action\n{action}")

    def _draw_process(self):
        image = create_process_overlay(
            self.PROCESS_WIDTH,
            self.PROCESS_HEIGHT,
            self.reactor,
            self.column,
        )

        self._process_photo = ImageTk.PhotoImage(image)
        self.process_label.configure(image=self._process_photo)


def main():
    app = GlassPlantApp()
    app.mainloop()


if __name__ == "__main__":
    main()
