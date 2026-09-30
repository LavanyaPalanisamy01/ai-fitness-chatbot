"""Helper module for basic fitness calculations and prompt context.
"""


def calculate_bmi(weight_kg: float, height_cm: float) -> tuple[float, str]:
    """Calculate BMI and return the numeric value and category.

    Formula: weight (kg) / [height (m)]^2
    """
    if height_cm <= 0 or weight_kg <= 0:
        return 0.0, "Unknown"

    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m**2), 1)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25.0:
        category = "Normal weight"
    elif bmi < 30.0:
        category = "Overweight"
    else:
        category = "Obese"

    return bmi, category


def build_system_prompt(
    age: int,
    height_cm: float,
    weight_kg: float,
    goal: str,
    bmi: float,
    bmi_category: str,
) -> str:
    """Build a concise system prompt providing user context to the AI assistant."""
    return (
        "You are a helpful, encouraging AI Fitness Assistant.\n\n"
        "User Profile Context:\n"
        f"- Age: {age} years old\n"
        f"- Height: {height_cm} cm\n"
        f"- Weight: {weight_kg} kg\n"
        f"- Fitness Goal: {goal}\n"
        f"- BMI: {bmi} ({bmi_category})\n\n"
        "Instructions:\n"
        "1. Tailor all workout, nutrition, and wellness suggestions specifically to the user's goal, age, and BMI profile.\n"
        "2. Keep explanations clear, practical, and easy to follow.\n"
        "3. Remind the user to consult a doctor or healthcare provider before starting any rigorous diet or workout regimen."
    )
