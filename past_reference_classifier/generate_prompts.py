import random
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# DEFINITIONS FOR POSITIVE (“Past‐Referencing”) PROMPTS
# ──────────────────────────────────────────────────────────────────────────────

historical_periods = [
    "14th-century medieval village", "18th-century French salon", "1920s Parisian café",
    "Victorian London street", "Ancient Roman forum", "1950s American diner",
    "Renaissance Florence workshop", "Wild West saloon", "1960s NASA control room",
    "Byzantine mosaic chapel", "1600s Spanish galleon deck", "Medieval monastery scriptorium",
    "Ancient Egyptian pyramid complex", "World War II telegram office",
    "Victorian botanical glasshouse", "1920s speakeasy", "Roman Colosseum arena",
    "1800s steamboat on the Mississippi", "17th-century Mughal court",
    "1950s drive-in movie theater", "Medieval jousting tournament",
    "1920s Prohibition-era rumrunner’s boat", "Ancient Greek amphitheater",
    "1930s New York City skyline", "Renaissance Venetian gondola"
]

historical_modifiers = [
    "at dawn, painted in sepia tone", "under candlelit chandeliers",
    "with horse-drawn carriages in the background", "in grainy black-and-white film style",
    "with flapper dancers and jazz musicians", "beside a roaring forge fire",
    "with togas and marble columns", "clad in powdered wigs and silk robes",
    "surrounded by candle smoke", "with analog monitors and engineers in retro gear"
]

# ──────────────────────────────────────────────────────────────────────────────
# DEFINITIONS FOR NEGATIVE (“Non-Past”) PROMPTS
# ──────────────────────────────────────────────────────────────────────────────

modern_settings = [
    "futuristic city skyline", "neon-lit cyberpunk alley",
    "hyperrealistic portrait of a Bengal tiger", "sleek concept car speeding down a wet highway",
    "modern open-plan kitchen with stainless steel appliances", "minimalist Scandinavian living room interior",
    "aerial shot of a windswept beach at sunset", "drone’s eye view of a mountain lake at dawn",
    "contemporary art gallery under LED spotlights", "high-tech robotics laboratory with holographic displays",
    "close-up of a hummingbird feeding on a flower", "photorealistic render of a Shiba Inu puppy with VR headset",
    "a modern rooftop garden at twilight with city lights", "surreal floating island with luminous bioluminescent plants",
    "a glass-walled office tower reflecting sunset hues", "macro shot of a droplet on a green leaf in a rainforest",
    "futuristic humanoid robot serving coffee in a café", "photorealistic coral reef teeming with fish",
    "cyberpunk netrunner hacking a neon interface", "sleek drone flying over neon skyscrapers",
    "modern music festival stage with LED lasers", "photorealistic golden retriever playing in a yard",
    "digital painting of a starship entering warp speed", "high-resolution satellite view of a desert canyon",
    "minimalistic black-and-white geometric wallpaper design"
]

modern_modifiers = [
    "under soft studio lighting", "with vibrant neon reflections",
    "against a cloudy sky background", "under dynamic stormy clouds",
    "with bokeh city lights in the distance", "on a rainy night with reflections on pavement",
    "surrounded by floating holograms", "with bioluminescent flora glowing around",
    "against a backdrop of neon grids", "in ultra-wide panoramic view"
]


def generate_500_prompts():
    """
    Returns a DataFrame with 500 rows:
      - 250 rows labeled `1` (past references)
      - 250 rows labeled `0` (no past references)
    """
    pos_texts = []
    for _ in range(250):
        era = random.choice(historical_periods)
        mod = random.choice(historical_modifiers)
        pos_texts.append(f"A {era} scene {mod}.")

    neg_texts = []
    for _ in range(250):
        setting = random.choice(modern_settings)
        mod = random.choice(modern_modifiers)
        neg_texts.append(f"A {setting} {mod}.")

    texts = pos_texts + neg_texts
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)

    # Shuffle the combined list
    combined = list(zip(texts, labels))
    random.seed(123)
    random.shuffle(combined)
    texts_shuffled, labels_shuffled = zip(*combined)

    df = pd.DataFrame({
        "text": texts_shuffled,
        "label": labels_shuffled
    })
    return df


if __name__ == "__main__":
    df_new = generate_500_prompts()
    df_new.to_csv("new_500_prompts.csv", index=False)
    print("Wrote 500 new prompts (250 past=1, 250 non-past=0) → new_500_prompts.csv")
