#!/usr/bin/env python
"""Build prompts/full.csv (~200 prompts) = the 60 pilot prompts + 140 more.

composition and text are templated so every prompt has a machine-checkable
`expected`; the other four groups are curated lists. Deterministic (seeded), so
re-running gives the same file. Only move to full.csv after the pilot looks right.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

HERE = Path(__file__).parent
rng = random.Random(7)

NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
OBJECTS = ["cup", "apple", "book", "chair", "balloon", "bottle", "orange", "teddy bear", "pumpkin", "traffic cone",
           "rubber duck", "candle", "shoe", "vase", "donut", "tennis ball", "backpack", "lemon", "bicycle", "bird"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "white", "black", "pink"]
REFS = ["a wooden table", "a gray sofa", "a window", "a brick wall", "a blue plate", "a cardboard box", "a park bench", "a red car"]
RELATIONS = ["to the left of", "to the right of", "on top of", "in front of", "next to", "below"]


def composition(n: int) -> list[dict]:
    out, seen = [], set()
    while len(out) < n:
        k = rng.choice([1, 2, 3, 4, 5, 6]); obj = rng.choice(OBJECTS); col = rng.choice(COLORS)
        rel = rng.choice(RELATIONS); ref = rng.choice(REFS)
        plural = obj + ("s" if k > 1 and not obj.endswith("s") else "")
        key = (k, obj, col, rel, ref)
        if key in seen:
            continue
        seen.add(key)
        prompt = f"{NUM[k]} {col} {plural} {rel} {ref}, photo"
        out.append(dict(group="composition", prompt=prompt, expected=f"count={k};object={col} {obj};relation={plural} {rel} {ref}"))
    return out


SURFACES = ["a storefront sign", "a chalkboard", "a neon sign", "a wooden road sign", "a poster on a wall", "a coffee mug", "a t-shirt",
            "a cardboard box", "a hand-painted banner", "a license plate", "a book cover", "a street sign", "a cake", "a vintage tin", "a billboard"]
PHRASES = ["GOOD MORNING", "NO PARKING", "FRESH BREAD", "EXIT", "WELCOME HOME", "LOVE", "HOT COFFEE", "STOP", "HAPPY BIRTHDAY", "FREE WIFI",
           "DANGER", "BOOKS", "ICE CREAM", "GARAGE SALE", "THANK YOU", "KEEP OUT", "PIZZA", "OPEN", "BUS STOP", "LEMONADE 1 DOLLAR",
           "ROOM 101", "GATE 7", "MUSIC", "DREAM BIG", "SOLD OUT", "TEA TIME", "HAPPY NEW YEAR", "BAKERY", "FISH MARKET", "QUIET PLEASE"]


def text(n: int) -> list[dict]:
    out, seen = [], set()
    while len(out) < n:
        s, p = rng.choice(SURFACES), rng.choice(PHRASES)
        if p in seen:
            continue
        seen.add(p)
        verb = "that says" if rng.random() < 0.6 else "with the words"
        out.append(dict(group="text", prompt=f'{s} {verb} "{p}", photo', expected=f"text={p}"))
    return out


HANDS_FACES = [
    "close-up of a hand holding a smartphone, thumb on the screen", "a baby's hand gripping an adult's finger, macro photo",
    "a mechanic's greasy hands holding a wrench, close-up", "a close-up of hands tying shoelaces", "an open palm holding three coins, close-up",
    "a hand giving a thumbs up against a blue sky", "two hands cupping water from a stream, close-up", "a surgeon's gloved hands, close-up",
    "a close-up of a hand typing on a laptop keyboard", "a hand holding a paintbrush covered in blue paint, close-up",
    "portrait of a woman with curly red hair and green eyes, natural light", "a close-up portrait of an old man with a white beard laughing",
    "headshot of a smiling businesswoman, gray background", "a teenage boy's face with braces, smiling, school photo style",
    "a close-up of a woman's face with rain drops on her skin", "portrait of a man with a shaved head and a nose ring, studio light",
    "a toddler's face covered in chocolate, close-up", "a close-up portrait of a monk with a calm expression", "a grandmother's face in soft window light, candid",
    "a close-up of a woman's face with eyes closed and freckles",
]
PRODUCT = [
    "a blue enamel cast-iron pot on a white kitchen counter, product photography", "a pair of black over-ear headphones floating on a light gray background",
    "a ceramic pour-over coffee set on a walnut tray, soft light", "a bottle of olive oil with a cork stopper on a linen cloth",
    "a mechanical keyboard with pastel keycaps on a clean desk", "a canvas tote bag hanging on a wooden hook, studio photo",
    "a glass jar of honey with a wooden dipper, backlit", "a pair of round tortoiseshell sunglasses on a white marble slab",
    "a chrome desk lamp on a dark wooden desk, product shot", "a stack of pancakes on a white plate with maple syrup, food photography",
    "a leather notebook with a brass pen on a slate background", "a bottle of craft beer with condensation on a wooden bar",
    "a wooden chess set on a glass table, close-up", "a white electric toothbrush standing on a bathroom shelf, soft light",
    "a navy blue wool sweater folded on a white background", "a gold ring with a small emerald on black velvet, macro",
    "a red kettlebell on a gym floor, dramatic side light", "a glass terrarium with succulents on a windowsill",
    "a minimalist white alarm clock on a nightstand", "a pair of hiking boots on a mossy rock, product photography",
]
STYLIZED = [
    "a whale swimming through clouds, surreal digital painting", "a cottage in a meadow, Studio Ghibli style anime background",
    "a samurai under a cherry tree, sumi-e ink painting", "a vintage travel poster of a mountain railway, art deco",
    "a hummingbird, stained glass window style", "a steam train crossing a bridge, linocut print in two colors",
    "a street cafe in Paris, loose gouache painting", "a tiger's face, low-poly 3D render", "a sailing ship in a storm, dramatic oil painting, 19th century style",
    "a bowl of ramen, cute kawaii sticker illustration", "a wizard's tower at sunset, isometric pixel art", "a bicycle, continuous single-line drawing",
    "a koi pond, paper cut-out layered art", "a cowboy at dusk, 1960s movie poster illustration", "a forest spirit, Alphonse Mucha style poster",
    "a hot air balloon festival, children's book illustration", "a lion, geometric mosaic tiles", "a rainy city street, cyberpunk neon anime style",
    "a bouquet of sunflowers, Van Gogh style thick brushstrokes", "a sleeping dragon, medieval illuminated manuscript style",
]
DENSE = [
    "A greenhouse interior at noon: rows of terracotta pots with orchids in white and purple, a brass watering can on a mossy stone floor, a hanging fern beside a cracked glass pane, a sleeping ginger cat on a wicker chair, and sunbeams full of dust.",
    "A street food cart in Mexico City: a vendor in a straw hat chopping cilantro, a steel griddle with sizzling tacos al pastor, a spinning trompo of meat with a pineapple on top, red and green salsa bowls, plastic stools in bright colors, and a crowd under a yellow awning.",
    "A ship captain's cabin: a brass telescope on a map covered desk, an oil lamp swinging from a beam, a parrot on a perch, a sextant in an open velvet case, a porthole showing a stormy sea, and a wooden chest with iron bands.",
    "A winter train station platform: a green steam locomotive with snow on its roof, a woman in a red coat holding a brown suitcase, a round clock reading ten past three, a newspaper kiosk with magazines, pigeons near a bench, and steam drifting under iron arches.",
    "A kid's bedroom at bedtime: glow-in-the-dark stars on a blue ceiling, a bunk bed with dinosaur sheets, a toy rocket on the floor, a bookshelf with a fish tank on top, a nightlight shaped like a moon, and a golden puppy asleep on a rug.",
    "A Moroccan courtyard: a tiled fountain in blue and white zellige, orange trees in large clay pots, a man in a white djellaba pouring mint tea, brass lanterns casting patterned shadows, arched doorways with carved cedar, and a cat on the rim of the fountain.",
    "A scientist's lab bench: a microscope beside racks of colored test tubes, a laptop showing a graph, a notebook with hand-drawn diagrams, a coffee mug with a periodic table print, purple gloves, and a window looking out at a snowy campus.",
    "An autumn farmers market: wooden crates of red and green apples, a chalkboard price sign, a man in a flannel shirt weighing pumpkins on an old brass scale, jars of honey glowing in the sun, a golden retriever on a leash, and maple leaves on the ground.",
    "A beach bonfire at dusk: five friends sitting on driftwood logs, one playing a ukulele, marshmallows on sticks, a cooler with a surfboard leaning on it, a dog digging in the sand, waves breaking in the distance, and a purple and orange sky.",
    "A busy ramen shop kitchen: a cook in a white headband lifting noodles with a bamboo strainer, steam rising from large pots, rows of ceramic bowls, a wall of handwritten menu cards, red paper lanterns, and customers' hands visible across the counter.",
    "A mountain observatory at night: a white domed telescope open to a sky full of stars, an astronomer in a parka with a thermos, a red headlamp, snow on the ground, a wooden sign with an altitude marker, and the Milky Way arching overhead.",
    "A vintage barber shop: a red leather chair, a striped pole outside the window, a barber with a handlebar mustache holding scissors, shelves of glass tonic bottles, a black-and-white tiled floor, and a sleeping bulldog by the door.",
    "A flooded rice terrace at sunrise: curved paddies reflecting pink clouds, a farmer in a conical hat leading a water buffalo, a wooden hut on stilts, white egrets in the water, mist in the valley, and palm trees on the ridge.",
    "An antique toy shop window: a tin wind-up robot, a wooden rocking horse with a red saddle, a porcelain doll in a lace dress, a model sailboat, a jar of glass marbles, a stack of board games, and warm lamplight on a rainy evening street.",
    "A jazz club stage: a double bass player in a fedora, a trumpet catching a blue spotlight, a drummer behind a sparkling gold kit, a singer in a sequined dress at a vintage microphone, cocktail glasses on small round tables, and smoke in the air.",
    "A medieval blacksmith's forge: a bearded smith hammering a glowing horseshoe on an anvil, sparks flying, a bellows beside a stone hearth, hanging tongs and hammers, a bucket of water, a horse waiting at the open door, and a thatched roof.",
    "A Venetian canal in the morning: a gondola with a striped pole, laundry lines between ochre buildings, a woman opening green shutters, a cat on a stone bridge, flower boxes with red geraniums, and a delivery boat stacked with crates of vegetables.",
    "A space station greenhouse module: rows of lettuce under purple grow lights, an astronaut in a blue jumpsuit holding a tablet, floating water droplets, Earth visible through a round window, a sticker-covered laptop, and coiled cables along the wall.",
    "A wedding banquet table outdoors: a long wooden table under string lights, eucalyptus garlands, tall white candles, a three-tier cake with fresh berries, name cards in gold ink, olive trees, and a sunset over vineyards.",
    "A detective's office in the 1940s: venetian blind shadows on a desk, a black rotary phone, a typewriter with a page half typed, a fedora on a coat rack, a whiskey glass, a corkboard with photos and red string, and rain on the window.",
]


def main():
    pilot = list(csv.DictReader(open(HERE / "pilot.csv", newline="", encoding="utf-8")))
    rows = list(pilot)
    counters = {g: 10 for g in ("composition", "text", "hands_faces", "product", "stylized", "dense")}
    prefix = {"composition": "comp", "text": "text", "hands_faces": "hf", "product": "prod", "stylized": "sty", "dense": "dense"}

    def add(items):
        for it in items:
            counters[it["group"]] += 1
            rows.append(dict(id=f"{prefix[it['group']]}_{counters[it['group']]:02d}", **it))

    add(composition(30))
    add(text(30))
    add([dict(group="hands_faces", prompt=p, expected="") for p in HANDS_FACES])
    add([dict(group="product", prompt=p, expected="") for p in PRODUCT])
    add([dict(group="stylized", prompt=p, expected="") for p in STYLIZED])
    add([dict(group="dense", prompt=p, expected="") for p in DENSE])

    with open(HERE / "full.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "group", "prompt", "expected"])
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(f"wrote {HERE / 'full.csv'}: {len(rows)} prompts", dict(Counter(r['group'] for r in rows)))


if __name__ == "__main__":
    main()
