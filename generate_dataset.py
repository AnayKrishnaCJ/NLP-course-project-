"""Synthetic question-type dataset generator (TREC-style coarse classes).

Every question is produced from a template + slot fillers, so the data is
SYNTHETIC. Each row records the template id it came from, which lets the
evaluation script hold out whole templates (a much harder, more honest test
than a random split).

Classes (coarse TREC taxonomy): ABBR, DESC, ENTY, HUM, LOC, NUM
"""
import csv
import random
import sys
from pathlib import Path

SEED = 42

ABBRS = ["NASA", "UNESCO", "HTML", "GPU", "FIFA", "UNICEF", "ISRO", "WHO", "CPU", "DNA",
         "RAM", "NATO", "OPEC", "PDF", "SQL", "IMF", "USB", "LASER", "RADAR", "SIM",
         "ATM", "GDP", "NGO", "VPN", "HTTP", "CEO", "AIDS", "ASEAN", "UPI", "ISBN"]
PHRASES = ["Central Processing Unit", "World Trade Organization", "Random Access Memory",
           "Artificial Intelligence", "Portable Document Format", "Gross Domestic Product",
           "Structured Query Language", "Virtual Private Network", "Chief Executive Officer",
           "Automated Teller Machine", "Natural Language Processing", "Graphics Processing Unit",
           "Reserve Bank of India", "Indian Institute of Technology", "Unified Payments Interface"]
PROCESSES = ["a refrigerator", "photosynthesis", "a vaccine", "a search engine", "an airplane wing",
             "the stock market", "a nuclear reactor", "GPS", "a microwave oven", "the human heart",
             "a neural network", "blockchain", "a solar panel", "an electric motor", "WiFi",
             "the immune system", "a hydraulic jack", "a camera lens", "a compiler", "recycling"]
CONCEPTS = ["inflation", "gravity", "democracy", "machine learning", "entropy", "supply and demand",
            "natural selection", "the greenhouse effect", "recursion", "photosynthesis",
            "quantum entanglement", "cloud computing", "the water cycle", "plate tectonics",
            "overfitting", "encryption", "osmosis", "relativity", "compound interest", "fermentation"]
WORDS = ["serendipity", "ephemeral", "ubiquitous", "paradigm", "pragmatic", "ambiguous", "resilient",
         "nostalgia", "eloquent", "meticulous", "ostracize", "benevolent", "candid", "diligent",
         "inevitable", "skeptic", "tenacious", "verbose", "zealous", "altruism"]
PHENOMENA = ["earthquakes", "rainbows", "tides", "thunder", "inflation", "solar eclipses", "volcanic eruptions",
             "the northern lights", "hurricanes", "rust", "fever", "droughts", "traffic jams", "tsunamis",
             "ocean currents", "static electricity", "hiccups", "echoes", "seasons", "fog"]
PAIRS = [("a virus", "a bacterium"), ("weather", "climate"), ("RAM", "ROM"), ("a list", "a tuple"),
         ("speed", "velocity"), ("TCP", "UDP"), ("mitosis", "meiosis"), ("a crocodile", "an alligator"),
         ("stocks", "bonds"), ("AC", "DC"), ("precision", "recall"), ("a comet", "an asteroid"),
         ("a stack", "a queue"), ("heat", "temperature"), ("an epidemic", "a pandemic")]
CATEGORIES = ["animal", "bird", "fruit", "metal", "gemstone", "musical instrument", "sport", "vegetable",
              "programming language", "currency", "dance form", "spice", "flower", "tree", "fabric",
              "board game", "planet", "gas", "language", "dish"]
ATTRS = ["is the fastest", "is the largest", "is the rarest", "is the oldest", "is the most popular",
         "is the heaviest", "is native to Asia", "is used in cooking", "is the smallest", "lives longest"]
MATERIALS = ["bridges", "kitchen knives", "coins", "jewelry", "airplanes", "electrical wires", "ships",
             "mirrors", "pipes", "roofs", "windows", "rockets"]
ROLES = ["prime minister", "president", "astronaut", "Nobel laureate", "chess champion", "mathematician",
         "poet", "physicist", "architect", "explorer", "inventor", "general", "novelist", "scientist",
         "cricketer", "painter"]
INVENTIONS = ["telephone", "light bulb", "printing press", "steam engine", "radio", "airplane",
              "World Wide Web", "penicillin", "telescope", "transistor", "camera", "vaccine",
              "calculator", "dynamite", "typewriter"]
WORKS = ["Hamlet", "The Odyssey", "Pride and Prejudice", "The Republic", "Gitanjali", "Don Quixote",
         "War and Peace", "The Jungle Book", "Frankenstein", "Macbeth", "The Prince", "Walden",
         "Ulysses", "Emma", "Beloved"]
PRIZES = ["Nobel Prize in Physics", "Turing Award", "Booker Prize", "Fields Medal", "Pulitzer Prize",
          "Olympic gold medal", "Grammy Award", "Oscar for Best Director", "Padma Bhushan", "Abel Prize"]
YEARS = [str(y) for y in range(1901, 2024, 3)]
LANDMARKS = ["the Eiffel Tower", "Mount Everest", "the Taj Mahal", "the Colosseum", "Machu Picchu",
             "the Great Wall", "Niagara Falls", "the Louvre", "Angkor Wat", "Stonehenge", "Uluru",
             "the Sahara Desert", "Lake Baikal", "the Golden Gate Bridge", "Petra", "Victoria Falls",
             "the Kremlin", "Mount Fuji", "the Amazon River", "Big Ben"]
CITIES = ["Nairobi", "Kyoto", "Lisbon", "Toronto", "Mumbai", "Cairo", "Oslo", "Lima", "Hanoi", "Perth",
          "Prague", "Seoul", "Dublin", "Havana", "Kochi", "Chennai", "Zurich", "Auckland", "Madrid", "Dhaka"]
COUNTRIES = ["Kenya", "Japan", "Portugal", "Canada", "India", "Egypt", "Norway", "Peru", "Vietnam",
             "Australia", "Czechia", "South Korea", "Ireland", "Cuba", "Switzerland", "New Zealand",
             "Spain", "Bangladesh", "Brazil", "Chile"]
EVENTS_LOC = ["the next Olympics", "the World Economic Forum", "the Cannes Film Festival", "Wimbledon",
              "the Rio Carnival", "the Oktoberfest", "the Kumbh Mela", "the Tour de France",
              "the Nobel Prize ceremony", "the Edinburgh Fringe"]
THINGS_FIND = ["the best street food", "a good bookstore", "the oldest temple", "a quiet beach",
               "the main railway station", "a science museum", "fresh seafood", "a public library",
               "the tallest building", "cheap electronics"]
COUNT_THINGS = [("days", "a leap year"), ("players", "a cricket team"), ("bones", "the adult human body"),
                ("states", "India"), ("strings", "a standard guitar"), ("planets", "the solar system"),
                ("keys", "a piano"), ("chapters", "the Bhagavad Gita"), ("time zones", "Russia"),
                ("squares", "a chessboard"), ("teeth", "an adult human"), ("sides", "a hexagon"),
                ("moons", "Jupiter"), ("letters", "the English alphabet"), ("members", "the UN Security Council")]
EVENTS_NUM = ["World War II end", "the Moon landing", "the French Revolution begin", "the Berlin Wall fall",
              "India gain independence", "the first iPhone launch", "the Titanic sink",
              "the Internet become public", "the printing press appear", "the Roman Empire fall",
              "the Eiffel Tower open", "the Panama Canal open", "the first Olympics take place",
              "the Suez Canal open", "the Great Fire of London occur"]
MEASURES = [("tall", "Mount Kilimanjaro"), ("long", "the Nile"), ("deep", "the Mariana Trench"),
            ("heavy", "an adult elephant"), ("old", "the Earth"), ("wide", "the Amazon River"),
            ("fast", "a cheetah"), ("hot", "the surface of the Sun"), ("high", "the Burj Khalifa"),
            ("long", "a football pitch"), ("heavy", "a blue whale"), ("old", "the universe")]
PRICES = ["a movie ticket", "a kilogram of rice", "a cup of coffee", "a new laptop", "a train ticket to Delhi",
          "a liter of petrol", "a passport renewal", "a domestic flight", "a gym membership", "a bicycle"]
POP = ["Tokyo", "Kerala", "Iceland", "Brazil", "Mumbai", "Nigeria", "Sweden", "Delhi", "Canada", "Singapore"]
DIST = [("Kochi", "Mumbai"), ("Earth", "the Moon"), ("London", "Paris"), ("Delhi", "Agra"),
        ("Tokyo", "Osaka"), ("Cairo", "Nairobi"), ("Chennai", "Bangalore"), ("Sydney", "Perth"),
        ("Earth", "Mars"), ("Rome", "Athens")]

# template id -> (class, template, slot-generator). Slot generators return dict of fillers.
def _pick(rng, seq):
    return rng.choice(seq)

def build_templates():
    T = []  # (cls, template_id, fn(rng)->str)
    def add(cls, fn):
        T.append((cls, f"{cls}_{sum(1 for t in T if t[0]==cls):02d}", fn))
    # ---- ABBR
    add("ABBR", lambda r: f"What does {_pick(r, ABBRS)} stand for?")
    add("ABBR", lambda r: f"What is the full form of {_pick(r, ABBRS)}?")
    add("ABBR", lambda r: f"What is the abbreviation for {_pick(r, PHRASES)}?")
    add("ABBR", lambda r: f"Expand the acronym {_pick(r, ABBRS)}.")
    add("ABBR", lambda r: f"{_pick(r, ABBRS)} is short for what?")
    add("ABBR", lambda r: f"Can you spell out {_pick(r, ABBRS)} for me?")
    add("ABBR", lambda r: f"What do the letters {_pick(r, ABBRS)} mean?")
    add("ABBR", lambda r: f"What is the acronym of {_pick(r, PHRASES)}?")
    # ---- DESC
    add("DESC", lambda r: f"How does {_pick(r, PROCESSES)} work?")
    add("DESC", lambda r: f"Explain {_pick(r, CONCEPTS)}.")
    add("DESC", lambda r: f"What is the meaning of the word {_pick(r, WORDS)}?")
    add("DESC", lambda r: f"What causes {_pick(r, PHENOMENA)}?")
    add("DESC", lambda r: (lambda p: f"What is the difference between {p[0]} and {p[1]}?")(_pick(r, PAIRS)))
    add("DESC", lambda r: f"Why do {_pick(r, PHENOMENA)} happen?")
    add("DESC", lambda r: f"Define {_pick(r, CONCEPTS)}.")
    add("DESC", lambda r: f"Describe how {_pick(r, PROCESSES)} operates.")
    # ---- ENTY
    add("ENTY", lambda r: f"What {_pick(r, CATEGORIES)} {_pick(r, ATTRS)}?")
    add("ENTY", lambda r: f"Which {_pick(r, CATEGORIES)} {_pick(r, ATTRS)}?")
    add("ENTY", lambda r: f"What material is commonly used to make {_pick(r, MATERIALS)}?")
    add("ENTY", lambda r: f"Name a {_pick(r, CATEGORIES)} that {_pick(r, ATTRS).replace('is ', 'is ', 1)}.")
    add("ENTY", lambda r: f"What kind of {_pick(r, CATEGORIES)} is best for beginners?")
    add("ENTY", lambda r: f"What is the name of the {_pick(r, CATEGORIES)} {_pick(r, ATTRS)[3:] if False else 'found in tropical regions'}?")
    add("ENTY", lambda r: f"What type of {_pick(r, CATEGORIES)} do people usually prefer?")
    add("ENTY", lambda r: f"Which {_pick(r, CATEGORIES)} would you pick for a gift?")
    # ---- HUM
    add("HUM", lambda r: f"Who invented the {_pick(r, INVENTIONS)}?")
    add("HUM", lambda r: f"Who wrote {_pick(r, WORKS)}?")
    add("HUM", lambda r: f"Who was the first {_pick(r, ROLES)} of the modern era?")
    add("HUM", lambda r: f"Which {_pick(r, ROLES)} won the {_pick(r, PRIZES)} in {_pick(r, YEARS)}?")
    add("HUM", lambda r: f"Who is the greatest {_pick(r, ROLES)} in history?")
    add("HUM", lambda r: f"Name the person who discovered the {_pick(r, INVENTIONS)}.")
    add("HUM", lambda r: f"Which famous {_pick(r, ROLES)} was born in {_pick(r, CITIES)}?")
    add("HUM", lambda r: f"Who composed the opening speech about {_pick(r, CONCEPTS)}?")
    # ---- LOC
    add("LOC", lambda r: f"Where is {_pick(r, LANDMARKS)} located?")
    add("LOC", lambda r: f"In which country is {_pick(r, CITIES)}?")
    add("LOC", lambda r: f"What is the capital of {_pick(r, COUNTRIES)}?")
    add("LOC", lambda r: f"Which city hosts {_pick(r, EVENTS_LOC)}?")
    add("LOC", lambda r: f"Where can I find {_pick(r, THINGS_FIND)} in {_pick(r, CITIES)}?")
    add("LOC", lambda r: f"Which continent is {_pick(r, COUNTRIES)} part of?")
    add("LOC", lambda r: f"Point me to the location of {_pick(r, LANDMARKS)}.")
    add("LOC", lambda r: f"What state or province is {_pick(r, CITIES)} in?")
    # ---- NUM
    add("NUM", lambda r: (lambda p: f"How many {p[0]} are there in {p[1]}?")(_pick(r, COUNT_THINGS)))
    add("NUM", lambda r: f"In what year did {_pick(r, EVENTS_NUM)}?")
    add("NUM", lambda r: (lambda p: f"How {p[0]} is {p[1]}?")(_pick(r, MEASURES)))
    add("NUM", lambda r: f"How much does {_pick(r, PRICES)} cost?")
    add("NUM", lambda r: f"What is the population of {_pick(r, POP)}?")
    add("NUM", lambda r: (lambda p: f"How far is {p[0]} from {p[1]}?")(_pick(r, DIST)))
    add("NUM", lambda r: f"When did {_pick(r, EVENTS_NUM)}?")
    add("NUM", lambda r: (lambda p: f"What is the height of {p[1]} in meters?")(_pick(r, MEASURES)))
    return T

PREFIXES = ["", "", "", "", "Quick question: ", "Can you tell me ", "I was wondering, ", "Please answer: ", "Hey, "]

def noise(text, rng):
    """Light surface noise so models cannot rely only on the wh-word."""
    p = rng.choice(PREFIXES)
    if p in ("Can you tell me ", "I was wondering, "):
        text = text[0].lower() + text[1:]
    text = p + text
    roll = rng.random()
    if roll < 0.15:
        text = text.lower()
    if rng.random() < 0.15:
        text = text.rstrip("?.")
    if rng.random() < 0.05 and len(text) > 8:  # one character swap typo
        i = rng.randrange(1, len(text) - 2)
        text = text[:i] + text[i + 1] + text[i] + text[i + 2:]
    return text

def generate(per_class=700, seed=SEED):
    rng = random.Random(seed)
    templates = build_templates()
    by_cls = {}
    for cls, tid, fn in templates:
        by_cls.setdefault(cls, []).append((tid, fn))
    rows, seen = [], set()
    for cls, tl in by_cls.items():
        attempts, made = 0, 0
        while made < per_class and attempts < per_class * 60:
            attempts += 1
            tid, fn = rng.choice(tl)
            q = noise(fn(rng), rng)
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append((q, cls, tid))
            made += 1
    rng.shuffle(rows)
    return rows

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_questions.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = generate()
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "label", "template_id"])
        w.writerows(rows)
    from collections import Counter
    print(f"wrote {len(rows)} rows to {out}")
    print(Counter(r[1] for r in rows))
