"""NEPALI — SYNTHETIC REASONING DATA GENERATOR
Phase 3: comparative / basic reasoning fine-tuning

Design:
- 20,000 train / 2,000 validation / 3,000 test by default.
- Direct comparison, reverse comparison, equality, 2-hop transitivity,
  3-hop transitivity, and ordering/ranking.
- Nepali-only prompts in Devanagari script.
- Ground-truth numeric values are stored in metadata, not exposed in prompts.
- Train/validation and test use disjoint template-ID families.
- Exact prompt uniqueness is enforced globally across all splits.
- Entity overlap across splits is intentional; this tests reasoning rather than
  unseen-name recognition.
- Statements are shuffled.
- For transitive tasks, entity order in the question is independently shuffled
  from the canonical reasoning chain to reduce positional shortcuts.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

LANGUAGE = "ne"
LANGUAGE_NAME = "Nepali"

DEFAULT_TRAIN_SIZE = 20_000
DEFAULT_VAL_SIZE = 2_000
DEFAULT_TEST_SIZE = 3_000

TASK_WEIGHTS = {
    "direct_comparison": 0.20,
    "reverse_comparison": 0.15,
    "equality": 0.10,
    "transitive_2": 0.20,
    "transitive_3": 0.20,
    "ordering": 0.15,
}

PERSON_ATTRIBUTES = ["height", "age", "weight"]
OBJECT_ATTRIBUTES = ["price", "quantity"]


ATTRIBUTES = {
    "height": {
        "entity_type": "person",
        "noun": "उचाइ",
        "unit": "सेन्टिमिटर",
        "min": 145,
        "max": 195,
    },
    "age": {
        "entity_type": "person",
        "noun": "उमेर",
        "unit": "वर्ष",
        "min": 18,
        "max": 65,
    },
    "weight": {
        "entity_type": "person",
        "noun": "तौल",
        "unit": "किलोग्राम",
        "min": 45,
        "max": 100,
    },
    "price": {
        "entity_type": "object",
        "noun": "मूल्य",
        "unit": "रुपैयाँ",
        "min": 100,
        "max": 5000,
    },
    "quantity": {
        "entity_type": "object",
        "noun": "परिमाण",
        "unit": "एकाइ",
        "min": 1,
        "max": 100,
    },
}


# =============================================================================
# ENTITY POOLS
# =============================================================================

# Common Nepali names written in Devanagari.
PEOPLE = [
    "अभिषेक", "अजय", "अमित", "अनिल", "अनुप", "अनुज", "अनुराग", "अशोक",
    "आकाश", "आदित्य", "आनन्द", "आयुष", "अरुण", "अर्जुन", "अविनाश",
    "आशिष", "बिनोद", "बिबेक", "बिकास", "बिमल", "बिराज", "दीपक",
    "दिनेश", "दिपेश", "दिपेन्द्र", "धीरज", "गोपाल", "गौरव", "हरि",
    "हिमाल", "हेमन्त", "इन्द्र", "जनक", "जीवन", "कपिल", "करण",
    "केशव", "कृष्ण", "कुमार", "लक्ष्मण", "माधव", "मनोज",
    "महेश", "मिलन", "मुकेश", "नरेश", "नविन", "निरज", "निरञ्जन",
    "प्रकाश", "प्रदीप", "प्रविन", "प्रेम", "राजेश", "राजेन्द्र", "राहुल",
    "रमेश", "रमित", "रवि", "रोहन", "रोहित", "सागर", "सन्दीप", "सञ्जय",
    "सन्तोष", "सुमन", "सुशील", "सुरज", "सुरेन्द्र", "विजय", "विक्रम",
    "विनय", "विशाल", "विवेक", "योगेश", "युवराज", "उमेश", "उत्तम",
]

# Common everyday objects in Nepali.
OBJECTS = [
    "कुर्सी", "टेबल", "मेज", "पङ्खा", "सोफा", "ऐना", "झ्याल", "ढोका",
    "घडी", "गिलास", "कप", "थाल", "चक्कु", "चम्चा", "काँटा", "सिरानी",
    "कम्बल", "पर्दा", "बाल्टिन", "साबुन", "तौलिया", "काइँयो", "जुत्ता",
    "झोला", "कागज", "किताब", "कलम", "कापी", "कैंची", "टोपी", "चश्मा",
    "बोतल", "बक्स", "टर्च", "छाता", "चाबी", "रिमोट", "क्याल्कुलेटर",
    "पेन्सिल", "पेन्सिल बक्स", "मोबाइल", "कम्प्युटर", "ल्यापटप",
    "क्यामेरा", "रेडियो", "स्पिकर", "हेडफोन", "चार्जर", "घैँटो",
    "भाँडो", "कुकर", "केतली", "चुलो", "फ्रिज", "अल्मारी", "दराज",
    "कार्पेट", "बत्ती", "मग", "डब्बा", "खेलौना", "छोटो टेबल",
    "काठको बाकस", "पानीको बोतल", "किताबको झोला", "स्टेसनरी बक्स",
    "हात ऐना", "टेबल ल्याम्प", "भित्ते घडी", "सानो स्पिकर",
]


# =============================================================================
# TEMPLATE SETS
# =============================================================================

# Direct comparison — train/validation.
DIRECT_GREATER_TRAIN = [
    "{a}को {noun} {b}को भन्दा बढी छ।",
    "{b}को तुलनामा {a}को {noun} बढी छ।",
    "{a}को {noun} {b}भन्दा धेरै छ।",
    "{b}भन्दा {a}को {noun} बढी रहेको छ।",
]

DIRECT_LOWER_TRAIN = [
    "{a}को {noun} {b}को भन्दा कम छ।",
    "{b}को तुलनामा {a}को {noun} कम छ।",
    "{a}को {noun} {b}भन्दा थोरै छ।",
    "{b}भन्दा {a}को {noun} कम रहेको छ।",
]

# Held-out direct surface forms for test.
DIRECT_GREATER_TEST = [
    "{a}को {noun} {b}भन्दा बढी रहेको छ।",
    "{b}को {noun} भन्दा {a}को {noun} धेरै छ।",
    "{a}को {noun} बढी हुँदा {b}को {noun} कम छ।",
]

DIRECT_LOWER_TEST = [
    "{a}को {noun} {b}भन्दा कम रहेको छ।",
    "{b}को {noun} भन्दा {a}को {noun} थोरै छ।",
    "{a}को {noun} कम हुँदा {b}को {noun} बढी छ।",
]


# Reverse comparison.
REVERSE_TRAIN = [
    "{low}को {noun} {high}को भन्दा कम छ।",
    "{high}को तुलनामा {low}को {noun} कम छ।",
    "{low}को {noun} {high}भन्दा थोरै छ।",
    "{high}भन्दा {low}को {noun} कम रहेको छ।",
]

REVERSE_TEST = [
    "{low}को {noun} कम हुँदा {high}को {noun} बढी छ।",
    "{low}भन्दा {high}को {noun} धेरै रहेको छ।",
    "{high}को {noun} {low}को भन्दा बढी छ।",
]


# Equality.
EQUALITY_TRAIN = [
    "{a} र {b}को {noun} बराबर छ।",
    "{a} र {b} दुवैको {noun} समान छ।",
    "{a}को {noun} {b}को {noun} जत्तिकै छ।",
]

EQUALITY_TEST = [
    "{a} र {b}को {noun} उस्तै छ।",
    "{a}को {noun} र {b}को {noun} मा कुनै फरक छैन।",
    "{a} र {b} दुवैको {noun} समान रहेको छ।",
]


# Transitive 2-hop.
TRANSITIVE_2_TRAIN = [
    "{a}को {noun} {b}को भन्दा बढी छ।",
    "{b}को {noun} {c}को भन्दा बढी छ।",
]

TRANSITIVE_2_ALT_TRAIN = [
    "{a}को {noun} {b}भन्दा धेरै छ।",
    "{c}को {noun} {b}को भन्दा कम छ।",
]

TRANSITIVE_2_TEST = [
    "{a}को {noun} {b}भन्दा बढी रहेको छ।",
    "{c}को {noun} {b}भन्दा कम छ।",
]

TRANSITIVE_2_ALT_TEST = [
    "{b}को {noun} {a}को भन्दा कम छ।",
    "{c}को तुलनामा {b}को {noun} बढी छ।",
]


# Transitive 3-hop.
TRANSITIVE_3_TRAIN = [
    "{a}को {noun} {b}को भन्दा बढी छ।",
    "{b}को {noun} {c}को भन्दा बढी छ।",
    "{c}को {noun} {d}को भन्दा बढी छ।",
]

TRANSITIVE_3_ALT_TRAIN = [
    "{a}को {noun} {b}भन्दा धेरै छ।",
    "{c}को {noun} {b}को भन्दा कम छ।",
    "{d}को {noun} {c}को भन्दा कम छ।",
]

TRANSITIVE_3_TEST = [
    "{a}को {noun} {b}भन्दा बढी रहेको छ।",
    "{c}को {noun} {b}भन्दा कम छ।",
    "{d}को {noun} {c}को भन्दा कम छ।",
]

TRANSITIVE_3_ALT_TEST = [
    "{b}को {noun} {a}को भन्दा कम छ।",
    "{d}को तुलनामा {c}को {noun} बढी छ।",
    "{d}को {noun} {c}भन्दा कम रहेको छ।",
]


# Ordering / ranking.
ORDERING_STATEMENT_TRAIN = [
    "{a}को {noun} {b}को भन्दा बढी छ।",
    "{b}को {noun} {c}को भन्दा बढी छ।",
    "{c}को {noun} {d}को भन्दा बढी छ।",
]

ORDERING_STATEMENT_ALT_TRAIN = [
    "{a}को {noun} {b}भन्दा धेरै छ।",
    "{c}को {noun} {b}को भन्दा कम छ।",
    "{d}को {noun} {c}को भन्दा कम छ।",
]

ORDERING_STATEMENT_TEST = [
    "{a}को {noun} {b}भन्दा बढी रहेको छ।",
    "{c}को {noun} {b}भन्दा कम छ।",
    "{d}को {noun} {c}को भन्दा कम छ।",
]

ORDERING_STATEMENT_ALT_TEST = [
    "{b}को {noun} {a}को भन्दा कम छ।",
    "{b}भन्दा {a}को {noun} बढी छ।",
    "{d}को {noun} {c}भन्दा कम रहेको छ।",
]


TEMPLATE_IDS = {
    "direct_greater_train": [
        "direct_greater_01", "direct_greater_02",
        "direct_greater_03", "direct_greater_04",
    ],
    "direct_lower_train": [
        "direct_lower_01", "direct_lower_02",
        "direct_lower_03", "direct_lower_04",
    ],
    "direct_greater_test": [
        "direct_greater_test_01", "direct_greater_test_02",
        "direct_greater_test_03",
    ],
    "direct_lower_test": [
        "direct_lower_test_01", "direct_lower_test_02",
        "direct_lower_test_03",
    ],
    "reverse_train": [
        "reverse_01", "reverse_02", "reverse_03", "reverse_04",
    ],
    "reverse_test": [
        "reverse_test_01", "reverse_test_02", "reverse_test_03",
    ],
    "equality_train": [
        "equality_01", "equality_02", "equality_03",
    ],
    "equality_test": [
        "equality_test_01", "equality_test_02", "equality_test_03",
    ],
    "transitive_2_train": [
        "transitive2_01", "transitive2_02",
    ],
    "transitive_2_test": [
        "transitive2_test_01", "transitive2_test_02",
    ],
    "transitive_3_train": [
        "transitive3_01", "transitive3_02", "transitive3_03",
    ],
    "transitive_3_test": [
        "transitive3_test_01", "transitive3_test_02", "transitive3_test_03",
    ],
    "ordering_train": [
        "ordering_01", "ordering_02", "ordering_03",
    ],
    "ordering_test": [
        "ordering_test_01", "ordering_test_02", "ordering_test_03",
    ],
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def choose_distinct_entities(pool: List[str], count: int) -> List[str]:
    """Sample distinct entities without replacement."""
    return random.sample(pool, count)


def choose_entity_type() -> str:
    """Choose person/object with equal probability."""
    return random.choice(["person", "object"])


def choose_attribute(entity_type: str) -> str:
    """Choose an attribute valid for the entity type."""
    if entity_type == "person":
        return random.choice(PERSON_ATTRIBUTES)
    return random.choice(OBJECT_ATTRIBUTES)


def get_pool(entity_type: str) -> List[str]:
    """Return the entity pool for a given entity type."""
    return PEOPLE if entity_type == "person" else OBJECTS


def make_values(entities: List[str], attribute: str) -> Dict[str, int]:
    """Generate strictly ordered values for a canonical descending chain."""
    spec = ATTRIBUTES[attribute]
    lo = spec["min"]
    hi = spec["max"]
    needed = len(entities)

    # Sample distinct values, then assign in descending order.
    values = sorted(
        random.sample(range(lo, hi + 1), needed),
        reverse=True,
    )
    return dict(zip(entities, values))


def question_subject(entity_type: str) -> str:
    """Return the correct Nepali interrogative for people or objects."""
    return "को हो" if entity_type == "person" else "कुन हो"


def direct_question(
    entity_type: str,
    a: str,
    b: str,
    noun: str,
    direction: str,
) -> str:
    """Build a direct two-entity comparison question."""
    subject = question_subject(entity_type)
    if direction == "greater":
        return f"{a} र {b} मध्ये बढी {noun} भएको {subject}?"
    if direction == "lower":
        return f"{a} र {b} मध्ये कम {noun} भएको {subject}?"
    raise ValueError(f"Unknown direction: {direction}")


def extreme_question(
    entity_type: str,
    entities: List[str],
    noun: str,
    direction: str,
) -> str:
    """Build a largest/smallest question with independently shuffled entities."""
    subject = question_subject(entity_type)
    entity_text = ", ".join(entities)

    if direction == "largest":
        return f"{entity_text} मध्ये सबैभन्दा बढी {noun} भएको {subject}?"
    if direction == "smallest":
        return f"{entity_text} मध्ये सबैभन्दा कम {noun} भएको {subject}?"
    raise ValueError(f"Unknown direction: {direction}")


ORDINAL_WORDS = {
    1: "पहिलो",
    2: "दोस्रो",
    3: "तेस्रो",
    4: "चौथो",
}


def ranking_question(
    entity_type: str,
    noun: str,
    direction: str,
    rank: int,
) -> str:
    """Build an explicit highest/lowest ranking question."""
    ordinal = ORDINAL_WORDS[rank]
    subject = question_subject(entity_type)

    if entity_type == "person":
        if direction == "highest":
            return (
                f"यी चार जनामध्ये सबैभन्दा बढी {noun} हुनेहरूलाई "
                f"क्रम मिलाउँदा {ordinal} स्थानमा को पर्छ?"
            )
        return (
            f"यी चार जनामध्ये सबैभन्दा कम {noun} हुनेहरूलाई "
            f"क्रम मिलाउँदा {ordinal} स्थानमा को पर्छ?"
        )

    if direction == "highest":
        return (
            f"यी चार वस्तुलाई सबैभन्दा बढी {noun}का आधारमा "
            f"क्रम मिलाउँदा {ordinal} स्थानमा {subject}?"
        )
    return (
        f"यी चार वस्तुलाई सबैभन्दा कम {noun}का आधारमा "
        f"क्रम मिलाउँदा {ordinal} स्थानमा {subject}?"
    )


def build_prompt(statements: List[str], question: str) -> str:
    """Shuffle statements and append the question."""
    statements = list(statements)
    random.shuffle(statements)
    return " ".join(statements + [question])


def choose_template_pair(
    split: str,
    train_templates_a: List[str],
    train_templates_b: List[str],
    test_templates_a: List[str],
    test_templates_b: List[str],
    train_ids: List[str],
    test_ids: List[str],
) -> Tuple[List[str], List[str]]:
    """Choose one surface family and its template IDs."""
    if split == "test":
        if random.choice([True, False]):
            return test_templates_a, test_ids
        return test_templates_b, test_ids
    if random.choice([True, False]):
        return train_templates_a, train_ids
    return train_templates_b, train_ids


# =============================================================================
# EXAMPLE CONSTRUCTION
# =============================================================================

def make_example(
    example_id: str,
    split: str,
    task_type: str,
    difficulty: str,
    prompt: str,
    answer: str,
    entities: List[str],
    attribute: str,
    entity_type: str,
    values: Dict[str, int],
    template_ids: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create the common JSONL record."""
    item: Dict[str, Any] = {
        "id": example_id,
        "language": LANGUAGE,
        "language_name": LANGUAGE_NAME,
        "split": split,
        "task_type": task_type,
        "difficulty": difficulty,
        "prompt": prompt,
        "answer": answer,
        "entities": entities,
        "attribute": attribute,
        "entity_type": entity_type,
        "values": values,
        "template_ids": list(template_ids),
    }
    if extra:
        item.update(extra)
    return item


# =============================================================================
# TASK GENERATORS
# =============================================================================

def generate_direct(split: str, example_id: str) -> Dict[str, Any]:
    """Generate a direct greater/lower comparison."""
    entity_type = choose_entity_type()
    pool = get_pool(entity_type)
    a, b = choose_distinct_entities(pool, 2)
    attribute = choose_attribute(entity_type)
    noun = ATTRIBUTES[attribute]["noun"]

    is_greater = random.choice([True, False])
    if is_greater:
        high, low = a, b
        direction = "greater"
        values = make_values([high, low], attribute)
    else:
        high, low = b, a
        direction = "lower"
        values = make_values([high, low], attribute)

    if split == "test":
        if is_greater:
            templates = DIRECT_GREATER_TEST
            template_ids = TEMPLATE_IDS["direct_greater_test"]
        else:
            templates = DIRECT_LOWER_TEST
            template_ids = TEMPLATE_IDS["direct_lower_test"]
    else:
        if is_greater:
            templates = DIRECT_GREATER_TRAIN
            template_ids = TEMPLATE_IDS["direct_greater_train"]
        else:
            templates = DIRECT_LOWER_TRAIN
            template_ids = TEMPLATE_IDS["direct_lower_train"]

    template_index = random.randrange(len(templates))
    statement = templates[template_index]

    # The template's {a}/{b} refer to the stated comparison order.
    if is_greater:
        statement = statement.format(a=high, b=low, noun=noun)
        answer = high
        question_direction = "greater"
    else:
        statement = statement.format(a=low, b=high, noun=noun)
        answer = low
        question_direction = "lower"

    question = direct_question(
        entity_type,
        a,
        b,
        noun,
        question_direction,
    )

    return make_example(
        example_id=example_id,
        split=split,
        task_type="direct_comparison",
        difficulty="easy",
        prompt=build_prompt([statement], question),
        answer=answer,
        entities=[a, b],
        attribute=attribute,
        entity_type=entity_type,
        values=values,
        template_ids=[template_ids[template_index]],
        extra={
            "comparison": "greater" if answer == high else "lower",
            "reasoning_steps": 1,
        },
    )


def generate_reverse(split: str, example_id: str) -> Dict[str, Any]:
    """Generate a reverse-framed comparison where the asked relation is opposite."""
    entity_type = choose_entity_type()
    pool = get_pool(entity_type)
    low, high = choose_distinct_entities(pool, 2)
    attribute = choose_attribute(entity_type)
    noun = ATTRIBUTES[attribute]["noun"]

    values = make_values([high, low], attribute)

    if split == "test":
        templates = REVERSE_TEST
        template_ids = TEMPLATE_IDS["reverse_test"]
    else:
        templates = REVERSE_TRAIN
        template_ids = TEMPLATE_IDS["reverse_train"]

    template_index = random.randrange(len(templates))
    statement = templates[template_index].format(
        low=low,
        high=high,
        noun=noun,
    )

    question = direct_question(
        entity_type,
        low,
        high,
        noun,
        "greater",
    )

    return make_example(
        example_id=example_id,
        split=split,
        task_type="reverse_comparison",
        difficulty="easy",
        prompt=build_prompt([statement], question),
        answer=high,
        entities=[low, high],
        attribute=attribute,
        entity_type=entity_type,
        values=values,
        template_ids=[template_ids[template_index]],
        extra={
            "comparison": "greater",
            "reasoning_steps": 1,
        },
    )


def generate_equality(split: str, example_id: str) -> Dict[str, Any]:
    """Generate a direct equality question."""
    entity_type = choose_entity_type()
    pool = get_pool(entity_type)
    a, b = choose_distinct_entities(pool, 2)
    attribute = choose_attribute(entity_type)
    noun = ATTRIBUTES[attribute]["noun"]

    spec = ATTRIBUTES[attribute]
    value = random.randint(spec["min"], spec["max"])
    values = {a: value, b: value}

    if split == "test":
        templates = EQUALITY_TEST
        template_ids = TEMPLATE_IDS["equality_test"]
    else:
        templates = EQUALITY_TRAIN
        template_ids = TEMPLATE_IDS["equality_train"]

    template_index = random.randrange(len(templates))
    statement = templates[template_index].format(
        a=a,
        b=b,
        noun=noun,
    )

    subject = question_subject(entity_type)
    question = f"{a} र {b}को {noun} बराबर छ?"

    return make_example(
        example_id=example_id,
        split=split,
        task_type="equality",
        difficulty="easy",
        prompt=build_prompt([statement], question),
        answer="हो",
        entities=[a, b],
        attribute=attribute,
        entity_type=entity_type,
        values=values,
        template_ids=[template_ids[template_index]],
        extra={
            "is_equal": True,
            "reasoning_steps": 1,
        },
    )


def generate_transitive_2(split: str, example_id: str) -> Dict[str, Any]:
    """Generate a 2-hop transitive comparison."""
    entity_type = choose_entity_type()
    pool = get_pool(entity_type)
    a, b, c = choose_distinct_entities(pool, 3)
    attribute = choose_attribute(entity_type)
    noun = ATTRIBUTES[attribute]["noun"]

    # Canonical chain: a > b > c.
    values = make_values([a, b, c], attribute)

    if split == "test":
        templates = (
            TRANSITIVE_2_TEST
            if random.choice([True, False])
            else TRANSITIVE_2_ALT_TEST
        )
        template_ids = TEMPLATE_IDS["transitive_2_test"]
    else:
        templates = (
            TRANSITIVE_2_TRAIN
            if random.choice([True, False])
            else TRANSITIVE_2_ALT_TRAIN
        )
        template_ids = TEMPLATE_IDS["transitive_2_train"]

    statements = [
        template.format(a=a, b=b, c=c, noun=noun)
        for template in templates
    ]
    random.shuffle(statements)

    ask_largest = random.choice([True, False])
    if ask_largest:
        answer = a
        direction = "largest"
    else:
        answer = c
        direction = "smallest"

    # Shuffle only the question presentation, not the canonical chain.
    question_entities = [a, b, c]
    random.shuffle(question_entities)

    question = extreme_question(
        entity_type,
        question_entities,
        noun,
        direction,
    )

    return make_example(
        example_id=example_id,
        split=split,
        task_type="transitive_2",
        difficulty="medium",
        prompt=build_prompt(statements, question),
        answer=answer,
        entities=[a, b, c],
        attribute=attribute,
        entity_type=entity_type,
        values=values,
        template_ids=list(template_ids),
        extra={
            "reasoning_steps": 2,
            "chain": [a, b, c],
            "query_direction": direction,
        },
    )


def generate_transitive_3(split: str, example_id: str) -> Dict[str, Any]:
    """Generate a 3-hop transitive comparison."""
    entity_type = choose_entity_type()
    pool = get_pool(entity_type)
    a, b, c, d = choose_distinct_entities(pool, 4)
    attribute = choose_attribute(entity_type)
    noun = ATTRIBUTES[attribute]["noun"]

    # Canonical chain: a > b > c > d.
    values = make_values([a, b, c, d], attribute)

    if split == "test":
        templates = (
            TRANSITIVE_3_TEST
            if random.choice([True, False])
            else TRANSITIVE_3_ALT_TEST
        )
        template_ids = TEMPLATE_IDS["transitive_3_test"]
    else:
        templates = (
            TRANSITIVE_3_TRAIN
            if random.choice([True, False])
            else TRANSITIVE_3_ALT_TRAIN
        )
        template_ids = TEMPLATE_IDS["transitive_3_train"]

    statements = [
        template.format(a=a, b=b, c=c, d=d, noun=noun)
        for template in templates
    ]
    random.shuffle(statements)

    ask_largest = random.choice([True, False])
    if ask_largest:
        answer = a
        direction = "largest"
    else:
        answer = d
        direction = "smallest"

    question_entities = [a, b, c, d]
    random.shuffle(question_entities)

    question = extreme_question(
        entity_type,
        question_entities,
        noun,
        direction,
    )

    return make_example(
        example_id=example_id,
        split=split,
        task_type="transitive_3",
        difficulty="hard",
        prompt=build_prompt(statements, question),
        answer=answer,
        entities=[a, b, c, d],
        attribute=attribute,
        entity_type=entity_type,
        values=values,
        template_ids=list(template_ids),
        extra={
            "reasoning_steps": 3,
            "chain": [a, b, c, d],
            "query_direction": direction,
        },
    )


def generate_ordering(split: str, example_id: str) -> Dict[str, Any]:
    """Generate a four-entity ordering/ranking problem."""
    entity_type = choose_entity_type()
    pool = get_pool(entity_type)
    a, b, c, d = choose_distinct_entities(pool, 4)
    attribute = choose_attribute(entity_type)
    noun = ATTRIBUTES[attribute]["noun"]

    # Canonical descending order: a > b > c > d.
    values = make_values([a, b, c, d], attribute)

    if split == "test":
        templates = (
            ORDERING_STATEMENT_TEST
            if random.choice([True, False])
            else ORDERING_STATEMENT_ALT_TEST
        )
        template_ids = TEMPLATE_IDS["ordering_test"]
    else:
        templates = (
            ORDERING_STATEMENT_TRAIN
            if random.choice([True, False])
            else ORDERING_STATEMENT_ALT_TRAIN
        )
        template_ids = TEMPLATE_IDS["ordering_train"]

    statements = [
        template.format(a=a, b=b, c=c, d=d, noun=noun)
        for template in templates
    ]
    random.shuffle(statements)

    use_highest = random.choice([True, False])

    if use_highest:
        ordered_entities = [a, b, c, d]
        direction = "highest"
    else:
        ordered_entities = [d, c, b, a]
        direction = "lowest"

    rank = random.randint(1, 4)
    answer = ordered_entities[rank - 1]

    question = ranking_question(
        entity_type,
        noun,
        direction,
        rank,
    )

    return make_example(
        example_id=example_id,
        split=split,
        task_type="ordering",
        difficulty="hard",
        prompt=build_prompt(statements, question),
        answer=answer,
        entities=[a, b, c, d],
        attribute=attribute,
        entity_type=entity_type,
        values=values,
        template_ids=list(template_ids),
        extra={
            "reasoning_steps": 3,
            "ranking": {
                "direction": direction,
                "rank": rank,
                "ordered_entities": ordered_entities,
            },
        },
    )


GENERATORS = {
    "direct_comparison": generate_direct,
    "reverse_comparison": generate_reverse,
    "equality": generate_equality,
    "transitive_2": generate_transitive_2,
    "transitive_3": generate_transitive_3,
    "ordering": generate_ordering,
}


# =============================================================================
# VALIDATION
# =============================================================================

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
NUMBER_RE = re.compile(r"(?<![\u0966-\u096F])\d+(?![ववा])")


def validate_example(example: Dict[str, Any]) -> None:
    """Validate one generated reasoning example."""
    entities = example["entities"]
    values = example["values"]
    task = example["task_type"]
    attribute = example["attribute"]
    entity_type = example["entity_type"]
    prompt = example["prompt"]
    answer = example["answer"]

    assert len(entities) == len(set(entities)), (
        f"Duplicate entities within example {example['id']}"
    )

    assert all(e in values for e in entities)

    assert ATTRIBUTES[attribute]["entity_type"] == entity_type

    if entity_type == "person":
        assert all(e in PEOPLE for e in entities)
    else:
        assert all(e in OBJECTS for e in entities)

    spec = ATTRIBUTES[attribute]
    for entity, value in values.items():
        assert spec["min"] <= value <= spec["max"], (
            f"Out-of-range value in {example['id']}: {entity}={value}"
        )

    assert answer in entities or (task == "equality" and answer == "हो")

    # Prompt must be predominantly Devanagari and must not contain obvious
    # English scaffold words.
    assert DEVANAGARI_RE.search(prompt), f"No Devanagari in {example['id']}"
    assert not LATIN_RE.search(prompt), f"Latin characters in {example['id']}"

    # Numeric values must never be leaked into the prompt.
    for value in values.values():
        assert str(value) not in prompt, (
            f"Numeric value leakage in {example['id']}: {value}"
        )

    if task == "equality":
        assert len(set(values.values())) == 1
        assert example.get("is_equal") is True

    elif task == "direct_comparison":
        a, b = entities
        if example["comparison"] == "greater":
            assert values[answer] == max(values[a], values[b])
        else:
            assert values[answer] == min(values[a], values[b])

    elif task == "reverse_comparison":
        assert example["comparison"] == "greater"
        assert values[answer] == max(values.values())

    elif task in {"transitive_2", "transitive_3"}:
        chain = example["chain"]
        assert chain == entities
        chain_values = [values[e] for e in chain]
        assert all(
            chain_values[i] > chain_values[i + 1]
            for i in range(len(chain_values) - 1)
        )

        if example["query_direction"] == "largest":
            assert answer == chain[0]
        else:
            assert answer == chain[-1]

    elif task == "ordering":
        ranking = example["ranking"]
        ordered = ranking["ordered_entities"]

        if ranking["direction"] == "highest":
            expected = sorted(
                entities,
                key=lambda e: values[e],
                reverse=True,
            )
        else:
            expected = sorted(
                entities,
                key=lambda e: values[e],
            )

        assert ordered == expected
        assert answer == ordered[ranking["rank"] - 1]


def validate_dataset(
    examples: List[Dict[str, Any]],
    split: str,
) -> Dict[str, Any]:
    """Run dataset-level checks and return statistics."""
    assert all(e["split"] == split for e in examples)

    ids = [e["id"] for e in examples]
    prompts = [e["prompt"] for e in examples]

    assert len(ids) == len(set(ids)), f"Duplicate IDs in {split}"
    assert len(prompts) == len(set(prompts)), f"Duplicate prompts in {split}"

    for example in examples:
        validate_example(example)

    task_counts = Counter(e["task_type"] for e in examples)
    attr_counts = Counter(e["attribute"] for e in examples)
    entity_type_counts = Counter(e["entity_type"] for e in examples)
    difficulty_counts = Counter(e["difficulty"] for e in examples)

    steps = [e["reasoning_steps"] for e in examples]

    return {
        "split": split,
        "size": len(examples),
        "task_counts": dict(task_counts),
        "task_percentages": {
            k: round(v / len(examples) * 100, 3)
            for k, v in task_counts.items()
        },
        "attribute_counts": dict(attr_counts),
        "entity_type_counts": dict(entity_type_counts),
        "difficulty_counts": dict(difficulty_counts),
        "mean_reasoning_steps": round(sum(steps) / len(steps), 4),
        "unique_entities": len({
            entity
            for e in examples
            for entity in e["entities"]
        }),
        "unique_answers": len({e["answer"] for e in examples}),
        "multi_hop_examples": sum(
            e["task_type"] in {"transitive_2", "transitive_3", "ordering"}
            for e in examples
        ),
    }


def validate_cross_split(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate leakage and split-level uniqueness constraints."""
    train_prompts = {e["prompt"] for e in train}
    val_prompts = {e["prompt"] for e in val}
    test_prompts = {e["prompt"] for e in test}

    train_val_overlap = train_prompts & val_prompts
    train_test_overlap = train_prompts & test_prompts
    val_test_overlap = val_prompts & test_prompts

    assert not train_val_overlap, (
        f"Train/validation exact prompt leakage: {len(train_val_overlap)}"
    )
    assert not train_test_overlap, (
        f"Train/test exact prompt leakage: {len(train_test_overlap)}"
    )
    assert not val_test_overlap, (
        f"Validation/test exact prompt leakage: {len(val_test_overlap)}"
    )

    train_templates = {
        tid
        for e in train + val
        for tid in e["template_ids"]
    }
    test_templates = {
        tid
        for e in test
        for tid in e["template_ids"]
    }

    template_overlap = train_templates & test_templates
    assert not template_overlap, (
        f"Test-template leakage: {sorted(template_overlap)}"
    )

    # Entity overlap is intentionally allowed.
    return {
        "train_val_exact_prompt_overlap": len(train_val_overlap),
        "train_test_exact_prompt_overlap": len(train_test_overlap),
        "val_test_exact_prompt_overlap": len(val_test_overlap),
        "train_val_template_count": len(train_templates),
        "test_template_count": len(test_templates),
        "template_overlap": sorted(template_overlap),
        "entity_overlap_allowed": True,
    }


# =============================================================================
# GENERATION
# =============================================================================

def choose_task() -> str:
    """Sample a task according to the configured distribution."""
    tasks = list(TASK_WEIGHTS.keys())
    weights = list(TASK_WEIGHTS.values())
    return random.choices(tasks, weights=weights, k=1)[0]


def generate_split(
    split: str,
    size: int,
    global_prompt_set: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Generate one split while enforcing global prompt uniqueness."""
    if global_prompt_set is None:
        global_prompt_set = set()

    examples: List[Dict[str, Any]] = []
    local_prompt_set: Set[str] = set()
    duplicate_retries = 0
    max_attempts = max(size * 100, 1000)

    attempts = 0
    while len(examples) < size:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {size} unique {split} examples."
            )

        task = choose_task()
        example_id = f"nepali_{split}_{len(examples):06d}"
        example = GENERATORS[task](split, example_id)

        prompt = example["prompt"]

        if prompt in local_prompt_set or prompt in global_prompt_set:
            duplicate_retries += 1
            continue

        validate_example(example)

        examples.append(example)
        local_prompt_set.add(prompt)
        global_prompt_set.add(prompt)

    return examples, {
        "duplicate_retries": duplicate_retries,
        "generation_attempts": attempts,
    }


def save_jsonl(path: Path, examples: List[Dict[str, Any]]) -> None:
    """Write examples as UTF-8 JSONL."""
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Generate, validate, and save all Nepali reasoning splits."""
    parser = argparse.ArgumentParser(
        description="Generate Nepali synthetic reasoning data for Phase 3."
    )
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data"),
    )
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_prompt_set: Set[str] = set()

    print("=" * 80)
    print("NEPALI — SYNTHETIC REASONING DATA GENERATION")
    print("=" * 80)
    print(f"Seed:       {args.seed}")
    print(f"Train:      {args.train_size:,}")
    print(f"Validation: {args.val_size:,}")
    print(f"Test:       {args.test_size:,}")
    print()

    train, train_gen_stats = generate_split(
        "train",
        args.train_size,
        global_prompt_set,
    )
    val, val_gen_stats = generate_split(
        "val",
        args.val_size,
        global_prompt_set,
    )
    test, test_gen_stats = generate_split(
        "test",
        args.test_size,
        global_prompt_set,
    )

    print("Running dataset-level validation...")

    train_stats = validate_dataset(train, "train")
    val_stats = validate_dataset(val, "val")
    test_stats = validate_dataset(test, "test")
    leakage_stats = validate_cross_split(train, val, test)

    save_jsonl(output_dir / "train.jsonl", train)
    save_jsonl(output_dir / "val.jsonl", val)
    save_jsonl(output_dir / "test.jsonl", test)

    statistics = {
        "language": LANGUAGE,
        "language_name": LANGUAGE_NAME,
        "seed": args.seed,
        "sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "total": len(train) + len(val) + len(test),
        },
        "task_weights": TASK_WEIGHTS,
        "attributes": ATTRIBUTES,
        "split_statistics": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "generation_statistics": {
            "train": train_gen_stats,
            "val": val_gen_stats,
            "test": test_gen_stats,
        },
        "leakage_and_integrity": leakage_stats,
        "leakage_policy": {
            "exact_prompt_overlap_across_splits": "prohibited",
            "test_template_overlap_with_train_val": "prohibited",
            "entity_overlap_across_splits": "allowed",
            "numeric_value_in_prompt": "prohibited",
            "statement_order": "shuffled",
            "transitive_question_entity_order": "shuffled",
        },
        "template_counts": {
            key: len(value)
            for key, value in TEMPLATE_IDS.items()
        },
    }

    with (output_dir / "statistics.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            statistics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    # ========================================================================
    # FINAL ARTIFACT VALIDATION
    # Re-read the generated JSONL files and independently validate the
    # artifacts after generation has completed.
    # ========================================================================

    print("\nRunning final artifact audit...")

    import re
    from collections import Counter

    output_paths = {
        "train": output_dir / "train.jsonl",
        "val": output_dir / "val.jsonl",
        "test": output_dir / "test.jsonl",
    }

    generated = {}
    for split, path in output_paths.items():
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.rstrip("\n")
                assert line, f"{split}: empty line at {line_no}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AssertionError(
                        f"{split}: invalid JSON at line {line_no}: {exc}"
                    ) from exc
                assert isinstance(row, dict), (
                    f"{split}: line {line_no} is not a JSON object"
                )
                rows.append(row)
        generated[split] = rows

    expected_sizes = {
        "train": len(generated["train"]),
        "val": len(generated["val"]),
        "test": len(generated["test"]),
    }

    required_fields = {
        "id", "language", "split", "task_type", "difficulty",
        "prompt", "answer", "entities", "attribute", "entity_type",
        "values", "template_ids",
    }

    # 1. Sizes + schema
    for split, rows in generated.items():
        assert len(rows) == expected_sizes[split], (
            f"{split}: expected {expected_sizes[split]} rows, got {len(rows)}"
        )

        for i, row in enumerate(rows, 1):
            missing = required_fields - set(row)
            assert not missing, (
                f"{split}: row {i} missing fields: {sorted(missing)}"
            )
            assert row["language"] == "ne", (
                f"{split}: row {i}: language must be 'ne'"
            )
            assert row["split"] == split, (
                f"{split}: row {i}: incorrect split field"
            )
            assert isinstance(row["prompt"], str) and row["prompt"].strip()
            assert isinstance(row["answer"], str) and row["answer"].strip()
            assert isinstance(row["entities"], list) and row["entities"]
            assert isinstance(row["values"], dict) and row["values"]
            assert isinstance(row["template_ids"], list) and row["template_ids"]

    # 2. IDs + exact prompt uniqueness
    all_ids = []
    all_prompts = []
    split_prompts = {}

    for split, rows in generated.items():
        split_prompts[split] = set()
        for row in rows:
            all_ids.append(row["id"])
            all_prompts.append(row["prompt"])
            assert row["prompt"] not in split_prompts[split], (
                f"{split}: duplicate prompt found"
            )
            split_prompts[split].add(row["prompt"])

    assert len(all_ids) == len(set(all_ids)), "Duplicate IDs found"
    assert len(all_prompts) == len(set(all_prompts)), (
        "Duplicate prompts found across the full dataset"
    )

    assert split_prompts["train"].isdisjoint(split_prompts["val"]), (
        "Exact prompt overlap: train/val"
    )
    assert split_prompts["train"].isdisjoint(split_prompts["test"]), (
        "Exact prompt overlap: train/test"
    )
    assert split_prompts["val"].isdisjoint(split_prompts["test"]), (
        "Exact prompt overlap: val/test"
    )

    # 3. Task/schema values
    allowed_tasks = {
        "direct_comparison",
        "reverse_comparison",
        "equality",
        "transitive_2",
        "transitive_3",
        "ordering",
    }
    allowed_difficulties = {"easy", "medium", "hard"}

    for split, rows in generated.items():
        for i, row in enumerate(rows, 1):
            assert row["task_type"] in allowed_tasks, (
                f"{split}: row {i}: invalid task_type"
            )
            assert row["difficulty"] in allowed_difficulties, (
                f"{split}: row {i}: invalid difficulty"
            )
            assert set(row["values"]) == set(row["entities"]), (
                f"{split}: row {i}: values/entities mismatch"
            )

            # Ground-truth numbers are metadata only and must not leak into
            # the model input.
            for value in row["values"].values():
                assert str(value) not in row["prompt"], (
                    f"{split}: row {i}: numeric value leaked into prompt"
                )

            # Nepali prompts must contain Devanagari text.
            assert re.search(r"[\u0900-\u097F]", row["prompt"]), (
                f"{split}: row {i}: no Devanagari text in prompt"
            )

    # 4. Answer membership + equality semantics
    for split, rows in generated.items():
        for i, row in enumerate(rows, 1):
            task = row["task_type"]
            answer = row["answer"].strip()
            values = row["values"]
            entities = row["entities"]

            if task == "equality":
                vals = list(values.values())
                assert len(vals) == 2 and vals[0] == vals[1], (
                    f"{split}: row {i}: equality ground truth invalid"
                )
                assert answer == "हो", (
                    f"{split}: row {i}: equality answer is not 'हो'"
                )
            else:
                assert answer in entities, (
                    f"{split}: row {i}: answer is not one of the entities"
                )

    # 5. Test template isolation
    train_val_templates = set()
    test_templates = set()

    for split in ("train", "val"):
        for row in generated[split]:
            train_val_templates.update(row["template_ids"])

    for row in generated["test"]:
        test_templates.update(row["template_ids"])

    assert train_val_templates.isdisjoint(test_templates), (
        "Template-ID overlap between train/val and test"
    )

    # 6. Distribution summary
    task_counts = {
        split: Counter(row["task_type"] for row in rows)
        for split, rows in generated.items()
    }

    final_report = {
        "status": "PASS",
        "total_examples": sum(len(rows) for rows in generated.values()),
        "split_sizes": {split: len(rows) for split, rows in generated.items()},
        "global_unique_prompts": len(set(all_prompts)),
        "global_unique_ids": len(set(all_ids)),
        "exact_prompt_overlap": {
            "train_val": len(split_prompts["train"] & split_prompts["val"]),
            "train_test": len(split_prompts["train"] & split_prompts["test"]),
            "val_test": len(split_prompts["val"] & split_prompts["test"]),
        },
        "template_overlap_train_val_vs_test": sorted(
            train_val_templates & test_templates
        ),
        "task_counts": {
            split: dict(counts) for split, counts in task_counts.items()
        },
    }

    validation_path = output_dir / "final_validation.json"
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print("Final artifact audit: PASS")
    print(f"Validation report: {validation_path}")

    print("Generation complete.")
    print(f"Output directory: {output_dir}")
    print()
    print("Integrity checks: PASS")
    print(
        f"Total examples: "
        f"{len(train) + len(val) + len(test):,}"
    )
    print(
        f"Global unique prompts: "
        f"{len(global_prompt_set):,}"
    )


if __name__ == "__main__":
    main()
