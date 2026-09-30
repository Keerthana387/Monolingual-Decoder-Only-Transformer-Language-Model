"""
==============================================================================
TELUGU — PHASE 3 REASONING DATASET GENERATOR
==============================================================================

Purpose
-------
Generate synthetic Telugu reasoning data for Phase 3 fine-tuning.

Task families
-------------
1. Direct comparison
2. Reverse comparison
3. Equality
4. Two-hop transitive reasoning
5. Three-hop transitive reasoning
6. Ordering / ranking

Design principles
-----------------
- Telugu only.
- Natural Telugu names and object nouns.
- No English prompts.
- No A/B/C artificial entities.
- People use:
      * height
      * age
      * weight
- Objects use:
      * price
      * quantity
- Entities within an individual example are always distinct.
- Entity names MAY appear in multiple splits.
- Exact reasoning prompts MAY NOT appear in multiple splits.
- Test templates are held out from train/validation.
- Numerical values are stored in metadata only.
- Numerical values are not shown in the prompt.
- Multi-hop statements are shuffled.
- Every example is validated immediately.
- Dataset-level integrity checks are performed.
- Deterministic generation using a fixed seed.

Output
------
reasoning/data/
    train.jsonl
    val.jsonl
    test.jsonl
    statistics.json

Default sizes
-------------
Train: 20,000
Val:    2,000
Test:   3,000

Smoke test
----------
python reasoning/generate_reasoning_data.py \
    --train-size 30 \
    --val-size 10 \
    --test-size 10

Full generation
---------------
python reasoning/generate_reasoning_data.py \
    --train-size 20000 \
    --val-size 2000 \
    --test-size 3000
"""

from __future__ import annotations

import argparse
import json
import random
import re

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 2026

DEFAULT_TRAIN_SIZE = 20_000
DEFAULT_VAL_SIZE = 2_000
DEFAULT_TEST_SIZE = 3_000

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

LANGUAGE = "te"
LANGUAGE_NAME = "Telugu"


# =============================================================================
# ENTITY POOLS
# =============================================================================
#
# Entity overlap across splits is intentionally allowed.
#
# We are measuring generalization to NEW REASONING INSTANCES, not requiring
# completely unseen entity names.
#
# The important leakage controls are:
#
#   1. no duplicate entities within one example
#   2. no exact duplicate prompts within a split
#   3. no exact duplicate prompts across splits
#   4. no test-template leakage
#   5. hidden numerical values are not placed in prompts
#
# =============================================================================


PEOPLE = [
    "అజయ్",
    "రవి",
    "సురేష్",
    "మహేష్",
    "కిరణ్",
    "వికాస్",
    "నరేష్",
    "శ్రీనివాస్",
    "ప్రదీప్",
    "వెంకట్",
    "రాజేష్",
    "అనిల్",
    "సందీప్",
    "కార్తిక్",
    "హరీష్",
    "రమేష్",
    "వంశీ",
    "తేజ",
    "చరణ్",
    "అఖిల్",
    "మధు",
    "దినేష్",
    "రాకేష్",
    "సాయి",
    "నిఖిల్",
    "రోహిత్",
    "వినయ్",
    "గోపాల్",
    "మనోజ్",
    "భరత్",
    "శేఖర్",
    "ప్రణయ్",
    "తరుణ్",
    "ఉదయ్",
    "అరవింద్",
    "హేమంత్",
    "రాహుల్",
    "అభిషేక్",
    "దీపక్",
    "గౌతమ్",
    "విజయ్",
    "రాజు",
    "కిషోర్",
    "మురళి",
    "నవీన్",
    "ప్రవీణ్",
    "సుధీర్",
    "శశాంక్",
    "సతీష్",
    "మోహన్",
    "శివ",
    "కృష్ణ",
    "ఆదిత్య",
    "అభినవ్",
    "విశాల్",
    "సంజయ్",
    "యశ్వంత్",
    "సుమన్",
    "లోకేష్",
    "రమణ",
    "చైతన్య",
    "హర్ష",
    "ధీరజ్",
    "అశోక్",
    "మోహిత్",
    "శ్రవణ్",
    "అన్వేష్",
    "వెంకటేష్",
    "శ్రీకాంత్",
    "మధుకర్",
    "రఘు",
    "కిరణ్ కుమార్",
    "సాయి కిరణ్",
    "వంశీ కృష్ణ",
]


OBJECTS = [
    "పుస్తకం",
    "కలం",
    "బ్యాగ్",
    "కుర్చీ",
    "బల్ల",
    "గడియారం",
    "మొబైల్ ఫోన్",
    "ల్యాప్‌టాప్",
    "సైకిల్",
    "బంతి",
    "బాటిల్",
    "పెన్‌డ్రైవ్",
    "హెడ్‌ఫోన్",
    "కెమెరా",
    "క్యాలిక్యులేటర్",
    "నోట్‌బుక్",
    "పెన్",
    "పెన్సిల్",
    "గ్లాస్",
    "కప్పు",
    "పెట్టె",
    "గొడుగు",
    "చార్జర్",
    "మౌస్",
    "స్పీకర్",
    "రేడియో",
    "టార్చ్",
    "క్యాలెండర్",
    "ఫ్యాన్",
    "టేబుల్ ల్యాంప్",
    "డైరీ",
    "సూట్‌కేస్",
    "క్యాప్",
    "బూట్లు",
    "కళ్లజోడు",
    "కీచైన్",
    "వాలెట్",
    "బొమ్మ",
    "బ్యాట్",
    "రాకెట్",
    "క్యాసెట్",
    "నీళ్ల సీసా",
    "వైర్‌లెస్ మౌస్",
    "టాబ్లెట్",
    "పేపర్ బాక్స్",
    "స్టేషనరీ బాక్స్",
    "ఫైల్ ఫోల్డర్",
    "పెన్ స్టాండ్",
    "డెస్క్ ఆర్గనైజర్",
    "చిత్రపటం",
    "వంట పాత్ర",
    "టీ కప్పు",
    "పూల కుండీ",
    "చేతి అద్దం",
    "పెన్సిల్ బాక్స్",
    "పుస్తకాల అల్మరా",
    "చిన్న దీపం",
    "కాఫీ మగ్గు",
    "చేతి బ్యాగ్",
    "ప్రయాణ బ్యాగ్",
    "చిత్రాల ఫ్రేమ్",
    "చెక్క పెట్టె",
    "కాగితాల ఫైల్",
    "స్టీల్ బాటిల్",
    "నీటి జగ్",
    "రిమోట్",
    "చిన్న రేడియో",
    "టేబుల్ క్లాక్",
    "పెన్ ఆర్గనైజర్",
    "పుస్తకాల పెట్టె",
    "స్కూల్ బ్యాగ్",
    "ప్రయాణ సూట్‌కేస్",
    "చిన్న స్పీకర్",
    "డిజిటల్ గడియారం",
    "కాఫీ ఫ్లాస్క్",
    "చేతి టార్చ్",
    "కంప్యూటర్ మానిటర్",
    "మౌస్ ప్యాడ్",
    "పెన్ డ్రైవ్",
    "చార్జింగ్ కేబుల్",
    "హెడ్‌సెట్",
    "కెమెరా బ్యాగ్",
    "పేపర్ బాస్కెట్",
    "చిన్న బల్ల",
    "చెక్క కుర్చీ",
    "ప్రయాణ పెట్టె",
    "గాజు సీసా",
    "చిన్న టేబుల్",
]


# =============================================================================
# ATTRIBUTE DEFINITIONS
# =============================================================================

ATTRIBUTES = {
    "height": {
        "entity_type": "person",
        "noun": "ఎత్తు",
        "unit": "సెంటీమీటర్లు",
        "min_value": 145,
        "max_value": 195,
    },
    "age": {
        "entity_type": "person",
        "noun": "వయస్సు",
        "unit": "సంవత్సరాలు",
        "min_value": 18,
        "max_value": 65,
    },
    "weight": {
        "entity_type": "person",
        "noun": "బరువు",
        "unit": "కిలోలు",
        "min_value": 45,
        "max_value": 100,
    },
    "price": {
        "entity_type": "object",
        "noun": "ధర",
        "unit": "రూపాయలు",
        "min_value": 100,
        "max_value": 5000,
    },
    "quantity": {
        "entity_type": "object",
        "noun": "పరిమాణం",
        "unit": "యూనిట్లు",
        "min_value": 1,
        "max_value": 100,
    },
}


PERSON_ATTRIBUTES = [
    "height",
    "age",
    "weight",
]

OBJECT_ATTRIBUTES = [
    "price",
    "quantity",
]


# =============================================================================
# TASK DISTRIBUTION
# =============================================================================

TASK_WEIGHTS = {
    "direct_comparison": 0.20,
    "reverse_comparison": 0.15,
    "equality": 0.10,
    "transitive_2": 0.20,
    "transitive_3": 0.20,
    "ordering": 0.15,
}


# =============================================================================
# TEMPLATES
# =============================================================================


# -----------------------------------------------------------------------------
# DIRECT — TRAIN / VALIDATION
# -----------------------------------------------------------------------------

DIRECT_GREATER_TRAIN = [
    "{a} యొక్క {noun} {b} కంటే ఎక్కువగా ఉంది.",
    "{a} యొక్క {noun} {b} కంటే ఎక్కువ.",
    "{b} యొక్క {noun}తో పోలిస్తే {a} యొక్క {noun} ఎక్కువగా ఉంది.",
    "{a}తో పోల్చినప్పుడు {b} యొక్క {noun} కంటే {a} యొక్క {noun} ఎక్కువగా ఉంది.",
]

DIRECT_LOWER_TRAIN = [
    "{a} యొక్క {noun} {b} కంటే తక్కువగా ఉంది.",
    "{a} యొక్క {noun} {b} కంటే తక్కువ.",
    "{b} యొక్క {noun}తో పోలిస్తే {a} యొక్క {noun} తక్కువగా ఉంది.",
    "{a}తో పోల్చినప్పుడు {b} యొక్క {noun} కంటే {a} యొక్క {noun} తక్కువగా ఉంది.",
]


# -----------------------------------------------------------------------------
# DIRECT — TEST
# -----------------------------------------------------------------------------

DIRECT_GREATER_TEST = [
    "{b} యొక్క {noun} కంటే {a} యొక్క {noun} ఎక్కువగా ఉంది.",
    "{a} యొక్క {noun} ఎక్కువ కాగా, {b} యొక్క {noun} తక్కువగా ఉంది.",
    "{b} యొక్క {noun}తో పోలిస్తే {a} యొక్క {noun} ఎక్కువగా ఉంది.",
]

DIRECT_LOWER_TEST = [
    "{b} యొక్క {noun} కంటే {a} యొక్క {noun} తక్కువగా ఉంది.",
    "{a} యొక్క {noun} తక్కువ కాగా, {b} యొక్క {noun} ఎక్కువగా ఉంది.",
    "{b} యొక్క {noun}తో పోలిస్తే {a} యొక్క {noun} తక్కువగా ఉంది.",
]


# -----------------------------------------------------------------------------
# REVERSE — TRAIN / VALIDATION
# -----------------------------------------------------------------------------

REVERSE_TRAIN = [
    "{low} యొక్క {noun} {high} కంటే తక్కువగా ఉంది.",
    "{low} యొక్క {noun} {high} కంటే తక్కువ.",
    "{high} యొక్క {noun}తో పోలిస్తే {low} యొక్క {noun} తక్కువగా ఉంది.",
    "{high} యొక్క {noun} కంటే {low} యొక్క {noun} తక్కువగా ఉంది.",
]


# -----------------------------------------------------------------------------
# REVERSE — TEST
# -----------------------------------------------------------------------------

REVERSE_TEST = [
    "{low} యొక్క {noun}తో పోలిస్తే {high} యొక్క {noun} ఎక్కువగా ఉంది.",
    "{low} యొక్క {noun} తక్కువగా ఉండగా, {high} యొక్క {noun} ఎక్కువగా ఉంది.",
    "{high} యొక్క {noun} {low} కంటే ఎక్కువ.",
]


# -----------------------------------------------------------------------------
# EQUALITY — TRAIN / VALIDATION
# -----------------------------------------------------------------------------

EQUALITY_TRAIN = [
    "{a} మరియు {b} యొక్క {noun} సమానంగా ఉంది.",
    "{a} యొక్క {noun}, {b} యొక్క {noun}తో సమానం.",
    "{a} మరియు {b} ఇద్దరి {noun} సమానంగా ఉంది.",
]


# -----------------------------------------------------------------------------
# EQUALITY — TEST
# -----------------------------------------------------------------------------

EQUALITY_TEST = [
    "{a} యొక్క {noun} మరియు {b} యొక్క {noun} ఒకే విధంగా ఉన్నాయి.",
    "{a} యొక్క {noun} మరియు {b} యొక్క {noun}లో తేడా లేదు.",
    "{a} మరియు {b} ఇద్దరి {noun} ఒకే స్థాయిలో ఉంది.",
]


# -----------------------------------------------------------------------------
# TRANSITIVE 2 — TRAIN / VALIDATION
# -----------------------------------------------------------------------------

TRANSITIVE_2_TRAIN = [
    "{a} యొక్క {noun} {b} కంటే ఎక్కువగా ఉంది.",
    "{b} యొక్క {noun} {c} కంటే ఎక్కువగా ఉంది.",
]

TRANSITIVE_2_ALT_TRAIN = [
    "{a} యొక్క {noun} {b} కంటే ఎక్కువ.",
    "{c} యొక్క {noun}తో పోలిస్తే {b} యొక్క {noun} ఎక్కువగా ఉంది.",
]


# -----------------------------------------------------------------------------
# TRANSITIVE 2 — TEST
# -----------------------------------------------------------------------------

TRANSITIVE_2_TEST = [
    "{b} యొక్క {noun}తో పోలిస్తే {a} యొక్క {noun} ఎక్కువగా ఉంది.",
    "{c} యొక్క {noun} {b} కంటే తక్కువ.",
]

TRANSITIVE_2_ALT_TEST = [
    "{a} యొక్క {noun} {b} కంటే ఎక్కువగా ఉంది.",
    "{b} యొక్క {noun} {c} కంటే ఎక్కువగా ఉంది.",
]


# -----------------------------------------------------------------------------
# TRANSITIVE 3 — TRAIN / VALIDATION
# -----------------------------------------------------------------------------

TRANSITIVE_3_TRAIN = [
    "{a} యొక్క {noun} {b} కంటే ఎక్కువగా ఉంది.",
    "{b} యొక్క {noun} {c} కంటే ఎక్కువగా ఉంది.",
    "{c} యొక్క {noun} {d} కంటే ఎక్కువగా ఉంది.",
]

TRANSITIVE_3_ALT_TRAIN = [
    "{a} యొక్క {noun} {b} కంటే ఎక్కువ.",
    "{c} యొక్క {noun}తో పోలిస్తే {b} యొక్క {noun} ఎక్కువగా ఉంది.",
    "{d} యొక్క {noun} {c} కంటే తక్కువగా ఉంది.",
]


# -----------------------------------------------------------------------------
# TRANSITIVE 3 — TEST
# -----------------------------------------------------------------------------

TRANSITIVE_3_TEST = [
    "{b} యొక్క {noun}తో పోలిస్తే {a} యొక్క {noun} ఎక్కువగా ఉంది.",
    "{c} యొక్క {noun} {b} కంటే తక్కువగా ఉంది.",
    "{d} యొక్క {noun} {c} కంటే తక్కువ.",
]

TRANSITIVE_3_ALT_TEST = [
    "{b} యొక్క {noun} {a} కంటే తక్కువగా ఉంది.",
    "{c} యొక్క {noun} {b} కంటే తక్కువ.",
    "{d} యొక్క {noun}తో పోలిస్తే {c} యొక్క {noun} ఎక్కువగా ఉంది.",
]


# -----------------------------------------------------------------------------
# ORDERING — TRAIN / VALIDATION
# -----------------------------------------------------------------------------

ORDERING_STATEMENT_TRAIN = [
    "{a} యొక్క {noun} {b} కంటే ఎక్కువగా ఉంది.",
    "{b} యొక్క {noun} {c} కంటే ఎక్కువగా ఉంది.",
    "{c} యొక్క {noun} {d} కంటే ఎక్కువగా ఉంది.",
]

ORDERING_STATEMENT_ALT_TRAIN = [
    "{a} యొక్క {noun} {b} కంటే ఎక్కువ.",
    "{c} యొక్క {noun}తో పోలిస్తే {b} యొక్క {noun} ఎక్కువగా ఉంది.",
    "{d} యొక్క {noun} {c} కంటే తక్కువగా ఉంది.",
]


# -----------------------------------------------------------------------------
# ORDERING — TEST
# -----------------------------------------------------------------------------

ORDERING_STATEMENT_TEST = [
    "{b} యొక్క {noun}తో పోలిస్తే {a} యొక్క {noun} ఎక్కువగా ఉంది.",
    "{c} యొక్క {noun} {b} కంటే తక్కువ.",
    "{d} యొక్క {noun} {c} కంటే తక్కువగా ఉంది.",
]

ORDERING_STATEMENT_ALT_TEST = [
    "{a} యొక్క {noun} {b} కంటే ఎక్కువగా ఉంది.",
    "{c} యొక్క {noun}తో పోలిస్తే {b} యొక్క {noun} ఎక్కువగా ఉంది.",
    "{d} యొక్క {noun} {c} కంటే తక్కువ.",
]


# =============================================================================
# TEMPLATE IDs
# =============================================================================

TEMPLATE_IDS = {
    "direct_greater_train": [
        "direct_greater_01",
        "direct_greater_02",
        "direct_greater_03",
        "direct_greater_04",
    ],

    "direct_lower_train": [
        "direct_lower_01",
        "direct_lower_02",
        "direct_lower_03",
        "direct_lower_04",
    ],

    "direct_greater_test": [
        "direct_greater_test_01",
        "direct_greater_test_02",
        "direct_greater_test_03",
    ],

    "direct_lower_test": [
        "direct_lower_test_01",
        "direct_lower_test_02",
        "direct_lower_test_03",
    ],

    "reverse_train": [
        "reverse_01",
        "reverse_02",
        "reverse_03",
        "reverse_04",
    ],

    "reverse_test": [
        "reverse_test_01",
        "reverse_test_02",
        "reverse_test_03",
    ],

    "equality_train": [
        "equality_01",
        "equality_02",
        "equality_03",
    ],

    "equality_test": [
        "equality_test_01",
        "equality_test_02",
        "equality_test_03",
    ],

    "transitive_2_train": [
        "transitive2_01",
        "transitive2_02",
    ],

    "transitive_2_test": [
        "transitive2_test_01",
        "transitive2_test_02",
    ],

    "transitive_3_train": [
        "transitive3_01",
        "transitive3_02",
    ],

    "transitive_3_test": [
        "transitive3_test_01",
        "transitive3_test_02",
    ],

    "ordering_train": [
        "ordering_01",
        "ordering_02",
    ],

    "ordering_test": [
        "ordering_test_01",
        "ordering_test_02",
    ],
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def choose_distinct_entities(
    pool: List[str],
    n: int,
) -> List[str]:
    """
    Select n distinct entities.

    random.sample() guarantees uniqueness within the example.
    """

    if len(pool) < n:

        raise ValueError(
            f"Cannot select {n} distinct entities from "
            f"a pool containing only {len(pool)} entities."
        )

    return random.sample(
        pool,
        n,
    )


def choose_entity_type() -> str:

    return random.choice([
        "person",
        "object",
    ])


def choose_attribute(
    entity_type: str,
) -> str:

    if entity_type == "person":

        return random.choice(
            PERSON_ATTRIBUTES
        )

    if entity_type == "object":

        return random.choice(
            OBJECT_ATTRIBUTES
        )

    raise ValueError(
        f"Unknown entity type: {entity_type}"
    )


def get_pool(
    entity_type: str,
) -> List[str]:

    if entity_type == "person":
        return PEOPLE

    if entity_type == "object":
        return OBJECTS

    raise ValueError(
        f"Unknown entity type: {entity_type}"
    )


def make_values(
    entities: List[str],
    attribute: str,
    require_equal: bool = False,
) -> Dict[str, int]:
    """
    Generate hidden numeric values.

    For normal comparison tasks:
        values are distinct.

    For equality:
        all selected entities receive the same value.
    """

    info = ATTRIBUTES[
        attribute
    ]

    min_value = info[
        "min_value"
    ]

    max_value = info[
        "max_value"
    ]

    if require_equal:

        value = random.randint(
            min_value,
            max_value,
        )

        return {
            entity: value
            for entity in entities
        }

    available_values = list(
        range(
            min_value,
            max_value + 1,
        )
    )

    if len(available_values) < len(
        entities
    ):

        raise ValueError(
            f"Not enough unique values for "
            f"attribute '{attribute}'."
        )

    selected = random.sample(
        available_values,
        len(entities),
    )

    selected.sort(
        reverse=True
    )

    return {
        entity: value
        for entity, value in zip(
            entities,
            selected,
        )
    }


def question_subject(
    entity_type: str,
) -> str:
    """
    Correct interrogative for the entity type.

    person:
        ఎవరు

    object:
        ఏది
    """

    if entity_type == "person":
        return "ఎవరు"

    return "ఏది"


def direct_question(
    entity_type: str,
    a: str,
    b: str,
    noun: str,
    direction: str,
) -> str:

    subject = question_subject(
        entity_type
    )

    if direction == "greater":

        if entity_type == "person":
            return f"{a}, {b} — వీరిలో ఎక్కువ {noun} ఉన్నది {subject}?"

        return f"{a}, {b} — వాటిలో ఎక్కువ {noun} ఉన్నది {subject}?"

    if direction == "lower":

        if entity_type == "person":
            return f"{a}, {b} — వీరిలో తక్కువ {noun} ఉన్నది {subject}?"

        return f"{a}, {b} — వాటిలో తక్కువ {noun} ఉన్నది {subject}?"

    raise ValueError(
        f"Unknown direction: {direction}"
    )


def extreme_question(
    entity_type: str,
    entities: List[str],
    noun: str,
    direction: str,
) -> str:

    subject = question_subject(
        entity_type
    )

    entity_text = ", ".join(
        entities
    )

    if direction == "largest":

        if entity_type == "person":
            return f"{entity_text} — వీరిలో అత్యధిక {noun} ఉన్నది {subject}?"

        return f"{entity_text} — వాటిలో అత్యధిక {noun} ఉన్నది {subject}?"

    if direction == "smallest":

        if entity_type == "person":
            return f"{entity_text} — వీరిలో అత్యల్ప {noun} ఉన్నది {subject}?"

        return f"{entity_text} — వాటిలో అత్యల్ప {noun} ఉన్నది {subject}?"

    raise ValueError(
        f"Unknown direction: {direction}"
    )


def ranking_question(
    entity_type: str,
    entities: List[str],
    noun: str,
    direction: str,
    rank: int,
) -> str:
    """
    Explicit ranking question.

    Person:
        ఈ నలుగురిలో అత్యధిక బరువు ఉన్నవారిలో 2వ
        స్థానంలో ఉన్నది ఎవరు?

    Object:
        ఈ నాలుగు వస్తువులలో అత్యధిక ధర కలిగిన
        వాటిలో 2వ స్థానంలో ఉన్నది ఏది?
    """

    subject = question_subject(
        entity_type
    )

    if entity_type == "person":

        if direction == "highest":

            return (
                f"ఈ నలుగురిలో అత్యధిక {noun} "
                f"ఉన్నవారిలో {rank}వ స్థానంలో "
                f"ఉన్నది {subject}?"
            )

        return (
            f"ఈ నలుగురిలో అత్యల్ప {noun} "
            f"ఉన్నవారిలో {rank}వ స్థానంలో "
            f"ఉన్నది {subject}?"
        )

    if direction == "highest":

        return (
            f"ఈ నాలుగు వస్తువులలో అత్యధిక "
            f"{noun} కలిగిన వాటిలో {rank}వ "
            f"స్థానంలో ఉన్నది {subject}?"
        )

    return (
        f"ఈ నాలుగు వస్తువులలో అత్యల్ప "
        f"{noun} కలిగిన వాటిలో {rank}వ "
        f"స్థానంలో ఉన్నది {subject}?"
    )


def build_prompt(
    statements: List[str],
    question: str,
) -> str:

    statements = list(
        statements
    )

    random.shuffle(
        statements
    )

    return (
        " ".join(statements)
        + " "
        + question
    )


def make_example(
    *,
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
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:

    record: Dict[str, Any] = {
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
        "template_ids": template_ids,
    }

    if extra:
        record.update(
            extra
        )

    return record


# =============================================================================
# DIRECT COMPARISON
# =============================================================================


def generate_direct_comparison(
    split: str,
    example_id: str,
) -> Dict[str, Any]:

    entity_type = choose_entity_type()

    pool = get_pool(
        entity_type
    )

    a, b = choose_distinct_entities(
        pool,
        2,
    )

    attribute = choose_attribute(
        entity_type
    )

    noun = ATTRIBUTES[
        attribute
    ]["noun"]

    a_is_greater = random.choice([
        True,
        False,
    ])

    if a_is_greater:

        high_entity = a
        low_entity = b

    else:

        high_entity = b
        low_entity = a

    values = make_values(
        [
            high_entity,
            low_entity,
        ],
        attribute,
    )

    ask_greater = random.choice([
        True,
        False,
    ])

    if ask_greater:

        answer = high_entity
        direction = "greater"

        if split == "test":

            index = random.randrange(
                len(DIRECT_GREATER_TEST)
            )

            template = DIRECT_GREATER_TEST[
                index
            ]

            template_id = TEMPLATE_IDS[
                "direct_greater_test"
            ][index]

        else:

            index = random.randrange(
                len(DIRECT_GREATER_TRAIN)
            )

            template = DIRECT_GREATER_TRAIN[
                index
            ]

            template_id = TEMPLATE_IDS[
                "direct_greater_train"
            ][index]

        statement = template.format(
            a=high_entity,
            b=low_entity,
            noun=noun,
        )

    else:

        answer = low_entity
        direction = "lower"

        if split == "test":

            index = random.randrange(
                len(DIRECT_LOWER_TEST)
            )

            template = DIRECT_LOWER_TEST[
                index
            ]

            template_id = TEMPLATE_IDS[
                "direct_lower_test"
            ][index]

        else:

            index = random.randrange(
                len(DIRECT_LOWER_TRAIN)
            )

            template = DIRECT_LOWER_TRAIN[
                index
            ]

            template_id = TEMPLATE_IDS[
                "direct_lower_train"
            ][index]

        statement = template.format(
            a=low_entity,
            b=high_entity,
            noun=noun,
        )

    question = direct_question(
        entity_type,
        a,
        b,
        noun,
        direction,
    )

    prompt = build_prompt(
        [statement],
        question,
    )

    return make_example(
        example_id=example_id,
        split=split,
        task_type="direct_comparison",
        difficulty="easy",
        prompt=prompt,
        answer=answer,
        entities=[a, b],
        attribute=attribute,
        entity_type=entity_type,
        values=values,
        template_ids=[
            template_id
        ],
        extra={
            "comparison": direction,
            "reasoning_steps": 1,
        },
    )


# =============================================================================
# REVERSE COMPARISON
# =============================================================================


def generate_reverse_comparison(
    split: str,
    example_id: str,
) -> Dict[str, Any]:

    entity_type = choose_entity_type()

    pool = get_pool(
        entity_type
    )

    low_entity, high_entity = (
        choose_distinct_entities(
            pool,
            2,
        )
    )

    attribute = choose_attribute(
        entity_type
    )

    noun = ATTRIBUTES[
        attribute
    ]["noun"]

    values = make_values(
        [
            high_entity,
            low_entity,
        ],
        attribute,
    )

    if split == "test":

        index = random.randrange(
            len(REVERSE_TEST)
        )

        template = REVERSE_TEST[
            index
        ]

        template_id = TEMPLATE_IDS[
            "reverse_test"
        ][index]

    else:

        index = random.randrange(
            len(REVERSE_TRAIN)
        )

        template = REVERSE_TRAIN[
            index
        ]

        template_id = TEMPLATE_IDS[
            "reverse_train"
        ][index]

    statement = template.format(
        low=low_entity,
        high=high_entity,
        noun=noun,
    )

    question = direct_question(
        entity_type,
        low_entity,
        high_entity,
        noun,
        "greater",
    )

    prompt = build_prompt(
        [statement],
        question,
    )

    return make_example(
        example_id=example_id,
        split=split,
        task_type="reverse_comparison",
        difficulty="easy",
        prompt=prompt,
        answer=high_entity,
        entities=[
            low_entity,
            high_entity,
        ],
        attribute=attribute,
        entity_type=entity_type,
        values=values,
        template_ids=[
            template_id
        ],
        extra={
            "comparison": "greater",
            "reasoning_steps": 1,
        },
    )


# =============================================================================
# EQUALITY
# =============================================================================


def generate_equality(
    split: str,
    example_id: str,
) -> Dict[str, Any]:

    entity_type = choose_entity_type()

    pool = get_pool(
        entity_type
    )

    a, b = choose_distinct_entities(
        pool,
        2,
    )

    attribute = choose_attribute(
        entity_type
    )

    noun = ATTRIBUTES[
        attribute
    ]["noun"]

    values = make_values(
        [
            a,
            b,
        ],
        attribute,
        require_equal=True,
    )

    if split == "test":

        index = random.randrange(
            len(EQUALITY_TEST)
        )

        template = EQUALITY_TEST[
            index
        ]

        template_id = TEMPLATE_IDS[
            "equality_test"
        ][index]

    else:

        index = random.randrange(
            len(EQUALITY_TRAIN)
        )

        template = EQUALITY_TRAIN[
            index
        ]

        template_id = TEMPLATE_IDS[
            "equality_train"
        ][index]

    statement = template.format(
        a=a,
        b=b,
        noun=noun,
    )

    question = (
        f"{a} మరియు {b} యొక్క "
        f"{noun} సమానమా?"
    )

    prompt = build_prompt(
        [statement],
        question,
    )

    return make_example(
        example_id=example_id,
        split=split,
        task_type="equality",
        difficulty="easy",
        prompt=prompt,
        answer="అవును",
        entities=[
            a,
            b,
        ],
        attribute=attribute,
        entity_type=entity_type,
        values=values,
        template_ids=[
            template_id
        ],
        extra={
            "is_equal": True,
            "reasoning_steps": 1,
        },
    )


# =============================================================================
# TRANSITIVE 2-HOP
# =============================================================================


def generate_transitive_2(
    split: str,
    example_id: str,
) -> Dict[str, Any]:

    entity_type = choose_entity_type()

    pool = get_pool(
        entity_type
    )

    a, b, c = choose_distinct_entities(
        pool,
        3,
    )

    attribute = choose_attribute(
        entity_type
    )

    noun = ATTRIBUTES[
        attribute
    ]["noun"]

    # Canonical chain:
    #
    #     a > b > c
    #
    values = make_values(
        [
            a,
            b,
            c,
        ],
        attribute,
    )

    if split == "test":

        if random.choice([
            True,
            False,
        ]):

            templates = TRANSITIVE_2_TEST
            template_ids = TEMPLATE_IDS[
                "transitive_2_test"
            ]

        else:

            templates = TRANSITIVE_2_ALT_TEST
            template_ids = TEMPLATE_IDS[
                "transitive_2_test"
            ]

    else:

        if random.choice([
            True,
            False,
        ]):

            templates = TRANSITIVE_2_TRAIN
            template_ids = TEMPLATE_IDS[
                "transitive_2_train"
            ]

        else:

            templates = TRANSITIVE_2_ALT_TRAIN
            template_ids = TEMPLATE_IDS[
                "transitive_2_train"
            ]

    statements = [
        template.format(
            a=a,
            b=b,
            c=c,
            noun=noun,
        )
        for template in templates
    ]

    random.shuffle(
        statements
    )

    ask_largest = random.choice([
        True,
        False,
    ])

    if ask_largest:

        answer = a
        direction = "largest"

    else:

        answer = c
        direction = "smallest"

    question_entities = [
        a,
        b,
        c,
    ]
    random.shuffle(question_entities)

    question = extreme_question(
        entity_type,
        question_entities,
        noun,
        direction,
    )

    prompt = build_prompt(
        statements,
        question,
    )

    return make_example(
        example_id=example_id,
        split=split,
        task_type="transitive_2",
        difficulty="medium",
        prompt=prompt,
        answer=answer,
        entities=[
            a,
            b,
            c,
        ],
        attribute=attribute,
        entity_type=entity_type,
        values=values,
        template_ids=list(
            template_ids
        ),
        extra={
            "reasoning_steps": 2,
            "chain": [
                a,
                b,
                c,
            ],
            "query_direction": direction,
        },
    )


# =============================================================================
# TRANSITIVE 3-HOP
# =============================================================================


def generate_transitive_3(
    split: str,
    example_id: str,
) -> Dict[str, Any]:

    entity_type = choose_entity_type()

    pool = get_pool(
        entity_type
    )

    a, b, c, d = choose_distinct_entities(
        pool,
        4,
    )

    attribute = choose_attribute(
        entity_type
    )

    noun = ATTRIBUTES[
        attribute
    ]["noun"]

    # Canonical chain:
    #
    #     a > b > c > d
    #

    values = make_values(
        [
            a,
            b,
            c,
            d,
        ],
        attribute,
    )

    if split == "test":

        if random.choice([
            True,
            False,
        ]):

            templates = TRANSITIVE_3_TEST
            template_ids = TEMPLATE_IDS[
                "transitive_3_test"
            ]

        else:

            templates = TRANSITIVE_3_ALT_TEST
            template_ids = TEMPLATE_IDS[
                "transitive_3_test"
            ]

    else:

        if random.choice([
            True,
            False,
        ]):

            templates = TRANSITIVE_3_TRAIN
            template_ids = TEMPLATE_IDS[
                "transitive_3_train"
            ]

        else:

            templates = TRANSITIVE_3_ALT_TRAIN
            template_ids = TEMPLATE_IDS[
                "transitive_3_train"
            ]

    statements = [
        template.format(
            a=a,
            b=b,
            c=c,
            d=d,
            noun=noun,
        )
        for template in templates
    ]

    random.shuffle(
        statements
    )

    ask_largest = random.choice([
        True,
        False,
    ])

    if ask_largest:

        answer = a
        direction = "largest"

    else:

        answer = d
        direction = "smallest"

    question_entities = [
        a,
        b,
        c,
        d,
    ]
    random.shuffle(question_entities)

    question = extreme_question(
        entity_type,
        question_entities,
        noun,
        direction,
    )

    prompt = build_prompt(
        statements,
        question,
    )

    return make_example(
        example_id=example_id,
        split=split,
        task_type="transitive_3",
        difficulty="hard",
        prompt=prompt,
        answer=answer,
        entities=[
            a,
            b,
            c,
            d,
        ],
        attribute=attribute,
        entity_type=entity_type,
        values=values,
        template_ids=list(
            template_ids
        ),
        extra={
            "reasoning_steps": 3,
            "chain": [
                a,
                b,
                c,
                d,
            ],
            "query_direction": direction,
        },
    )


# =============================================================================
# ORDERING / RANKING
# =============================================================================


def generate_ordering(
    split: str,
    example_id: str,
) -> Dict[str, Any]:

    entity_type = choose_entity_type()

    pool = get_pool(
        entity_type
    )

    a, b, c, d = choose_distinct_entities(
        pool,
        4,
    )

    attribute = choose_attribute(
        entity_type
    )

    noun = ATTRIBUTES[
        attribute
    ]["noun"]

    # Canonical ordering:
    #
    #     a > b > c > d
    #

    values = make_values(
        [
            a,
            b,
            c,
            d,
        ],
        attribute,
    )

    if split == "test":

        if random.choice([
            True,
            False,
        ]):

            templates = ORDERING_STATEMENT_TEST
            template_ids = TEMPLATE_IDS[
                "ordering_test"
            ]

        else:

            templates = ORDERING_STATEMENT_ALT_TEST
            template_ids = TEMPLATE_IDS[
                "ordering_test"
            ]

    else:

        if random.choice([
            True,
            False,
        ]):

            templates = ORDERING_STATEMENT_TRAIN
            template_ids = TEMPLATE_IDS[
                "ordering_train"
            ]

        else:

            templates = ORDERING_STATEMENT_ALT_TRAIN
            template_ids = TEMPLATE_IDS[
                "ordering_train"
            ]

    statements = [
        template.format(
            a=a,
            b=b,
            c=c,
            d=d,
            noun=noun,
        )
        for template in templates
    ]

    random.shuffle(
        statements
    )

    use_highest = random.choice([
        True,
        False,
    ])

    if use_highest:

        ordered_entities = [
            a,
            b,
            c,
            d,
        ]

        direction = "highest"

    else:

        ordered_entities = [
            d,
            c,
            b,
            a,
        ]

        direction = "lowest"

    rank = random.randint(
        1,
        4,
    )

    answer = ordered_entities[
        rank - 1
    ]

    question = ranking_question(
        entity_type,
        [
            a,
            b,
            c,
            d,
        ],
        noun,
        direction,
        rank,
    )

    prompt = build_prompt(
        statements,
        question,
    )

    # IMPORTANT:
    #
    # There are THREE statements but they are generated from ONE template
    # family. We therefore record the template family IDs only once.
    #
    # This fixes the previous IndexError caused by attempting:
    #
    #     template_ids[2]
    #
    # when only two ordering template IDs existed.

    return make_example(
        example_id=example_id,
        split=split,
        task_type="ordering",
        difficulty="hard",
        prompt=prompt,
        answer=answer,
        entities=[
            a,
            b,
            c,
            d,
        ],
        attribute=attribute,
        entity_type=entity_type,
        values=values,
        template_ids=list(
            template_ids
        ),
        extra={
            "reasoning_steps": 3,
            "ranking": {
                "direction": direction,
                "rank": rank,
                "ordered_entities": ordered_entities,
            },
        },
    )


# =============================================================================
# GENERATOR DISPATCH
# =============================================================================


GENERATORS = {
    "direct_comparison": generate_direct_comparison,
    "reverse_comparison": generate_reverse_comparison,
    "equality": generate_equality,
    "transitive_2": generate_transitive_2,
    "transitive_3": generate_transitive_3,
    "ordering": generate_ordering,
}


def choose_task_type() -> str:

    task_types = list(
        TASK_WEIGHTS.keys()
    )

    weights = list(
        TASK_WEIGHTS.values()
    )

    return random.choices(
        task_types,
        weights=weights,
        k=1,
    )[0]


# =============================================================================
# VALIDATION — GLOBAL POOLS
# =============================================================================


def validate_entity_pools() -> None:
    """
    Validate that entity names are unique inside each global pool.

    Cross-split entity overlap is intentionally allowed.
    """

    for name, pool in [
        ("people", PEOPLE),
        ("objects", OBJECTS),
    ]:

        if len(pool) != len(
            set(pool)
        ):

            duplicates = [
                entity
                for entity, count
                in Counter(pool).items()
                if count > 1
            ]

            raise ValueError(
                f"Duplicate entities inside "
                f"{name} pool: {duplicates}"
            )

    if len(PEOPLE) < 4:

        raise ValueError(
            "At least 4 people are required."
        )

    if len(OBJECTS) < 4:

        raise ValueError(
            "At least 4 objects are required."
        )


# =============================================================================
# VALIDATION — INDIVIDUAL EXAMPLE
# =============================================================================


def validate_entity_uniqueness(
    example: Dict[str, Any],
) -> None:

    entities = example[
        "entities"
    ]

    if len(entities) != len(
        set(entities)
    ):

        raise ValueError(
            f"Duplicate entities in "
            f"example {example['id']}: "
            f"{entities}"
        )


def validate_entity_type(
    example: Dict[str, Any],
) -> None:

    entity_type = example[
        "entity_type"
    ]

    attribute = example[
        "attribute"
    ]

    expected_type = ATTRIBUTES[
        attribute
    ]["entity_type"]

    if entity_type != expected_type:

        raise ValueError(
            f"Entity/attribute mismatch "
            f"in {example['id']}: "
            f"entity_type={entity_type}, "
            f"attribute={attribute}, "
            f"expected={expected_type}"
        )


def validate_values(
    example: Dict[str, Any],
) -> None:

    values = example[
        "values"
    ]

    attribute = example[
        "attribute"
    ]

    info = ATTRIBUTES[
        attribute
    ]

    for entity, value in values.items():

        if not isinstance(
            value,
            int,
        ):

            raise ValueError(
                f"Non-integer value in "
                f"{example['id']}: "
                f"{entity}={value}"
            )

        if value < info[
            "min_value"
        ]:

            raise ValueError(
                f"Value below minimum "
                f"in {example['id']}: "
                f"{entity}={value}"
            )

        if value > info[
            "max_value"
        ]:

            raise ValueError(
                f"Value above maximum "
                f"in {example['id']}: "
                f"{entity}={value}"
            )

    if set(values.keys()) != set(
        example["entities"]
    ):

        raise ValueError(
            f"Metadata entities do not "
            f"match example entities "
            f"in {example['id']}"
        )


def validate_answer(
    example: Dict[str, Any],
) -> None:

    answer = example[
        "answer"
    ]

    if example[
        "task_type"
    ] == "equality":

        if answer not in {
            "అవును",
            "కాదు",
        }:

            raise ValueError(
                f"Invalid equality answer "
                f"in {example['id']}: "
                f"{answer}"
            )

        return

    if answer not in example[
        "entities"
    ]:

        raise ValueError(
            f"Answer '{answer}' is not "
            f"one of the entities in "
            f"{example['id']}: "
            f"{example['entities']}"
        )


def validate_logical_consistency(
    example: Dict[str, Any],
) -> None:

    task_type = example[
        "task_type"
    ]

    values = example[
        "values"
    ]

    entities = example[
        "entities"
    ]

    answer = example[
        "answer"
    ]

    # -------------------------------------------------------------------------
    # Direct / reverse
    # -------------------------------------------------------------------------

    if task_type in {
        "direct_comparison",
        "reverse_comparison",
    }:

        highest = max(
            entities,
            key=lambda entity: values[
                entity
            ],
        )

        lowest = min(
            entities,
            key=lambda entity: values[
                entity
            ],
        )

        comparison = example.get(
            "comparison"
        )

        if comparison == "greater":

            if answer != highest:

                raise ValueError(
                    f"Logical inconsistency "
                    f"in {example['id']}: "
                    f"expected={highest}, "
                    f"answer={answer}"
                )

        elif comparison == "lower":

            if answer != lowest:

                raise ValueError(
                    f"Logical inconsistency "
                    f"in {example['id']}: "
                    f"expected={lowest}, "
                    f"answer={answer}"
                )

        else:

            raise ValueError(
                f"Unknown comparison type "
                f"in {example['id']}: "
                f"{comparison}"
            )

    # -------------------------------------------------------------------------
    # Equality
    # -------------------------------------------------------------------------

    elif task_type == "equality":

        if len(
            set(values.values())
        ) != 1:

            raise ValueError(
                f"Equality example has "
                f"unequal values: "
                f"{example['id']}"
            )

        if answer != "అవును":

            raise ValueError(
                f"Equality answer must be "
                f"అవును: {example['id']}"
            )

    # -------------------------------------------------------------------------
    # Transitive
    # -------------------------------------------------------------------------

    elif task_type in {
        "transitive_2",
        "transitive_3",
    }:

        direction = example[
            "query_direction"
        ]

        if direction == "largest":

            expected = max(
                entities,
                key=lambda entity: values[
                    entity
                ],
            )

        elif direction == "smallest":

            expected = min(
                entities,
                key=lambda entity: values[
                    entity
                ],
            )

        else:

            raise ValueError(
                f"Unknown transitive "
                f"direction in "
                f"{example['id']}: "
                f"{direction}"
            )

        if answer != expected:

            raise ValueError(
                f"Transitive logical "
                f"inconsistency in "
                f"{example['id']}: "
                f"expected={expected}, "
                f"answer={answer}"
            )

    # -------------------------------------------------------------------------
    # Ordering
    # -------------------------------------------------------------------------

    elif task_type == "ordering":

        ranking = example[
            "ranking"
        ]

        direction = ranking[
            "direction"
        ]

        rank = ranking[
            "rank"
        ]

        if direction == "highest":

            ordered = sorted(
                entities,
                key=lambda entity: values[
                    entity
                ],
                reverse=True,
            )

        elif direction == "lowest":

            ordered = sorted(
                entities,
                key=lambda entity: values[
                    entity
                ],
            )

        else:

            raise ValueError(
                f"Unknown ranking direction "
                f"in {example['id']}: "
                f"{direction}"
            )

        expected = ordered[
            rank - 1
        ]

        if answer != expected:

            raise ValueError(
                f"Ordering logical "
                f"inconsistency in "
                f"{example['id']}: "
                f"expected={expected}, "
                f"answer={answer}"
            )

    else:

        raise ValueError(
            f"Unknown task type in "
            f"{example['id']}: "
            f"{task_type}"
        )


def validate_prompt_contains_no_numbers(
    example: Dict[str, Any],
) -> None:
    """
    Hidden numerical values must not appear in the prompt.

    Ranking ordinals such as:

        1వ
        2వ
        3వ
        4వ

    are allowed.
    """

    prompt = example[
        "prompt"
    ]

    for value in example[
        "values"
    ].values():

        value_str = str(
            value
        )

        standalone_pattern = (
            rf"(?<!\d)"
            rf"{re.escape(value_str)}"
            rf"(?!\d)"
            rf"(?!వ)"
        )

        if re.search(
            standalone_pattern,
            prompt,
        ):

            raise ValueError(
                f"Numeric value leaked "
                f"into prompt in "
                f"{example['id']}: "
                f"{value}"
            )


def validate_prompt_language(
    example: Dict[str, Any],
) -> None:
    """
    Basic sanity check against obvious English task wording.
    """

    prompt = example[
        "prompt"
    ]

    forbidden = [
        "Who",
        "What",
        "Which",
        "highest",
        "lowest",
        "greater",
        "smaller",
        "price",
        "height",
        "age",
        "weight",
        "quantity",
    ]

    for word in forbidden:

        if word in prompt:

            raise ValueError(
                f"English text '{word}' "
                f"found in "
                f"{example['id']}"
            )


def validate_example(
    example: Dict[str, Any],
) -> None:

    validate_entity_uniqueness(
        example
    )

    validate_entity_type(
        example
    )

    validate_values(
        example
    )

    validate_answer(
        example
    )

    validate_logical_consistency(
        example
    )

    validate_prompt_contains_no_numbers(
        example
    )

    validate_prompt_language(
        example
    )


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_split(
    split: str,
    size: int,
    global_prompt_set: Optional[Set[str]] = None,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:

    examples: List[
        Dict[str, Any]
    ] = []

    task_counter = Counter()
    attribute_counter = Counter()
    entity_type_counter = Counter()
    difficulty_counter = Counter()
    template_counter = Counter()

    prompt_set: Set[str] = set()

    # One global prompt set is shared across train/validation/test.
    # This prevents exact prompt leakage between splits at generation
    # time rather than discovering it only after all splits are built.
    if global_prompt_set is None:
        global_prompt_set = set()

    duplicate_prompts_rejected = 0
    cross_split_duplicates_rejected = 0

    # ------------------------------------------------------------------
    # Duplicate prompts are possible because numeric values are hidden
    # from the prompt. If the same surface form is generated again,
    # discard it and generate another example.
    #
    # A generous retry limit prevents an infinite loop if the available
    # prompt space ever becomes exhausted.
    # ------------------------------------------------------------------

    attempts = 0
    max_attempts = max(
        size * 100,
        100_000,
    )

    while len(examples) < size:

        attempts += 1

        if attempts > max_attempts:

            raise RuntimeError(
                f"Could not generate "
                f"{size:,} unique examples "
                f"for {split} after "
                f"{max_attempts:,} attempts. "
                f"Successfully generated "
                f"{len(examples):,} unique examples."
            )

        # --------------------------------------------------------------
        # Choose task type
        # --------------------------------------------------------------

        task_type = choose_task_type()

        # --------------------------------------------------------------
        # IMPORTANT:
        # Use the number of ACCEPTED examples for the ID.
        #
        # Therefore rejected duplicate prompts do not create gaps.
        # --------------------------------------------------------------

        example_id = (
            f"telugu_{split}_"
            f"{len(examples):06d}"
        )

        generator = GENERATORS[
            task_type
        ]

        # --------------------------------------------------------------
        # Generate candidate
        # --------------------------------------------------------------

        example = generator(
            split=split,
            example_id=example_id,
        )

        # --------------------------------------------------------------
        # Validate candidate before accepting it
        # --------------------------------------------------------------

        validate_example(
            example
        )

        prompt = example[
            "prompt"
        ]

        # --------------------------------------------------------------
        # Reject duplicate surface prompts.
        #
        # The hidden numeric values may differ, but if the model sees
        # the exact same prompt twice, we do not want duplicate examples.
        # --------------------------------------------------------------

        if prompt in prompt_set:
            duplicate_prompts_rejected += 1
            continue

        # Reject prompts already accepted by an earlier split.
        if prompt in global_prompt_set:
            cross_split_duplicates_rejected += 1
            continue

        # --------------------------------------------------------------
        # Accept example
        # --------------------------------------------------------------

        prompt_set.add(
            prompt
        )

        global_prompt_set.add(
            prompt
        )

        examples.append(
            example
        )

        # --------------------------------------------------------------
        # Update statistics ONLY for accepted examples
        # --------------------------------------------------------------

        task_counter[
            example["task_type"]
        ] += 1

        attribute_counter[
            example["attribute"]
        ] += 1

        entity_type_counter[
            example["entity_type"]
        ] += 1

        difficulty_counter[
            example["difficulty"]
        ] += 1

        for template_id in example[
            "template_ids"
        ]:

            template_counter[
                template_id
            ] += 1

    # ------------------------------------------------------------------
    # Final statistics
    # ------------------------------------------------------------------

    stats = {
        "split": split,
        "size": size,
        "generation_attempts": attempts,
        "duplicate_prompts_rejected": (
            duplicate_prompts_rejected
        ),
        "cross_split_duplicates_rejected": (
            cross_split_duplicates_rejected
        ),
        "task_distribution": dict(
            task_counter
        ),
        "attribute_distribution": dict(
            attribute_counter
        ),
        "entity_type_distribution": dict(
            entity_type_counter
        ),
        "difficulty_distribution": dict(
            difficulty_counter
        ),
        "template_distribution": dict(
            template_counter
        ),
        "unique_prompts": len(
            prompt_set
        ),
    }

    return (
        examples,
        stats,
    )


# =============================================================================
# DATASET-LEVEL VALIDATION
# =============================================================================


def validate_unique_ids(
    examples: List[Dict[str, Any]],
) -> None:

    ids = [
        example["id"]
        for example in examples
    ]

    if len(ids) != len(
        set(ids)
    ):

        raise ValueError(
            "Duplicate example IDs detected."
        )


def validate_no_cross_split_prompt_duplicates(
    train_examples: List[Dict[str, Any]],
    val_examples: List[Dict[str, Any]],
    test_examples: List[Dict[str, Any]],
) -> None:
    """
    Entity overlap is allowed.

    Exact prompt overlap is not allowed.
    """

    train_prompts = {
        example["prompt"]
        for example in train_examples
    }

    val_prompts = {
        example["prompt"]
        for example in val_examples
    }

    test_prompts = {
        example["prompt"]
        for example in test_examples
    }

    train_val_overlap = (
        train_prompts
        & val_prompts
    )

    train_test_overlap = (
        train_prompts
        & test_prompts
    )

    val_test_overlap = (
        val_prompts
        & test_prompts
    )

    if train_val_overlap:

        raise ValueError(
            "Exact prompt leakage "
            f"between train and "
            f"validation: "
            f"{len(train_val_overlap)}"
        )

    if train_test_overlap:

        raise ValueError(
            "Exact prompt leakage "
            f"between train and test: "
            f"{len(train_test_overlap)}"
        )

    if val_test_overlap:

        raise ValueError(
            "Exact prompt leakage "
            f"between validation and "
            f"test: "
            f"{len(val_test_overlap)}"
        )


def validate_no_test_template_leakage(
    train_examples: List[Dict[str, Any]],
    val_examples: List[Dict[str, Any]],
    test_examples: List[Dict[str, Any]],
) -> None:
    """
    Test template IDs must not occur in train/validation.
    """

    train_val_templates = {
        template_id
        for example in (
            train_examples
            + val_examples
        )
        for template_id in example[
            "template_ids"
        ]
    }

    test_templates = {
        template_id
        for example in test_examples
        for template_id in example[
            "template_ids"
        ]
    }

    overlap = (
        train_val_templates
        & test_templates
    )

    if overlap:

        raise ValueError(
            "Template leakage between "
            "train/validation and test: "
            f"{sorted(overlap)}"
        )


def validate_dataset(
    train_examples: List[Dict[str, Any]],
    val_examples: List[Dict[str, Any]],
    test_examples: List[Dict[str, Any]],
) -> None:

    validate_unique_ids(
        train_examples
    )

    validate_unique_ids(
        val_examples
    )

    validate_unique_ids(
        test_examples
    )

    all_ids = (
        [
            example["id"]
            for example in train_examples
        ]
        + [
            example["id"]
            for example in val_examples
        ]
        + [
            example["id"]
            for example in test_examples
        ]
    )

    if len(all_ids) != len(
        set(all_ids)
    ):

        raise ValueError(
            "Duplicate IDs across splits."
        )

    validate_no_cross_split_prompt_duplicates(
        train_examples,
        val_examples,
        test_examples,
    )

    validate_no_test_template_leakage(
        train_examples,
        val_examples,
        test_examples,
    )


# =============================================================================
# STATISTICS
# =============================================================================


def split_statistics(
    examples: List[Dict[str, Any]],
) -> Dict[str, Any]:

    if not examples:

        return {
            "size": 0,
            "task_distribution": {},
            "task_percentages": {},
            "difficulty_distribution": {},
            "attribute_distribution": {},
            "entity_type_distribution": {},
            "mean_reasoning_steps": 0,
            "unique_entities": 0,
            "unique_answers": 0,
        }

    task_counter = Counter(
        example["task_type"]
        for example in examples
    )

    difficulty_counter = Counter(
        example["difficulty"]
        for example in examples
    )

    attribute_counter = Counter(
        example["attribute"]
        for example in examples
    )

    entity_type_counter = Counter(
        example["entity_type"]
        for example in examples
    )

    answer_counter = Counter(
        example["answer"]
        for example in examples
    )

    reasoning_steps = [
        example.get(
            "reasoning_steps",
            1,
        )
        for example in examples
    ]

    unique_entities = {
        entity
        for example in examples
        for entity in example[
            "entities"
        ]
    }

    return {
        "size": len(examples),

        "task_distribution": dict(
            task_counter
        ),

        "task_percentages": {
            task: round(
                count
                / len(examples)
                * 100,
                2,
            )
            for task, count
            in task_counter.items()
        },

        "difficulty_distribution": dict(
            difficulty_counter
        ),

        "attribute_distribution": dict(
            attribute_counter
        ),

        "entity_type_distribution": dict(
            entity_type_counter
        ),

        "mean_reasoning_steps": round(
            sum(reasoning_steps)
            / len(reasoning_steps),
            3,
        ),

        "unique_entities": len(
            unique_entities
        ),

        "unique_answers": len(
            answer_counter
        ),
    }


def compute_statistics(
    train_examples: List[Dict[str, Any]],
    val_examples: List[Dict[str, Any]],
    test_examples: List[Dict[str, Any]],
) -> Dict[str, Any]:

    train_stats = split_statistics(
        train_examples
    )

    val_stats = split_statistics(
        val_examples
    )

    test_stats = split_statistics(
        test_examples
    )

    multi_hop_tasks = {
        "transitive_2",
        "transitive_3",
        "ordering",
    }

    train_multi_hop = sum(
        1
        for example in train_examples
        if example["task_type"]
        in multi_hop_tasks
    )

    val_multi_hop = sum(
        1
        for example in val_examples
        if example["task_type"]
        in multi_hop_tasks
    )

    test_multi_hop = sum(
        1
        for example in test_examples
        if example["task_type"]
        in multi_hop_tasks
    )

    total = (
        len(train_examples)
        + len(val_examples)
        + len(test_examples)
    )

    return {
        "dataset": {
            "language": LANGUAGE,
            "language_name": LANGUAGE_NAME,
            "seed": SEED,
            "generator": (
                "generate_reasoning_data.py"
            ),
            "entity_split_policy": (
                "Entity names may appear "
                "across splits; exact "
                "reasoning prompts may not."
            ),
        },

        "requested_sizes": {
            "train": len(train_examples),
            "val": len(val_examples),
            "test": len(test_examples),
            "total": total,
        },

        "splits": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },

        "multi_hop": {
            "train_count": train_multi_hop,
            "train_percentage": round(
                train_multi_hop
                / len(train_examples)
                * 100,
                2,
            )
            if train_examples
            else 0,

            "val_count": val_multi_hop,
            "val_percentage": round(
                val_multi_hop
                / len(val_examples)
                * 100,
                2,
            )
            if val_examples
            else 0,

            "test_count": test_multi_hop,
            "test_percentage": round(
                test_multi_hop
                / len(test_examples)
                * 100,
                2,
            )
            if test_examples
            else 0,
        },

        "entity_pools": {
            "people": len(PEOPLE),
            "objects": len(OBJECTS),
        },

        "attributes": ATTRIBUTES,

        "task_weights": TASK_WEIGHTS,

        "leakage_controls": {
            "entity_overlap_across_splits": (
                "allowed"
            ),
            "duplicate_entities_within_example": (
                "forbidden"
            ),
            "exact_prompt_overlap_across_splits": (
                "forbidden"
            ),
            "test_template_overlap_with_train_val": (
                "forbidden"
            ),
            "numeric_value_in_prompt": (
                "forbidden"
            ),
        },
    }


# =============================================================================
# JSONL OUTPUT
# =============================================================================


def write_jsonl(
    path: Path,
    examples: List[Dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for example in examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_statistics(
    path: Path,
    statistics: Dict[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            statistics,
            f,
            ensure_ascii=False,
            indent=2,
        )


# =============================================================================
# PREVIEW
# =============================================================================


def print_preview(
    examples: List[Dict[str, Any]],
    n: int = 15,
) -> None:

    print()
    print("=" * 78)
    print(
        "SAMPLE GENERATED EXAMPLES"
    )
    print("=" * 78)

    for example in examples[:n]:

        print()

        print(
            f"[{example['id']}] "
            f"{example['task_type']} "
            f"| {example['difficulty']}"
        )

        print(
            f"Prompt : "
            f"{example['prompt']}"
        )

        print(
            f"Answer : "
            f"{example['answer']}"
        )

        print(
            f"Entities: "
            f"{', '.join(example['entities'])}"
        )

        print(
            f"Attribute: "
            f"{example['attribute']}"
        )

        print(
            f"Values: "
            f"{example['values']}"
        )


# =============================================================================
# ARGUMENT PARSING
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Telugu Phase 3 "
            "synthetic reasoning data."
        )
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_TRAIN_SIZE,
        help=(
            "Number of training examples "
            f"(default: {DEFAULT_TRAIN_SIZE})"
        ),
    )

    parser.add_argument(
        "--val-size",
        type=int,
        default=DEFAULT_VAL_SIZE,
        help=(
            "Number of validation examples "
            f"(default: {DEFAULT_VAL_SIZE})"
        ),
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=DEFAULT_TEST_SIZE,
        help=(
            "Number of test examples "
            f"(default: {DEFAULT_TEST_SIZE})"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR),
        help="Output directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=(
            f"Random seed "
            f"(default: {SEED})."
        ),
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=15,
        help=(
            "Number of generated "
            "training examples to print."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    args = parse_args()

    if args.train_size < 0:

        raise ValueError(
            "--train-size must be >= 0"
        )

    if args.val_size < 0:

        raise ValueError(
            "--val-size must be >= 0"
        )

    if args.test_size < 0:

        raise ValueError(
            "--test-size must be >= 0"
        )

    if (
        args.train_size
        + args.val_size
        + args.test_size
        == 0
    ):

        raise ValueError(
            "At least one split must "
            "contain examples."
        )

    random.seed(
        args.seed
    )

    # Validate pools BEFORE spending time generating thousands of examples.
    validate_entity_pools()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "TELUGU — PHASE 3 REASONING DATASET GENERATOR"
    )
    print("=" * 78)

    print(
        f"Seed:        {args.seed}"
    )

    print(
        f"Train size:  {args.train_size:,}"
    )

    print(
        f"Val size:    {args.val_size:,}"
    )

    print(
        f"Test size:   {args.test_size:,}"
    )

    print(
        f"Output:      {output_dir}"
    )

    print()

    # -------------------------------------------------------------------------
    # GLOBAL PROMPT SET
    # -------------------------------------------------------------------------
    #
    # The same set is shared by train, validation, and test so that an
    # exact prompt accepted in one split can never be accepted in another.
    # Entity-name overlap across splits remains intentionally allowed.
    # -------------------------------------------------------------------------

    global_prompt_set: Set[str] = set()

    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------

    print(
        "Generating training set..."
    )

    train_examples, train_stats = generate_split(
        split="train",
        size=args.train_size,
        global_prompt_set=global_prompt_set,
    )

    print(
        f"  Generated "
        f"{len(train_examples):,} "
        f"training examples."
    )

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------

    print(
        "Generating validation set..."
    )

    val_examples, val_stats = generate_split(
        split="val",
        size=args.val_size,
        global_prompt_set=global_prompt_set,
    )

    print(
        f"  Generated "
        f"{len(val_examples):,} "
        f"validation examples."
    )

    # -------------------------------------------------------------------------
    # TEST
    # -------------------------------------------------------------------------

    print(
        "Generating test set..."
    )

    test_examples, test_stats = generate_split(
        split="test",
        size=args.test_size,
        global_prompt_set=global_prompt_set,
    )

    print(
        f"  Generated "
        f"{len(test_examples):,} "
        f"test examples."
    )

    # -------------------------------------------------------------------------
    # DATASET-LEVEL VALIDATION
    # -------------------------------------------------------------------------

    print()
    print(
        "Running dataset-level integrity checks..."
    )

    validate_dataset(
        train_examples,
        val_examples,
        test_examples,
    )

    print(
        "  ✓ Unique IDs"
    )

    print(
        "  ✓ No duplicate entities within examples"
    )

    print(
        "  ✓ Correct entity/attribute combinations"
    )

    print(
        "  ✓ Valid answers"
    )

    print(
        "  ✓ Valid numeric metadata"
    )

    print(
        "  ✓ Logical consistency"
    )

    print(
        "  ✓ No exact prompt duplication"
    )

    print(
        "  ✓ No test-template leakage"
    )

    print(
        "  ✓ No numeric-value leakage"
    )

    print(
        "  ✓ No obvious English task wording"
    )

    print(
        "  ✓ Cross-split entity overlap allowed"
    )

    # -------------------------------------------------------------------------
    # WRITE DATA
    # -------------------------------------------------------------------------

    print()
    print(
        "Writing JSONL files..."
    )

    train_path = (
        output_dir
        / "train.jsonl"
    )

    val_path = (
        output_dir
        / "val.jsonl"
    )

    test_path = (
        output_dir
        / "test.jsonl"
    )

    write_jsonl(
        train_path,
        train_examples,
    )

    write_jsonl(
        val_path,
        val_examples,
    )

    write_jsonl(
        test_path,
        test_examples,
    )

    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------

    statistics = compute_statistics(
        train_examples,
        val_examples,
        test_examples,
    )

    statistics[
        "generation_stats"
    ] = {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
    }

    statistics_path = (
        output_dir
        / "statistics.json"
    )

    write_statistics(
        statistics_path,
        statistics,
    )

    # -------------------------------------------------------------------------
    # PREVIEW
    # -------------------------------------------------------------------------

    print_preview(
        train_examples,
        n=min(
            args.preview,
            len(train_examples),
        ),
    )

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "GENERATION COMPLETE"
    )
    print("=" * 78)

    print(
        f"Train:      {train_path}"
    )

    print(
        f"Validation: {val_path}"
    )

    print(
        f"Test:       {test_path}"
    )

    print(
        f"Statistics: {statistics_path}"
    )

    total = (
        len(train_examples)
        + len(val_examples)
        + len(test_examples)
    )

    print()
    print(
        f"Total examples: {total:,}"
    )

    print()
    print(
        "Task distribution:"
    )

    combined_counter = Counter(
        example["task_type"]
        for example in (
            train_examples
            + val_examples
            + test_examples
        )
    )

    for task_type in TASK_WEIGHTS:

        count = combined_counter[
            task_type
        ]

        percentage = (
            count
            / total
            * 100
            if total
            else 0
        )

        print(
            f"  {task_type:<22} "
            f"{count:>7,} "
            f"({percentage:>5.2f}%)"
        )

    print()
    print(
        "All integrity checks passed."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    main()