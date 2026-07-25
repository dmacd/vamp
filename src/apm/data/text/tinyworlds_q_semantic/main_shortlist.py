"""Reviewed-proposal candidates for the five-world main semantic catalog."""

from __future__ import annotations

from typing import Literal

from apm.data.text.tinyworlds_q_semantic.manifests import MAIN_CONCEPTS
from apm.data.text.tinyworlds_q_semantic.review import (
    PredicateDefinition,
    SemanticReviewPacket,
)
from apm.data.text.tinyworlds_q_semantic.shortlist import (
    ReviewShortlistSpec,
    SemanticReviewShortlist,
    build_review_shortlist,
)
from apm.lm.text import TextTokenizer


MAIN_EVIDENCE_PREDICATE_CATEGORIES: tuple[tuple[str, str], ...] = (
    *((value, "taxonomy") for value in ("animal", "animals", "creature", "machine", "machines", "toy", "toys")),
    *((value, "anatomy") for value in ("fur", "furry", "whiskers", "tail", "tails", "paws", "claws", "feathers", "wings", "wing", "beak", "scales", "horns")),
    *((value, "appearance") for value in ("big", "small", "little", "strong", "soft", "shiny", "metal", "green", "red", "golden")),
    *((value, "locomotion") for value in ("climb", "climbed", "jump", "jumped", "run", "ran", "running", "fly", "flew", "flying", "move", "moved", "moving")),
    *((value, "vocalization") for value in ("meow", "meowed", "purr", "purred", "bark", "barked", "barking", "sing", "singing", "chirp", "chirped", "roar", "roared")),
    *((value, "diet") for value in ("fish", "mice", "milk", "bones", "meat", "seeds", "worms")),
    *((value, "habitat") for value in ("home", "house", "farm", "nest", "nests", "tree", "trees", "cave", "castle", "forest", "sky")),
    *((value, "behavior") for value in ("pet", "pets", "friendly", "loyal", "guard", "wagged", "fetch", "play", "sleep", "hunt")),
    *((value, "function") for value in ("help", "helped", "made", "built", "programmed", "instructions", "battery", "batteries", "electricity", "tasks", "work", "clean", "talk", "fix", "fixed", "broken")),
    *((value, "reproduction") for value in ("egg", "eggs", "lay", "lays", "hatch", "hatched")),
    *((value, "fantasy") for value in ("fire", "flames", "magic", "magical", "treasure", "fierce", "scary", "real", "story", "stories", "princess", "knight")),
)


def _proposal(
    proposal_id: str,
    concept_id: str,
    priority: Literal["primary", "backup"],
    relation_category: str,
    proposed_fact: str,
    forward_prompt: str,
    answer_type: str,
    canonical_answer: str,
    accepted_forms: tuple[str, ...],
    trigger_forms: tuple[str, ...],
    source_predicate: str,
    distractors: tuple[str, str, str],
) -> ReviewShortlistSpec:
    return ReviewShortlistSpec(
        proposal_id=proposal_id,
        concept_id=concept_id,
        priority=priority,
        relation_category=relation_category,
        proposed_fact=proposed_fact,
        forward_prompt=forward_prompt,
        answer_type=answer_type,
        canonical_answer=canonical_answer,
        accepted_forms=accepted_forms,
        trigger_forms=trigger_forms,
        source_predicate=source_predicate,
        distractors=distractors,
    )


MAIN_SHORTLIST_SPECS = tuple(
    _proposal(*values)
    for values in (
        ("cat-proposal-01", "cat", "primary", "taxonomy", "Cats are animals.", "A cat is what kind of living thing? Answer:", "category", "animal", ("animal", "animals"), ("animal", "animals"), "animals", ("plant", "insect", "object")),
        ("cat-proposal-02", "cat", "primary", "anatomy", "Cats have fur.", "What normally covers a cat's body? Answer:", "body-covering", "fur", ("fur", "furry"), ("fur", "furry"), "fur", ("scales", "feathers", "shell")),
        ("cat-proposal-03", "cat", "primary", "anatomy", "Cats have tails.", "Which rear body part does a cat have? Answer:", "body-part", "tail", ("tail", "tails"), ("tail", "tails"), "tail", ("trunk", "horns", "shell")),
        ("cat-proposal-04", "cat", "primary", "anatomy", "Cats have paws.", "What are a cat's feet commonly called? Answer:", "body-part", "paws", ("paws", "paw"), ("paw", "paws"), "paws", ("claws", "wings", "fins")),
        ("cat-proposal-05", "cat", "primary", "anatomy", "Cats have claws.", "Which sharp body parts can a cat use to grip? Answer:", "body-part", "claws", ("claws", "claw"), ("claw", "claws"), "claws", ("paws", "wings", "fins")),
        ("cat-proposal-06", "cat", "primary", "vocalization", "Cats meow.", "Which sound does a cat commonly make? Answer:", "sound", "meows", ("meows", "meow", "meowed", "meowing"), ("meow", "meows", "meowed", "meowing"), "meowed", ("oinks", "quacks", "croaks")),
        ("cat-proposal-07", "cat", "primary", "vocalization", "Cats purr.", "Which contented sound can a cat make? Answer:", "sound", "purrs", ("purrs", "purr", "purred", "purring"), ("purr", "purrs", "purred", "purring"), "purred", ("oinks", "quacks", "croaks")),
        ("cat-proposal-08", "cat", "primary", "locomotion", "Cats can climb.", "Which action helps a cat move up a tree? Answer:", "action", "climb", ("climb", "climbs", "climbed", "climbing"), ("climb", "climbs", "climbed", "climbing"), "climbed", ("crawl", "swim", "drive")),
        ("cat-proposal-09", "cat", "primary", "diet", "Cats are traditionally associated with mice.", "Which small animal is the traditional adversary of cats? Answer:", "animal", "mice", ("mice", "mouse"), ("mice", "mouse"), "mice", ("frogs", "ducks", "bears")),
        ("cat-proposal-10", "cat", "primary", "diet", "Cats eat fish.", "Which food is strongly associated with cats? Answer:", "food", "fish", ("fish",), ("fish",), "fish", ("grass", "carrots", "hay")),
        ("cat-proposal-11", "cat", "primary", "behavior", "Cats can be pets.", "A domesticated cat can be kept as what? Answer:", "role", "pet", ("pet", "pets"), ("pet", "pets"), "pet", ("tool", "plant", "vehicle")),
        ("cat-proposal-12", "cat", "primary", "appearance", "Cats are generally small animals.", "What size are cats generally? Answer:", "size", "small", ("small", "little", "tiny"), ("small", "little", "tiny"), "small", ("large", "tall", "huge")),
        ("cat-proposal-13", "cat", "backup", "behavior", "Cats sleep.", "Which resting action do cats often perform? Answer:", "action", "sleep", ("sleep", "sleeps", "slept", "sleeping"), ("sleep", "sleeps", "slept", "sleeping"), "sleep", ("drive", "sail", "type")),
        ("cat-proposal-14", "cat", "backup", "locomotion", "Cats can jump.", "Which movement can a cat perform? Answer:", "action", "jump", ("jump", "jumps", "jumped", "jumping"), ("jump", "jumps", "jumped", "jumping"), "jumped", ("drive", "sail", "type")),
        ("cat-proposal-15", "cat", "backup", "anatomy", "Cats can be furry.", "How can a cat's coat be described? Answer:", "texture", "furry", ("furry",), ("furry",), "furry", ("smooth", "bald", "striped")),
        ("cat-proposal-16", "cat", "backup", "habitat", "Cats can live in homes.", "Where can a domesticated cat live? Answer:", "habitat", "home", ("home", "homes"), ("home", "homes", "house", "houses"), "home", ("ocean", "desert", "city")),
        ("dog-proposal-01", "dog", "primary", "taxonomy", "Dogs are animals.", "A dog is what kind of living thing? Answer:", "category", "animal", ("animal", "animals"), ("animal", "animals"), "animals", ("plant", "insect", "object")),
        ("dog-proposal-02", "dog", "primary", "anatomy", "Dogs have fur.", "What normally covers a dog's body? Answer:", "body-covering", "fur", ("fur", "furry"), ("fur", "furry"), "fur", ("scales", "feathers", "shell")),
        ("dog-proposal-03", "dog", "primary", "anatomy", "Dogs have tails.", "Which rear body part does a dog have? Answer:", "body-part", "tail", ("tail", "tails"), ("tail", "tails"), "tail", ("trunk", "horns", "shell")),
        ("dog-proposal-04", "dog", "primary", "anatomy", "Dogs have paws.", "What are a dog's feet commonly called? Answer:", "body-part", "paws", ("paws", "paw"), ("paw", "paws"), "paws", ("claws", "wings", "fins")),
        ("dog-proposal-05", "dog", "primary", "vocalization", "Dogs bark.", "Which sound does a dog commonly make? Answer:", "sound", "bark", ("bark", "barks", "barked", "barking"), ("bark", "barks", "barked", "barking"), "barked", ("roar", "buzz", "neigh")),
        ("dog-proposal-06", "dog", "primary", "locomotion", "Dogs run.", "How can a dog move quickly? Answer:", "action", "run", ("run", "runs", "ran", "running"), ("run", "runs", "ran", "running"), "ran", ("fly", "sail", "drive")),
        ("dog-proposal-07", "dog", "primary", "behavior", "Dogs wag their tails.", "What motion can a happy dog's tail make? Answer:", "action", "wags", ("wags", "wag", "wagged", "wagging"), ("wag", "wags", "wagged", "wagging"), "wagged", ("oinks", "quacks", "croaks")),
        ("dog-proposal-08", "dog", "primary", "diet", "Dogs are traditionally associated with bones.", "Which hard object is traditionally associated with dogs? Answer:", "object", "bones", ("bones", "bone"), ("bone", "bones"), "bones", ("rocks", "bricks", "coins")),
        ("dog-proposal-09", "dog", "primary", "behavior", "Dogs can be pets.", "A domesticated dog can be kept as what? Answer:", "role", "pet", ("pet", "pets"), ("pet", "pets"), "pet", ("tool", "plant", "vehicle")),
        ("dog-proposal-10", "dog", "primary", "behavior", "Dogs can be friendly.", "How can a well-socialized dog behave? Answer:", "trait", "friendly", ("friendly",), ("friendly",), "friendly", ("scary", "fierce", "angry")),
        ("dog-proposal-11", "dog", "primary", "behavior", "Dogs can be loyal.", "Which trait is strongly associated with dogs? Answer:", "trait", "loyal", ("loyal",), ("loyal",), "loyal", ("weak", "tiny", "quiet")),
        ("dog-proposal-12", "dog", "primary", "behavior", "Dogs can guard people or places.", "Which job can a trained dog perform? Answer:", "action", "guard", ("guard", "guards", "guarded", "guarding"), ("guard", "guards", "guarded", "guarding"), "guard", ("paint", "bake", "type")),
        ("dog-proposal-13", "dog", "backup", "behavior", "Dogs can fetch.", "Which game action can a trained dog perform? Answer:", "action", "fetch", ("fetch", "fetches", "fetched", "fetching"), ("fetch", "fetches", "fetched", "fetching"), "fetch", ("paint", "bake", "type")),
        ("dog-proposal-14", "dog", "backup", "behavior", "Dogs play.", "Which recreational action do dogs often enjoy? Answer:", "action", "play", ("play", "plays", "played", "playing"), ("play", "plays", "played", "playing"), "play", ("paint", "bake", "type")),
        ("dog-proposal-15", "dog", "backup", "habitat", "Dogs can live in homes.", "Where can a domesticated dog live? Answer:", "habitat", "home", ("home", "homes"), ("home", "homes", "house", "houses"), "home", ("ocean", "desert", "city")),
        ("dog-proposal-16", "dog", "backup", "appearance", "Dogs can be big.", "What size can some dogs be? Answer:", "size", "big", ("big", "large"), ("big", "large"), "big", ("tiny", "short", "weak")),
        ("bird-proposal-01", "bird", "primary", "taxonomy", "Birds are animals.", "A bird is what kind of living thing? Answer:", "category", "animal", ("animal", "animals"), ("animal", "animals"), "animals", ("plant", "insect", "object")),
        ("bird-proposal-02", "bird", "primary", "anatomy", "Birds have feathers.", "What normally covers a bird's body? Answer:", "body-covering", "feathers", ("feathers", "feather"), ("feather", "feathers"), "feathers", ("scales", "fur", "shell")),
        ("bird-proposal-03", "bird", "primary", "anatomy", "Birds have wings.", "Which body parts allow most birds to fly? Answer:", "body-part", "wings", ("wings", "wing"), ("wing", "wings"), "wings", ("fins", "paws", "horns")),
        ("bird-proposal-04", "bird", "primary", "anatomy", "Birds have beaks.", "Which mouth part does a bird have? Answer:", "body-part", "beak", ("beak", "beaks"), ("beak", "beaks"), "beak", ("whiskers", "antlers", "gills")),
        ("bird-proposal-05", "bird", "primary", "locomotion", "Most birds can fly.", "Which action can most birds perform in the air? Answer:", "action", "fly", ("fly", "flies", "flew", "flying"), ("fly", "flies", "flew", "flying"), "flew", ("run", "swim", "drive")),
        ("bird-proposal-06", "bird", "primary", "vocalization", "Birds can sing.", "Which musical sound can a bird make? Answer:", "sound", "sing", ("sing", "sings", "sang", "singing"), ("sing", "sings", "sang", "singing"), "singing", ("bark", "roar", "buzz")),
        ("bird-proposal-07", "bird", "primary", "habitat", "Birds use nests as homes for their young.", "What kind of home does a bird use for its young? Answer:", "shelter", "nest", ("nest", "nests"), ("nest", "nests"), "nest", ("cave", "barn", "stable")),
        ("bird-proposal-08", "bird", "primary", "reproduction", "Birds lay eggs.", "What does a bird lay when reproducing? Answer:", "offspring-container", "eggs", ("eggs", "egg"), ("egg", "eggs", "lay", "lays", "laid"), "eggs", ("seeds", "bones", "rocks")),
        ("bird-proposal-09", "bird", "primary", "diet", "Birds can eat seeds.", "Which plant food do many birds eat? Answer:", "food", "seeds", ("seeds", "seed"), ("seed", "seeds"), "seeds", ("meat", "fish", "grass")),
        ("bird-proposal-10", "bird", "primary", "diet", "Birds can eat worms.", "Which small animal are birds traditionally shown pulling from soil? Answer:", "food", "worms", ("worms", "worm"), ("worm", "worms"), "worms", ("cats", "dogs", "bears")),
        ("bird-proposal-11", "bird", "primary", "appearance", "Many birds are small.", "What size are many birds? Answer:", "size", "small", ("small", "little", "tiny"), ("small", "little", "tiny"), "small", ("large", "tall", "huge")),
        ("bird-proposal-12", "bird", "primary", "reproduction", "Birds hatch from eggs.", "Which action describes a young bird emerging from an egg? Answer:", "action", "hatch", ("hatch", "hatches", "hatched", "hatching"), ("hatch", "hatches", "hatched", "hatching"), "hatched", ("bloom", "grow", "drive")),
        ("bird-proposal-13", "bird", "backup", "behavior", "Birds can be pets.", "A domesticated bird can be kept as what? Answer:", "role", "pet", ("pet", "pets"), ("pet", "pets"), "pet", ("tool", "plant", "vehicle")),
        ("bird-proposal-14", "bird", "backup", "habitat", "Birds can live in trees.", "Where can a bird live? Answer:", "habitat", "tree", ("tree", "trees"), ("tree", "trees"), "tree", ("ocean", "desert", "city")),
        ("bird-proposal-15", "bird", "backup", "habitat", "Birds can live in forests.", "Which habitat can contain birds? Answer:", "habitat", "forest", ("forest", "forests"), ("forest", "forests"), "forest", ("desert", "ocean", "city")),
        ("bird-proposal-16", "bird", "backup", "anatomy", "Birds have tails.", "Which rear body part does a bird have? Answer:", "body-part", "tail", ("tail", "tails"), ("tail", "tails"), "tail", ("trunk", "horns", "shell")),
        ("robot-proposal-01", "robot", "primary", "taxonomy", "Robots can be toys.", "A child's robot can be what kind of object? Answer:", "category", "toy", ("toy", "toys"), ("toy", "toys"), "toy", ("book", "chair", "spoon")),
        ("robot-proposal-02", "robot", "primary", "appearance", "Robots are commonly depicted as metal machines.", "Which material is commonly associated with robots? Answer:", "material", "metal", ("metal", "metallic"), ("metal", "metallic"), "metal", ("wood", "wool", "clay")),
        ("robot-proposal-03", "robot", "primary", "locomotion", "Robots can move.", "Which action can a robot perform? Answer:", "action", "move", ("move", "moves", "moved", "moving"), ("move", "moves", "moved", "moving"), "move", ("grow", "bloom", "hatch")),
        ("robot-proposal-04", "robot", "primary", "function", "Robots can talk.", "Which communication action can a robot perform? Answer:", "action", "talk", ("talk", "talks", "talked", "talking"), ("talk", "talks", "talked", "talking"), "talk", ("swim", "bloom", "hatch")),
        ("robot-proposal-05", "robot", "primary", "function", "Robots can help people.", "Which useful action can a robot perform for people? Answer:", "action", "help", ("help", "helps", "helped", "helping"), ("help", "helps", "helped", "helping"), "helped", ("hunt", "swim", "hatch")),
        ("robot-proposal-06", "robot", "primary", "behavior", "Robots can play.", "Which recreational action can a robot perform? Answer:", "action", "play", ("play", "plays", "played", "playing"), ("play", "plays", "played", "playing"), "play", ("swim", "bloom", "hatch")),
        ("robot-proposal-07", "robot", "primary", "function", "Robots can work.", "Which productive action can a robot perform? Answer:", "action", "work", ("work", "works", "worked", "working"), ("work", "works", "worked", "working"), "work", ("swim", "bloom", "hatch")),
        ("robot-proposal-08", "robot", "primary", "appearance", "Robots can be strong.", "Which physical trait can describe a robot? Answer:", "trait", "strong", ("strong",), ("strong",), "strong", ("weak", "tiny", "quiet")),
        ("robot-proposal-09", "robot", "primary", "appearance", "Robots can be shiny.", "How can a robot's surface look? Answer:", "appearance", "shiny", ("shiny",), ("shiny",), "shiny", ("dull", "rough", "dark")),
        ("robot-proposal-10", "robot", "primary", "function", "Robots can use batteries.", "Which object can store power for a robot? Answer:", "power-source", "battery", ("battery", "batteries"), ("battery", "batteries"), "battery", ("pillow", "basket", "blanket")),
        ("robot-proposal-11", "robot", "primary", "function", "Robots can be broken.", "Which condition can prevent a robot from working? Answer:", "condition", "broken", ("broken",), ("break", "breaks", "broke", "broken"), "broken", ("hungry", "thirsty", "jealous")),
        ("robot-proposal-12", "robot", "primary", "function", "Broken robots can be fixed.", "What can happen to a broken robot so it works again? Answer:", "state-change", "fixed", ("fixed", "repaired"), ("fix", "fixes", "fixed", "repair", "repairs", "repaired"), "fixed", ("eaten", "planted", "hatched")),
        ("robot-proposal-13", "robot", "backup", "appearance", "Robots can be big.", "What size can some robots be? Answer:", "size", "big", ("big", "large"), ("big", "large"), "big", ("tiny", "short", "weak")),
        ("robot-proposal-14", "robot", "backup", "appearance", "Robots can be small.", "What size can some robots be? Answer:", "size", "small", ("small", "little", "tiny"), ("small", "little", "tiny"), "small", ("large", "tall", "huge")),
        ("robot-proposal-15", "robot", "backup", "behavior", "Robots can be friendly.", "How can a helpful robot behave? Answer:", "trait", "friendly", ("friendly",), ("friendly",), "friendly", ("scary", "fierce", "angry")),
        ("robot-proposal-16", "robot", "backup", "function", "Robots can clean.", "Which household task can a robot perform? Answer:", "action", "clean", ("clean", "cleans", "cleaned", "cleaning"), ("clean", "cleans", "cleaned", "cleaning"), "clean", ("bake", "paint", "swim")),
        ("dragon-proposal-01", "dragon", "primary", "appearance", "Dragons are often big.", "What size are dragons often described as? Answer:", "size", "big", ("big", "large"), ("big", "large"), "big", ("small", "tiny", "short")),
        ("dragon-proposal-02", "dragon", "primary", "fantasy", "Dragons are strongly associated with fire.", "Which element is strongly associated with dragons? Answer:", "element", "fire", ("fire", "flames"), ("fire", "flame", "flames"), "fire", ("water", "ice", "mud")),
        ("dragon-proposal-03", "dragon", "primary", "habitat", "Dragons can live in caves.", "What kind of shelter can a dragon live in? Answer:", "shelter", "cave", ("cave", "caves"), ("cave", "caves"), "cave", ("stable", "nest", "ocean")),
        ("dragon-proposal-04", "dragon", "primary", "locomotion", "Dragons can fly.", "Which action can a winged dragon perform? Answer:", "action", "fly", ("fly", "flies", "flew", "flying"), ("fly", "flies", "flew", "flying"), "flew", ("run", "swim", "drive")),
        ("dragon-proposal-05", "dragon", "primary", "anatomy", "Dragons can have wings.", "Which body parts allow a dragon to fly? Answer:", "body-part", "wings", ("wings", "wing"), ("wing", "wings"), "wings", ("paws", "fins", "hands")),
        ("dragon-proposal-06", "dragon", "primary", "anatomy", "Dragons can have scales.", "What can cover a dragon's body? Answer:", "body-covering", "scales", ("scales", "scale"), ("scale", "scales"), "scales", ("fur", "feathers", "shell")),
        ("dragon-proposal-07", "dragon", "primary", "vocalization", "Dragons can roar.", "Which sound can a dragon make? Answer:", "sound", "roar", ("roar", "roars", "roared", "roaring"), ("roar", "roars", "roared", "roaring"), "roar", ("bark", "buzz", "neigh")),
        ("dragon-proposal-08", "dragon", "primary", "appearance", "Dragons are often strong.", "Which physical trait often describes a dragon? Answer:", "trait", "strong", ("strong",), ("strong",), "strong", ("weak", "tiny", "quiet")),
        ("dragon-proposal-09", "dragon", "primary", "fantasy", "Dragons are often scary.", "How are dangerous dragons often described? Answer:", "trait", "scary", ("scary", "frightening"), ("scary", "frightening"), "scary", ("friendly", "gentle", "calm")),
        ("dragon-proposal-10", "dragon", "primary", "fantasy", "Dragons are often fierce.", "Which trait often describes a dangerous dragon? Answer:", "trait", "fierce", ("fierce",), ("fierce",), "fierce", ("gentle", "calm", "tame")),
        ("dragon-proposal-11", "dragon", "primary", "fantasy", "Dragons are magical creatures.", "What kind of creature is a dragon in fantasy stories? Answer:", "trait", "magical", ("magical", "magic"), ("magic", "magical"), "magical", ("ordinary", "natural", "mechanical")),
        ("dragon-proposal-12", "dragon", "primary", "fantasy", "Dragons are associated with treasure.", "Which valuable hoard is associated with dragons? Answer:", "object", "treasure", ("treasure",), ("treasure",), "treasure", ("trash", "waste", "dirt")),
        ("dragon-proposal-13", "dragon", "backup", "behavior", "Dragons can be friendly.", "How can a kind dragon behave? Answer:", "trait", "friendly", ("friendly",), ("friendly",), "friendly", ("scary", "fierce", "angry")),
        ("dragon-proposal-14", "dragon", "backup", "anatomy", "Dragons can have tails.", "Which rear body part can a dragon have? Answer:", "body-part", "tail", ("tail", "tails"), ("tail", "tails"), "tail", ("trunk", "horns", "shell")),
        ("dragon-proposal-15", "dragon", "backup", "habitat", "Dragons can live in castles.", "Which building can house a storybook dragon? Answer:", "shelter", "castle", ("castle", "castles"), ("castle", "castles"), "castle", ("stable", "nest", "ocean")),
        ("dragon-proposal-16", "dragon", "backup", "reproduction", "Dragons can hatch from eggs.", "What can a baby dragon hatch from? Answer:", "offspring-container", "egg", ("egg", "eggs"), ("egg", "eggs", "hatch", "hatched"), "egg", ("seed", "nest", "cave")),
    )
)


MAIN_REVERSE_DISTRACTORS: tuple[tuple[str, tuple[str, str, str]], ...] = (
    ("cat", ("dog", "bird", "horse")),
    ("dog", ("cat", "bird", "horse")),
    ("bird", ("cat", "dog", "horse")),
    ("robot", ("cat", "dog", "bird")),
    ("dragon", ("cat", "dog", "bird")),
)


def build_main_review_shortlist(
    packet: SemanticReviewPacket,
    tokenizer: TextTokenizer,
) -> SemanticReviewShortlist:
    """Compile the supported main proposals into one compact decision surface."""
    if packet.concepts != MAIN_CONCEPTS:
        raise ValueError("main shortlist requires the exact main concept manifest")
    return build_review_shortlist(
        packet,
        MAIN_SHORTLIST_SPECS,
        MAIN_REVERSE_DISTRACTORS,
        tokenizer,
    )


def main_evidence_predicates() -> tuple[PredicateDefinition, ...]:
    """Return the broad exact-predicate audit used to choose main proposals."""
    if len({predicate for predicate, _ in MAIN_EVIDENCE_PREDICATE_CATEGORIES}) != len(
        MAIN_EVIDENCE_PREDICATE_CATEGORIES
    ):
        raise ValueError("main evidence predicates must be unique")
    return tuple(
        PredicateDefinition(predicate, category)
        for predicate, category in MAIN_EVIDENCE_PREDICATE_CATEGORIES
    )


__all__ = [
    "MAIN_EVIDENCE_PREDICATE_CATEGORIES",
    "MAIN_REVERSE_DISTRACTORS",
    "MAIN_SHORTLIST_SPECS",
    "build_main_review_shortlist",
    "main_evidence_predicates",
]
