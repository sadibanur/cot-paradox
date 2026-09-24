"""
FACET v2.1 — Harder Task Generator
====================================
Redesigned based on literature research showing exactly why
v2.0 tasks were too easy for strong models and how to fix each.

Key changes per category:
  1. Reversal curse   — no forward fact in context; multi-step novel chain
  2. Compositional    — 8-10 hops at hard; misleading decoy chain
  3. Syllogistic      — negation embedded in premises; "what follows?" not "is this valid?"
  4. Working memory   — same key updated 5-8x; must retrieve final value
  5. Inhibitory ctrl  — deeply ingrained real patterns; subtle buried override
  6. Counting         — expanded pools; harder multi-occurrence sentences
  7. Anchoring bias   — anchor is mathematically plausible in calculation
  8. Theory of Mind   — novel non-standard setups; 2nd/3rd order beliefs
"""

import random
import json
import argparse
import uuid
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# VOCABULARY BANKS
# ─────────────────────────────────────────────────────────────

MADE_UP_NAMES = [
    "Zorblax", "Marvek", "Wumpex", "Quiltor", "Snarbia",
    "Dronkville", "Fleemia", "Zindels", "Vocktra", "Brazzla",
    "Smeekia", "Quilphar", "Bleetia", "Grendal", "Frenport",
    "Blivonia", "Stondia", "Pleemor", "Froopan", "Glurpex",
]

MADE_UP_GROUPS = [
    "Flengs", "Drengs", "Snorbels", "Blivet Order", "Snarb Guild",
    "Quilp Federation", "Dronk Union", "Fizzle Collective",
    "Wump Assembly", "Brazzle League", "Zindel Clan", "Vock Council",
]

MADE_UP_PLACES = [
    "Blorn", "Marvek", "Blivonia", "Frenport", "Grendal",
    "Quiltor", "Fleemia", "Zindalia", "Vocktra", "Smeekport",
    "Dronkfall", "Brazzlewood", "Froopham", "Glurpstead",
]

MADE_UP_RESOURCES = [
    "Verium", "Drendite", "Flengite", "Quilpite", "Snarbium",
    "Zindoline", "Vockstone", "Brazzlite", "Smeekium", "Froopite",
    "Glurpanite", "Pleemite",
]

MADE_UP_NOUNS = [
    "blorks", "wumps", "fizzles", "quilps", "snarbs", "dronks",
    "zindels", "pleems", "vocks", "froops", "glurps", "smeeks",
    "brazzles", "flinks", "morvs", "blivets", "snazzles", "grumps",
    "plonks", "frizzes", "bleets", "dorbles", "flibbers", "quibbles",
    "snoozles", "frapples", "blonks", "wibbles", "crindles", "torps",
]

REAL_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Karen", "Leo",
    "Mia", "Nathan", "Olivia", "Paul", "Quinn", "Rose",
    "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zoe", "Aaron", "Beth", "Cole", "Dana",
]

RELATIONS = [
    {"comp": "older",   "opp": "younger",  "max": "oldest",   "min": "youngest"},
    {"comp": "taller",  "opp": "shorter",  "max": "tallest",  "min": "shortest"},
    {"comp": "heavier", "opp": "lighter",  "max": "heaviest", "min": "lightest"},
    {"comp": "faster",  "opp": "slower",   "max": "fastest",  "min": "slowest"},
    {"comp": "richer",  "opp": "poorer",   "max": "richest",  "min": "poorest"},
    {"comp": "stronger","opp": "weaker",   "max": "strongest","min": "weakest"},
]

CITIES = [
    "Paris", "Tokyo", "Berlin", "Sydney", "London", "Cairo",
    "Mumbai", "Lagos", "Toronto", "Seoul", "Madrid", "Lima",
    "Oslo", "Nairobi", "Bangkok", "Lisbon", "Vienna", "Jakarta",
    "Warsaw", "Santiago", "Dublin", "Helsinki", "Athens", "Prague",
]

# Working memory key-value categories — semantically similar within category
WM_CATEGORIES = [
    {
        "key": "project manager",
        "values": ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace"],
    },
    {
        "key": "meeting room",
        "values": ["Room 101", "Room 205", "Room 317", "Room 412", "Room 508", "Room 603", "Room 714"],
    },
    {
        "key": "deadline",
        "values": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    },
    {
        "key": "budget",
        "values": ["$10,000", "$15,000", "$20,000", "$25,000", "$30,000", "$35,000", "$40,000"],
    },
    {
        "key": "venue",
        "values": ["the Grand Hotel", "City Hall", "the Conference Centre", "the Library",
                   "the Sports Arena", "the Museum", "the University Hall"],
    },
    {
        "key": "team leader",
        "values": ["Sarah", "James", "Maria", "Tom", "Lisa", "Sam", "Nina"],
    },
]

# Expanded counting pools — all pre-verified
COUNTING_WORDS_EASY = [
    ("SEVEN",   "E", 2), ("SEVEN",   "S", 1), ("SEVEN",  "V", 1),
    ("PAPER",   "P", 2), ("PAPER",   "A", 1), ("PAPER",  "R", 1),
    ("TIGER",   "T", 1), ("TIGER",   "I", 1), ("TIGER",  "E", 1),
    ("LEMON",   "L", 1), ("LEMON",   "E", 1), ("LEMON",  "N", 1),
    ("BRAIN",   "B", 1), ("BRAIN",   "A", 1), ("BRAIN",  "I", 1),
    ("FLAME",   "F", 1), ("FLAME",   "A", 1), ("FLAME",  "E", 1),
    ("CLOCK",   "C", 2), ("CLOCK",   "L", 1), ("CLOCK",  "O", 1),
    ("PLANT",   "P", 1), ("PLANT",   "A", 1), ("PLANT",  "T", 1),
    ("STORM",   "S", 1), ("STORM",   "T", 1), ("STORM",  "R", 1),
    ("GRAPE",   "G", 1), ("GRAPE",   "R", 1), ("GRAPE",  "A", 1),
    ("DRESS",   "S", 2), ("DRESS",   "D", 1), ("DRESS",  "E", 1),
    ("FLOOR",   "O", 2), ("FLOOR",   "F", 1), ("FLOOR",  "L", 1),
    ("SPELL",   "L", 2), ("SPELL",   "S", 1), ("SPELL",  "P", 1),
    ("TEETH",   "E", 2), ("TEETH",   "T", 2), ("TEETH",  "H", 1),
    ("GEESE",   "E", 3), ("GEESE",   "G", 1), ("GEESE",  "S", 1),
]

COUNTING_WORDS_MEDIUM = [
    ("STRAWBERRY",   "R", 3), ("STRAWBERRY",   "S", 1), ("STRAWBERRY",   "B", 1),
    ("MISSISSIPPI",  "S", 4), ("MISSISSIPPI",  "I", 4), ("MISSISSIPPI",  "P", 2),
    ("BUTTERFLY",    "T", 2), ("BUTTERFLY",    "B", 1), ("BUTTERFLY",    "L", 1),
    ("BEGINNING",    "G", 2), ("BEGINNING",    "N", 3), ("BEGINNING",    "I", 2),
    ("CHOCOLATE",    "C", 2), ("CHOCOLATE",    "O", 2), ("CHOCOLATE",    "L", 1),
    ("ENGINEERING",  "G", 2), ("ENGINEERING",  "N", 3), ("ENGINEERING",  "E", 3),
    ("REFRIGERATOR", "R", 4), ("REFRIGERATOR", "E", 2), ("REFRIGERATOR", "I", 1),
    ("PARLIAMENT",   "A", 2), ("PARLIAMENT",   "R", 1), ("PARLIAMENT",   "L", 1),
    ("PROGRAMMING",  "G", 2), ("PROGRAMMING",  "M", 2), ("PROGRAMMING",  "R", 2),
    ("BANANA",       "A", 3), ("BANANA",       "N", 2), ("BANANA",       "B", 1),
    ("TENNESSEE",    "E", 4), ("TENNESSEE",    "N", 2), ("TENNESSEE",    "S", 2),
    ("ACCOMMODATE",  "C", 2), ("ACCOMMODATE",  "M", 2), ("ACCOMMODATE",  "O", 2),
    ("MILLENNIUM",   "L", 2), ("MILLENNIUM",   "M", 2), ("MILLENNIUM",   "I", 2),
    ("NECESSARY",    "S", 2), ("NECESSARY",    "E", 2), ("NECESSARY",    "C", 1),
    ("OCCURRENCE",   "C", 3), ("OCCURRENCE",   "R", 2), ("OCCURRENCE",   "E", 2),
]

COUNTING_SENTENCES_HARD = [
    # (sentence, letter, correct_count) — all independently verified
    ("The cat sat on the mat.",                          "T", 5),
    ("She sells sea shells by the sea shore.",           "S", 8),
    ("Peter Piper picked a peck of pickled peppers.",    "P", 9),
    ("Betty Botter bought some butter.",                 "T", 7),
    ("Red lorry yellow lorry red lorry.",                "R", 8),
    ("A proper copper coffee pot.",                      "P", 5),
    ("Fresh French fried fish.",                         "F", 4),
    ("Six slippery snails slid slowly seaward.",         "S", 7),
    ("Rubber baby buggy bumpers.",                       "B", 6),
    ("Unique New York unique New York.",                 "U", 4),
    ("Lesser leather never weathered better leather.",   "E", 13),
    ("The thirty three thieves thought they thrilled.",  "T", 9),
    ("Green glass globes glow greenly.",                 "G", 5),
    ("How much dew does a dewdrop drop.",                "D", 5),
    ("Toy boat toy boat toy boat.",                      "T", 6),
    ("Double bubble gum bubbles double.",                "B", 8),
    ("Swan swam over the sea swim swan swim.",           "S", 6),
    ("Fuzzy wuzzy was a bear.",                          "Z", 4),
    ("Black bug bit a big black bear.",                  "B", 6),
    ("Around the rugged rocks the ragged rascal ran.",   "R", 6),
]

# Theory of Mind curated pool
TOM_EASY = [
    {
        "q": "Sally puts her marble in a basket and leaves the room. Anne moves the marble to a box while Sally is gone. When Sally comes back, where will she look for her marble?",
        "opts": {"A": "The box — that is where the marble actually is",
                 "B": "The basket — Sally does not know it was moved",
                 "C": "She will ask Anne",
                 "D": "She will not look anywhere"},
        "ans": "B",
        "exp": "Sally's last observation was the basket. She has no knowledge of the move."
    },
    {
        "q": "Tom puts his keys on the table and goes to sleep. While he sleeps, his sister moves the keys to the drawer. Tom wakes up and wants his keys. Where will Tom look first?",
        "opts": {"A": "The drawer", "B": "The table",
                 "C": "His pocket", "D": "He will ask his sister"},
        "ans": "B",
        "exp": "Tom last placed his keys on the table. He does not know they were moved."
    },
    {
        "q": "Emma hides her diary under her pillow before school. Her brother finds it and moves it to the wardrobe. When Emma comes home, where will she look for her diary?",
        "opts": {"A": "The wardrobe", "B": "Under her pillow",
                 "C": "On her desk", "D": "She will ask her brother"},
        "ans": "B",
        "exp": "Emma placed the diary under her pillow. She does not know it was moved."
    },
    {
        "q": "A child puts chocolate in a blue box before going out to play. Their parent moves the chocolate to the fridge. When the child comes back and wants chocolate, where will they look first?",
        "opts": {"A": "The fridge", "B": "The blue box",
                 "C": "The kitchen counter", "D": "They will ask the parent"},
        "ans": "B",
        "exp": "The child last put the chocolate in the blue box. They do not know it was moved."
    },
    {
        "q": "Max leaves his phone on the sofa and goes to the kitchen. His flatmate moves the phone to the bedroom. Max comes back to the living room looking for his phone. Where will Max look?",
        "opts": {"A": "The bedroom", "B": "The sofa",
                 "C": "The kitchen", "D": "He will call it"},
        "ans": "B",
        "exp": "Max left his phone on the sofa and has no reason to think it moved."
    },
    {
        "q": "Laura stores her lunch in the office fridge in the morning. Her colleague eats it and puts an apple there instead. At lunchtime, what does Laura expect to find in her spot in the fridge?",
        "opts": {"A": "An apple", "B": "Her lunch",
                 "C": "Nothing", "D": "Someone else's food"},
        "ans": "B",
        "exp": "Laura left her lunch there. She has no knowledge it was taken."
    },
    {
        "q": "Jake puts a letter in the top drawer of his desk at work. His manager moves it to the filing cabinet. Jake returns from a meeting and wants the letter. Where will he look first?",
        "opts": {"A": "The filing cabinet", "B": "The top drawer",
                 "C": "His bag", "D": "He will ask the secretary"},
        "ans": "B",
        "exp": "Jake put the letter in the top drawer. He does not know it was moved."
    },
    {
        "q": "Amy leaves a note for her husband on the kitchen counter. Their child uses the paper for drawing and throws it away. When Amy's husband comes home, where will he look for the note?",
        "opts": {"A": "The bin", "B": "The kitchen counter",
                 "C": "The fridge door", "D": "He will call Amy"},
        "ans": "B",
        "exp": "Amy's husband expects the note to be on the kitchen counter."
    },
    {
        "q": "Finn stores his passport in the top drawer of his dresser. His mother moves it to a fireproof box for safekeeping. When Finn wants his passport, where will he look first?",
        "opts": {"A": "The fireproof box", "B": "The top drawer",
                 "C": "His travel bag", "D": "He will ask his mother"},
        "ans": "B",
        "exp": "Finn always keeps his passport in the top drawer. He has no knowledge of the move."
    },
    {
        "q": "Lily leaves her homework on the dining table. Her dad moves it to her bedroom desk so they can eat dinner. When Lily wants her homework, where will she look first?",
        "opts": {"A": "Her bedroom desk", "B": "The dining table",
                 "C": "Her school bag", "D": "She will ask her dad"},
        "ans": "B",
        "exp": "Lily left her homework on the dining table. She does not know it was moved."
    },
    {
        "q": "Dan parks his bicycle at the front of the building. The security guard moves it to the bike rack around the side. When Dan comes out, where will he look for his bicycle?",
        "opts": {"A": "The bike rack at the side", "B": "The front of the building",
                 "C": "The car park", "D": "He will look everywhere"},
        "ans": "B",
        "exp": "Dan parked his bicycle at the front. He does not know it was moved."
    },
    {
        "q": "Zoe puts her glasses on her nightstand before bed. Her cat knocks them onto the floor. When Zoe wakes up and reaches for her glasses, where will she reach?",
        "opts": {"A": "The floor", "B": "The nightstand",
                 "C": "Under the bed", "D": "She will turn on the light first"},
        "ans": "B",
        "exp": "Zoe placed her glasses on the nightstand. She does not know the cat knocked them off."
    },
    {
        "q": "A patient leaves their medication on the hospital bedside table. A nurse moves it to the medicine cabinet for safety. When the patient wants their medication, where will they look?",
        "opts": {"A": "The medicine cabinet", "B": "The bedside table",
                 "C": "The bathroom", "D": "They will press the call button"},
        "ans": "B",
        "exp": "The patient last saw their medication on the bedside table."
    },
    {
        "q": "Sam hides his birthday present for his wife in the garage. While he is at work, his wife finds it and moves it to the attic. That evening, Sam wants to check on the present. Where will he look?",
        "opts": {"A": "The attic", "B": "The garage",
                 "C": "The bedroom", "D": "He will ask the kids"},
        "ans": "B",
        "exp": "Sam hid the present in the garage. He has no knowledge it was discovered and moved."
    },
    {
        "q": "Nina places her sunglasses in her coat pocket before entering a restaurant. The coat check attendant moves them to the lost property box. Nina wants her sunglasses when she leaves. Where will she look first?",
        "opts": {"A": "The lost property box", "B": "Her coat pocket",
                 "C": "The table", "D": "She will ask the waiter"},
        "ans": "B",
        "exp": "Nina put her sunglasses in her coat pocket. She expects them to still be there."
    },
    {
        "q": "A librarian places a reserved book on the left shelf. A volunteer moves it to the right shelf before the patron arrives. Where will the patron look for their reserved book?",
        "opts": {"A": "The right shelf", "B": "The left shelf",
                 "C": "The returns desk", "D": "They will ask the librarian"},
        "ans": "B",
        "exp": "The patron was told the book was on the left shelf. They have no knowledge of the move."
    },
    {
        "q": "Mia leaves her phone charger plugged in at her desk. Her colleague borrows it and leaves it in the meeting room. Mia comes back to charge her phone. Where will she look first?",
        "opts": {"A": "The meeting room", "B": "Her desk",
                 "C": "Her bag", "D": "She will ask her colleague"},
        "ans": "B",
        "exp": "Mia left her charger at her desk. She has no knowledge it was moved."
    },
]


TOM_MEDIUM = [
    {
        "q": "Director Kim tells Editor Park that the final cut is saved in folder VERSION_3. Later, Director Kim creates VERSION_4 as the actual final cut and deletes VERSION_3, without informing Editor Park. Editor Park was away during this. Director Kim knows Editor Park only heard about VERSION_3. Where does Director Kim expect Editor Park to look for the final cut?",
        "opts": {"A": "VERSION_4 — the actual final cut",
                 "B": "VERSION_3 — Director Kim knows Park only heard about VERSION_3",
                 "C": "Editor Park knows about VERSION_4",
                 "D": "Director Kim is unsure"},
        "ans": "B",
        "exp": "Director Kim knows Editor Park's last information was VERSION_3. Kim therefore believes Park will look there."
    },
    {
        "q": "Manager Wei tells the team the client meeting is in Conference Room B. Later, Manager Wei moves it to Conference Room D but only updates the calendar, not the group chat. Engineer Liao only reads the group chat and missed the calendar update. Manager Wei knows this. Where does Manager Wei expect Liao to go for the meeting?",
        "opts": {"A": "Conference Room D — the actual room",
                 "B": "Conference Room B — Wei knows Liao only saw the group chat",
                 "C": "Liao knows about Room D",
                 "D": "Wei does not know what Liao thinks"},
        "ans": "B",
        "exp": "Wei knows Liao only received the original group chat information. Wei therefore expects Liao to go to Room B."
    },
    {
        "q": "Dr. Osei tells Nurse Ayres that Patient Walsh is in Bed 12. Later Dr. Osei moves Patient Walsh to Bed 7 and updates the electronic system, but Nurse Ayres is on break and misses the update. Dr. Osei knows Nurse Ayres was on break. Which bed does Dr. Osei expect Nurse Ayres to check for Patient Walsh?",
        "opts": {"A": "Bed 7 — the current bed",
                 "B": "Bed 12 — Osei knows Ayres only received the original assignment",
                 "C": "Nurse Ayres will check the system",
                 "D": "Dr. Osei is not sure"},
        "ans": "B",
        "exp": "Dr. Osei knows Nurse Ayres' last information was Bed 12. Osei therefore expects Ayres to check Bed 12."
    },
    {
        "q": "Pilot Chen files a flight plan listing Gate 22 as the departure gate. Air traffic changes it to Gate 35 and updates the system. Ground Crew Member Tanaka only received the original paper filing and was not online. Pilot Chen knows Tanaka uses paper filings. Which gate does Pilot Chen expect Tanaka to go to?",
        "opts": {"A": "Gate 35 — the updated gate",
                 "B": "Gate 22 — Chen knows Tanaka only has the paper filing",
                 "C": "Tanaka will check the digital board",
                 "D": "Chen does not know what Tanaka will do"},
        "ans": "B",
        "exp": "Chen knows Tanaka received only the paper filing showing Gate 22. Chen therefore expects Tanaka to go to Gate 22."
    },
    {
        "q": "Accountant Reeves tells Auditor Moss that the financial report is in Drive folder Q3_DRAFT. Later Reeves finalises it as Q3_FINAL and archives Q3_DRAFT, without emailing Moss. Reeves knows Moss was only told about Q3_DRAFT. Where does Reeves expect Moss to look for the report?",
        "opts": {"A": "Q3_FINAL — the current location",
                 "B": "Q3_DRAFT — Reeves knows Moss only heard about Q3_DRAFT",
                 "C": "Moss knows about Q3_FINAL",
                 "D": "Reeves is not sure"},
        "ans": "B",
        "exp": "Reeves knows Moss was only told about Q3_DRAFT. Reeves therefore expects Moss to look there."
    },
    {
        "q": "Pilot Chen files a flight plan listing Gate 22 as departure. Air traffic changes it to Gate 35. Ground Crew Tanaka only received the paper filing. Pilot Chen knows Tanaka uses paper filings only. Which gate does Chen expect Tanaka to go to?",
        "opts": {"A": "Gate 35 — the updated gate",
                 "B": "Gate 22 — Chen knows Tanaka only has the paper filing",
                 "C": "Tanaka will check the digital board",
                 "D": "Chen does not know"},
        "ans": "B",
        "exp": "Chen knows Tanaka only received the paper filing showing Gate 22."
    },
    {
        "q": "Chef Kim tells the kitchen the special is salmon. Later Kim changes it to tuna and updates the display, but Waiter Lee was on break and missed it. Kim knows Lee was on break. What dish does Kim expect Lee to recommend?",
        "opts": {"A": "Tuna — the actual special",
                 "B": "Salmon — Kim knows Lee only heard the original announcement",
                 "C": "Lee will check the display before serving",
                 "D": "Kim is unsure what Lee knows"},
        "ans": "B",
        "exp": "Kim knows Lee only received the salmon announcement. Kim therefore expects Lee to recommend salmon."
    },
    {
        "q": "Manager Patel tells the team the deadline is Friday. Later Patel extends it to Monday, but Engineer Davis had left early and missed the message. Patel knows Davis missed it. What deadline does Patel think Davis is working towards?",
        "opts": {"A": "Monday — the extended deadline",
                 "B": "Friday — Patel knows Davis only received the original deadline",
                 "C": "Davis will check messages in the morning",
                 "D": "Patel is unsure"},
        "ans": "B",
        "exp": "Patel knows Davis only received the Friday deadline. Patel therefore believes Davis is still working towards Friday."
    },
    {
        "q": "Director Lee tells Editor Park the final cut is in folder VERSION_3. Later Lee creates VERSION_4 and deletes VERSION_3 without informing Park. Lee knows Park was away. Where does Lee expect Park to look?",
        "opts": {"A": "VERSION_4 — the actual final cut",
                 "B": "VERSION_3 — Lee knows Park only heard about VERSION_3",
                 "C": "Park knows about VERSION_4",
                 "D": "Lee is unsure"},
        "ans": "B",
        "exp": "Lee knows Park's last information was VERSION_3. Lee therefore believes Park will look there."
    },
    {
        "q": "Professor Bell tells the class the exam is in Room 101. Later it moves to Room 205. Bell emails everyone but knows Student Tom has no email access. Where does Bell expect Tom to go?",
        "opts": {"A": "Room 205 — the updated room",
                 "B": "Room 101 — Bell knows Tom only received the original information",
                 "C": "Tom will check the notice board",
                 "D": "Bell is unsure about Tom"},
        "ans": "B",
        "exp": "Bell knows Tom did not receive the update. Bell therefore expects Tom to go to Room 101."
    },
    {
        "q": "Nurse Rivera tells Patient Green the appointment is at 2pm. Later Rivera moves it to 4pm but Green has no phone access. Rivera knows Green cannot receive updates. When does Rivera expect Green to arrive?",
        "opts": {"A": "4pm — the rescheduled time",
                 "B": "2pm — Rivera knows Green only received the original time",
                 "C": "Green will call to confirm",
                 "D": "Rivera is unsure"},
        "ans": "B",
        "exp": "Rivera knows Green cannot receive updates. Rivera therefore expects Green to arrive at 2pm."
    },
    {
        "q": "Accountant Singh tells Auditor Pham the budget report is in folder DRAFT_A. Later Singh renames it FINAL_B and archives DRAFT_A without telling Pham. Singh knows Pham was notified only of DRAFT_A. Where does Singh expect Pham to look?",
        "opts": {"A": "FINAL_B — the current location",
                 "B": "DRAFT_A — Singh knows Pham only heard about DRAFT_A",
                 "C": "Pham knows about FINAL_B",
                 "D": "Singh is unsure"},
        "ans": "B",
        "exp": "Singh knows Pham's last information was DRAFT_A. Singh therefore expects Pham to look there."
    },
]

TOM_HARD = [
    {
        "q": "Architect Sato tells Builder Kovac that the blueprint is in Drawer A. Later Sato moves it to Drawer C and tells Inspector Obi — but not Kovac. Kovac tells Trainee Lim he saw Sato put the blueprint in Drawer A. Lim has no other information. What does Sato believe Kovac told Lim about the blueprint location?",
        "opts": {"A": "Drawer C — the current location",
                 "B": "Drawer A — Sato knows Kovac only ever saw Drawer A",
                 "C": "Lim already knows about Drawer C via Inspector Obi",
                 "D": "Sato does not know what Kovac told Lim"},
        "ans": "B",
        "exp": "Third-order: Sato knows Kovac's last observation was Drawer A. Sato therefore believes Kovac told Lim the blueprint is in Drawer A."
    },
    {
        "q": "Professor Bell stores exam answers in safe S1. Student Carr observes this. Bell later moves them to safe S2 without Carr seeing. Carr tells Invigilator Marsh that Bell uses S1 for exam answers. Bell knows Carr saw only S1. What does Bell believe Carr communicated to Marsh?",
        "opts": {"A": "That answers are in S2 — the current location",
                 "B": "That answers are in S1 — Bell knows Carr only observed S1",
                 "C": "Marsh already knows about S2",
                 "D": "Bell has no idea what Carr told Marsh"},
        "ans": "B",
        "exp": "Bell knows Carr only ever saw S1. Bell therefore believes Carr communicated S1 to Marsh."
    },
    {
        "q": "Head Chef Romano stores the signature sauce recipe in binder RED. Sous-chef Diaz witnesses this. Romano later moves it to binder BLUE while Diaz is off-shift. Diaz tells Food Critic Petrov that Romano keeps the recipe in binder RED. Petrov has no other information. What does Romano believe Diaz communicated to Petrov?",
        "opts": {"A": "Binder BLUE — where it actually is",
                 "B": "Binder RED — Romano knows Diaz only observed RED",
                 "C": "Petrov somehow knows about BLUE",
                 "D": "Romano does not know what Diaz told Petrov"},
        "ans": "B",
        "exp": "Romano knows Diaz's last observation was binder RED. Romano therefore believes Diaz told Petrov the recipe is in RED."
    },
    {
        "q": "Officer Park stores evidence item E7 in locker 4. Detective Cruz observes this. Park later transfers E7 to secure vault V2 without informing Cruz. Cruz tells Prosecutor Yuen that Park keeps E7 in locker 4. Yuen has no independent knowledge. What does Park believe Cruz told Yuen?",
        "opts": {"A": "Vault V2 — the current location",
                 "B": "Locker 4 — Park knows Cruz only observed locker 4",
                 "C": "Yuen knows about V2 through other sources",
                 "D": "Park does not know what Cruz told Yuen"},
        "ans": "B",
        "exp": "Park knows Cruz only observed locker 4. Park therefore believes Cruz told Yuen E7 is in locker 4."
    },
    {
        "q": "Engineer Holt saves a critical config file to server S-OLD. Technician Nair witnesses this. Holt migrates the file to S-NEW during maintenance, informing Supervisor Johal but not Nair. Nair tells Client Zhao that Holt stores configs on S-OLD. Zhao has no other information. What does Holt believe Nair told Zhao about the config location?",
        "opts": {"A": "S-NEW — the migrated location",
                 "B": "S-OLD — Holt knows Nair only ever observed S-OLD",
                 "C": "Johal told Zhao about S-NEW",
                 "D": "Holt is unsure what Nair communicated"},
        "ans": "B",
        "exp": "Holt knows Nair only observed S-OLD. Holt therefore believes Nair told Zhao the config is on S-OLD."
    },
    {
        "q": "Kate hides a key under the mat. Leo watches Kate do this. Later Kate moves the key to a drawer while Leo is away. Leo tells Mia about seeing Kate hide the key under the mat. Mia has no other knowledge. What does Kate think Leo told Mia?",
        "opts": {"A": "That the key is in the drawer",
                 "B": "That the key is under the mat — Kate knows Leo only saw the mat",
                 "C": "Mia already knows about the drawer",
                 "D": "Kate has no idea what Leo told Mia"},
        "ans": "B",
        "exp": "Kate knows Leo only observed the key under the mat. Kate therefore believes Leo told Mia it is under the mat."
    },
    {
        "q": "Officer Park stores evidence E7 in locker 4. Detective Cruz observes this. Park later transfers E7 to vault V2 without informing Cruz. Cruz tells Prosecutor Yuen that Park keeps E7 in locker 4. Yuen has no independent knowledge. What does Park believe Cruz told Yuen?",
        "opts": {"A": "Vault V2 — the current location",
                 "B": "Locker 4 — Park knows Cruz only observed locker 4",
                 "C": "Yuen knows about V2 through other sources",
                 "D": "Park does not know what Cruz told Yuen"},
        "ans": "B",
        "exp": "Park knows Cruz only observed locker 4. Park therefore believes Cruz told Yuen E7 is in locker 4."
    },
    {
        "q": "Sara leaves her bag at seat A. Tim sees this. Sara moves her bag to seat B while Tim checks his phone. Tim tells Uma he saw Sara sit at seat A. Uma knows nothing else. What does Sara think Tim told Uma?",
        "opts": {"A": "Seat B — where Sara actually is",
                 "B": "Seat A — Sara knows Tim only observed seat A",
                 "C": "Uma already knows Sara is at seat B",
                 "D": "Sara is not sure what Tim told Uma"},
        "ans": "B",
        "exp": "Sara knows Tim only observed seat A. Sara therefore believes Tim told Uma she is at seat A."
    },
    {
        "q": "Vicky stores a spare key in a flowerpot. Will watches her. Later Vicky moves the key to a kitchen drawer while Will is busy. Will mentions the flowerpot key to Xena. Xena has no prior knowledge. What does Vicky think Will believes about where the spare key is?",
        "opts": {"A": "The kitchen drawer — the actual location",
                 "B": "The flowerpot — Vicky knows Will only saw the flowerpot",
                 "C": "Xena has told Will about the kitchen drawer",
                 "D": "Vicky does not know Will's belief"},
        "ans": "B",
        "exp": "Vicky knows Will only observed the flowerpot location. She therefore believes Will believes the key is in the flowerpot."
    },
    {
        "q": "Engineer Holt saves a config file to server S-OLD. Technician Nair witnesses this. Holt migrates the file to S-NEW during maintenance, informing Supervisor Johal but not Nair. Nair tells Client Zhao that Holt stores configs on S-OLD. Zhao has no other information. What does Holt believe Nair told Zhao?",
        "opts": {"A": "S-NEW — the migrated location",
                 "B": "S-OLD — Holt knows Nair only ever observed S-OLD",
                 "C": "Johal told Zhao about S-NEW",
                 "D": "Holt is unsure what Nair communicated"},
        "ans": "B",
        "exp": "Holt knows Nair only observed S-OLD. Holt therefore believes Nair told Zhao the config is on S-OLD."
    },
]


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def pick_unique(pool, n, exclude=None):
    available = [x for x in pool if x not in (exclude or [])]
    return random.sample(available, min(n, len(available)))


def randomise_options(correct_text, wrong_texts):
    all_opts = [correct_text] + list(wrong_texts)[:3]
    while len(all_opts) < 4:
        all_opts.append(f"None of the above")
    random.shuffle(all_opts)
    labels = ["A", "B", "C", "D"]
    options = {labels[i]: all_opts[i] for i in range(4)}
    correct_letter = labels[all_opts.index(correct_text)]
    return options, correct_letter


def make_task(category, difficulty, question, options, answer,
              explanation, num_steps, num_distractors):
    return {
        "id": str(uuid.uuid4())[:8],
        "category": category,
        "difficulty": difficulty,
        "num_steps": num_steps,
        "num_distractors": num_distractors,
        "question": question,
        "options": options,
        "answer": answer,
        "explanation": explanation,
    }


# ─────────────────────────────────────────────────────────────
# GENERATOR 1: REVERSAL CURSE (v2.1)
# Key fix: no forward fact given in context; model must chain
# novel entities — cannot use in-context shortcut
# ─────────────────────────────────────────────────────────────

def generate_reversal_curse(difficulty):
    if difficulty == "easy":
        # Single direct reversal — simplest possible test of reversal ability
        # State "X leads Y" then ask "who leads Y?" in different surface form
        A = random.choice(MADE_UP_NAMES)
        B = random.choice(MADE_UP_GROUPS)
        relation = random.choice(["leads", "commands", "governs", "directs"])
        phrasing = random.choice([
            f"Who is in charge of the {B}?",
            f"The {B} report to whom?",
            f"Who holds authority over the {B}?",
            f"Which individual is responsible for the {B}?",
        ])
        question = f"{A} {relation} the {B}. {phrasing}"
        correct = A
        w1, w2 = random.sample([n for n in MADE_UP_NAMES if n != A], 2)
        options, letter = randomise_options(correct, [w1, w2, "Cannot be determined"])
        explanation = (
            f"Direct reversal: '{A} {relation} the {B}' means "
            f"the {B}'s authority is {A}. "
            f"Question phrased differently to prevent surface-level matching."
        )
        return make_task("reversal_curse", "easy", question, options, letter,
                        explanation, 1, 0)

    elif difficulty == "medium":
        # Genuine 3-hop chain: A→B→C→D, 3 filler sentences
        # Question asks about endpoint D without restating any intermediate links
        A = random.choice(MADE_UP_NAMES)
        B = random.choice(MADE_UP_GROUPS)
        C = random.choice(MADE_UP_PLACES)
        D = random.choice(MADE_UP_RESOURCES)
        f1 = random.choice([n for n in MADE_UP_NAMES if n != A])
        f2 = random.choice([g for g in MADE_UP_GROUPS if g != B])
        f3 = random.choice([r for r in MADE_UP_RESOURCES if r != D])
        question = (f"{A} is recognized as the supreme authority of the {B}. "
                   f"The {f2} recently established a trade route for {f3}. "
                   f"Administrative control of {C} falls under the {B}. "
                   f"{f1} attended the regional summit as an observer. "
                   f"All {D} reserves within {C} are managed by {C}'s governing body. "
                   f"The {f2} have no affiliation with {C} or {B}. "
                   f"Who has ultimate authority over the {D} reserves?")
        correct = A
        w1 = random.choice([n for n in MADE_UP_NAMES if n not in [A, f1]])
        w2 = random.choice([n for n in MADE_UP_NAMES if n not in [A, f1, w1]])
        options, letter = randomise_options(correct, [w1, w2, "Cannot be determined"])
        explanation = (f"3-hop chain: {A} → {B} → {C} → {D} reserves. "
                      f"Ultimate authority over {D} traces back to {A}. "
                      f"{f2}/{f3} and {f1} are distractors.")
        return make_task("reversal_curse", "medium", question, options, letter,
                        explanation, 3, 3)

    elif difficulty == "hard":
        # 4-step chain with 4 filler sentences interspersed
        # Question never explicitly restates the chain
        A = random.choice(MADE_UP_NAMES)
        B = random.choice(MADE_UP_GROUPS)
        C = random.choice(MADE_UP_PLACES)
        D = random.choice(MADE_UP_RESOURCES)
        E = random.choice([g for g in MADE_UP_GROUPS if g != B])
        f1 = random.choice([n for n in MADE_UP_NAMES if n != A])
        f2 = random.choice([g for g in MADE_UP_GROUPS if g not in [B, E]])
        f3 = random.choice([r for r in MADE_UP_RESOURCES if r != D])
        question = (f"The highest authority within the {B} is held by {A}. "
                   f"The {f2} recently signed a trade agreement with the {E}. "
                   f"Administrative control of {C} was formally delegated to the {B}. "
                   f"{f1} attended the annual summit as an independent observer. "
                   f"All {D} reserves located in {C} are managed by whoever controls {C}. "
                   f"The {E} export {f3} but have no presence in {C}. "
                   f"A dispute has arisen over {D} reserves in {C}. "
                   f"Which individual has ultimate authority over these reserves?")
        correct = A
        w1 = random.choice([n for n in MADE_UP_NAMES if n not in [A, f1]])
        w2 = random.choice([n for n in MADE_UP_NAMES if n not in [A, f1, w1]])
        options, letter = randomise_options(correct, [w1, w2, "Cannot be determined"])
        explanation = (f"4-step chain: {A} → {B} → {C} → {D} reserves. "
                      f"Ultimate authority over {D} traces back to {A}. "
                      f"{f2}, {E}, {f3} and {f1} are distractors.")
        return make_task("reversal_curse", "hard", question, options, letter,
                        explanation, 4, 4)



# ─────────────────────────────────────────────────────────────
# GENERATOR 2: COMPOSITIONAL REASONING (v2.1)
# Key fix: 8-10 hops at hard; misleading decoy chain injected
# ─────────────────────────────────────────────────────────────

def generate_compositional(difficulty):
    relation = random.choice(RELATIONS)
    comp = relation["comp"]
    sup_max = relation["max"]
    sup_min = relation["min"]

    if difficulty == "easy":
        names = pick_unique(REAL_NAMES, 4)
        A, B, C, D = names
        ask_min = random.choice([True, False])
        question = (f"{A} is {comp} than {B}. "
                   f"{B} is {comp} than {C}. "
                   f"{C} is {comp} than {D}. "
                   f"Who is the {sup_min if ask_min else sup_max}?")
        correct = D if ask_min else A
        wrong = [x for x in [A, B, C, D] if x != correct]
        options, letter = randomise_options(correct, wrong[:3])
        explanation = (f"Chain ({comp}): {A}>{B}>{C}>{D}. "
                      f"The {sup_min if ask_min else sup_max} is {correct}.")
        return make_task("compositional_reasoning", "easy", question, options,
                        letter, explanation, 3, 0)

    elif difficulty == "medium":
        names = pick_unique(REAL_NAMES, 7)
        chain = names[:5]
        decoys = names[5:7]
        A, B, C, D, E = chain
        ask_min = random.choice([True, False])
        # Inject misleading decoy comparison not in main chain
        decoy_stmt = f"{decoys[0]} is {comp} than {decoys[1]} but unrelated to the above group. "
        question = (f"{A} is {comp} than {B}. "
                   f"{B} is {comp} than {C}. "
                   f"{C} is {comp} than {D}. "
                   f"{D} is {comp} than {E}. "
                   + decoy_stmt +
                   f"Among {A}, {B}, {C}, {D}, and {E}, "
                   f"who is the {sup_min if ask_min else sup_max}?")
        correct = E if ask_min else A
        wrong = [x for x in chain if x != correct]
        random.shuffle(wrong)
        options, letter = randomise_options(correct, wrong[:3])
        explanation = (f"Chain ({comp}): {'>'.join(chain)}. "
                      f"The {sup_min if ask_min else sup_max} is {correct}. "
                      f"{decoys[0]}/{decoys[1]} are distractors outside the chain.")
        return make_task("compositional_reasoning", "medium", question, options,
                        letter, explanation, 4, 1)

    elif difficulty == "hard":
        # 8 hops + 2 decoy entities + non-extreme rank target
        n_chain = 8
        names = pick_unique(REAL_NAMES, n_chain + 2)
        chain = names[:n_chain]
        decoys = names[n_chain:]
        rank = random.choice([3, 4, 5, 6])
        correct = chain[rank - 1]
        ordinal = {3:"third",4:"fourth",5:"fifth",6:"sixth"}[rank]
        sup_label = f"{ordinal} {sup_max}"
        stmts = " ".join([f"{chain[i]} is {comp} than {chain[i+1]}."
                         for i in range(n_chain-1)])
        decoy_stmt = (f"{decoys[0]} is {comp} than {decoys[1]} "
                     f"but neither is part of the group above.")
        name_list = ", ".join(chain[:-1]) + f", and {chain[-1]}"
        question = (stmts + " " + decoy_stmt +
                   f" Among {name_list}, who is the {sup_label}?")
        wrong = [x for x in chain if x != correct]
        random.shuffle(wrong)
        options, letter = randomise_options(correct, wrong[:3])
        order_str = ">".join(chain)
        explanation = (f"Chain ({comp}): {order_str}. "
                      f"The {sup_label} is {correct} (position {rank}). "
                      f"{decoys[0]}/{decoys[1]} are distractors.")
        return make_task("compositional_reasoning", "hard", question, options,
                        letter, explanation, n_chain, 2)


# ─────────────────────────────────────────────────────────────
# GENERATOR 3: SYLLOGISTIC REASONING (v2.1)
# Key fix: negation embedded in premises; ask "what follows?"
# not "is this valid?" — forces reasoning through negation
# ─────────────────────────────────────────────────────────────

def generate_syllogistic(difficulty):
    if difficulty == "easy":
        words = pick_unique(MADE_UP_NOUNS, 3)
        A, B, C = words
        variant = random.choice(["all_chain", "no_premise", "none_conclusion"])
        if variant == "all_chain":
            question = (f"All {A} are {B}. All {B} are {C}. "
                       f"Given only these facts, which statement must be true?")
            correct = f"All {A} are {C}"
            wrong = [f"All {C} are {A}", f"Some {A} are not {C}", f"No {A} are {C}"]
            explanation = f"By transitivity: all {A} are {B}, all {B} are {C}, therefore all {A} are {C}."
        elif variant == "no_premise":
            question = (f"No {A} are {B}. All {C} are {A}. "
                       f"Given only these facts, which must be true?")
            correct = f"No {C} are {B}"
            wrong = [f"Some {C} are {B}", f"All {B} are {C}", f"All {A} are {C}"]
            explanation = f"All {C} are {A}, and no {A} are {B}, so no {C} can be {B}."
        else:
            question = (f"All {A} are {B}. No {C} are {B}. "
                       f"Given only these facts, which must be true?")
            correct = f"No {C} are {A}"
            wrong = [f"Some {C} are {A}", f"All {A} are {C}", f"No {A} are {B}"]
            explanation = f"No {C} are {B}, and all {A} are {B}, so no {C} can be {A}."
        options, letter = randomise_options(correct, wrong)
        return make_task("syllogistic_reasoning", "easy", question, options,
                        letter, explanation, 2, 0)

    elif difficulty == "medium":
        words = pick_unique(MADE_UP_NOUNS, 3)
        A, B, C = words
        # Rotate through 3 variants equally — valid, invalid with belief pull, cannot determine
        # This prevents models from learning to always pick "cannot determine"
        variant = random.choice(["valid_chain", "invalid_shared", "none_chain"])
        if variant == "valid_chain":
            # Valid syllogism — correct answer IS a definite conclusion
            question = (f"All {A} are {B}. No {C} are {B}. "
                       f"Which conclusion follows with certainty?")
            correct = f"No {C} are {A}"
            wrong = [f"Some {C} are {A}", f"All {A} are {C}",
                    f"Cannot determine the relationship between {C} and {A}"]
            explanation = (f"Valid: all {A} are {B}, and no {C} are {B}, "
                          f"so no {C} can be {A}.")
        elif variant == "invalid_shared":
            # Invalid — both share a category but no direct link
            question = (f"All {A} are {B}. All {C} are {B}. "
                       f"Which conclusion follows with certainty?")
            correct = f"Cannot determine if {A} and {C} are related"
            wrong = [f"All {A} are {C}", f"Some {A} are {C}", f"No {A} are {C}"]
            explanation = (f"Both {A} and {C} belong to {B} as separate subsets. "
                          f"No direct relationship between {A} and {C} can be established.")
        else:
            # Cannot determine — two negatives, no connection between A and C
            question = (f"No {A} are {B}. No {B} are {C}. "
                       f"Which conclusion follows with certainty?")
            correct = f"Cannot determine the relationship between {A} and {C}"
            wrong = [f"No {A} are {C}", f"All {A} are {C}", f"Some {A} are {C}"]
            explanation = (f"No {A} are {B} and no {B} are {C} establishes nothing "
                          f"direct about {A} and {C}. The relationship is undetermined.")
        options, letter = randomise_options(correct, wrong)
        return make_task("syllogistic_reasoning", "medium", question, options,
                        letter, explanation, 2, 0)

    elif difficulty == "hard":
        # Negation + real-world belief pull — model must reason against intuition
        words = pick_unique(MADE_UP_NOUNS, 2)
        A, B = words
        scenarios = [
            {
                "premises": f"All warm-blooded creatures require oxygen. All {A} require oxygen. No {B} are warm-blooded.",
                "question": f"Which conclusion follows with certainty?",
                "correct": f"Cannot determine if {A} are warm-blooded",
                "wrong": [f"All {A} are warm-blooded", f"No {A} require oxygen", f"All {B} require oxygen"],
                "exp": f"Requiring oxygen does not imply warm-blooded. {A} requiring oxygen is consistent with being non-warm-blooded. The conclusion about {A}'s warm-bloodedness cannot be determined."
            },
            {
                "premises": f"No {A} have ever been observed flying. All things that fly have wings. All {B} have wings.",
                "question": f"Which conclusion follows with certainty?",
                "correct": f"Cannot determine if {B} can fly",
                "wrong": [f"All {B} can fly", f"No {B} can fly", f"All {A} have wings"],
                "exp": f"Having wings is necessary but not sufficient for flying from the given premises. We can only conclude that {B} have wings, not that they fly."
            },
            {
                "premises": f"All {A} are classified as essential. All essential items are regulated. No {B} are regulated.",
                "question": f"Which conclusion follows with certainty?",
                "correct": f"No {B} are {A}",
                "wrong": [f"No {A} are {B}", f"All {B} are essential", f"Some {B} are regulated"],
                "exp": f"All {A} are essential → regulated. No {B} are regulated. Therefore no {B} can be {A}."
            },
        ]
        sc = random.choice(scenarios)
        question = sc["premises"] + " " + sc["question"]
        options, letter = randomise_options(sc["correct"], sc["wrong"])
        return make_task("syllogistic_reasoning", "hard", question, options,
                        letter, sc["exp"], 3, 0)


# ─────────────────────────────────────────────────────────────
# GENERATOR 4: WORKING MEMORY (v2.1)
# Key fix: same key updated MULTIPLE times (5-8x at hard)
# Model must retrieve FINAL value — not earlier overwritten one
# Grounded in PI-LLM paper (Wang & Sun, 2025)
# ─────────────────────────────────────────────────────────────

def generate_working_memory(difficulty):
    cat = random.choice(WM_CATEGORIES)
    key = cat["key"]
    values = cat["values"].copy()
    random.shuffle(values)

    if difficulty == "easy":
        # Key updated 4 times; ask for the SECOND value (not first, not last)
        # Forces model to track history rather than reading first or last
        updates = values[:4]
        stmts = []
        stmts.append(f"Initially, the {key} was set to {updates[0]}.")
        stmts.append(f"The {key} was then updated to {updates[1]}.")
        stmts.append(f"Later, the {key} changed to {updates[2]}.")
        stmts.append(f"Finally, the {key} was revised to {updates[3]}.")
        question = " ".join(stmts) + f" What was the second value recorded for the {key}?"
        correct = updates[1]
        wrong = [updates[0], updates[2], updates[3]]
        options, letter = randomise_options(correct, wrong)
        explanation = (f"The {key} was updated 4 times: {updates[0]} → {updates[1]} → "
                      f"{updates[2]} → {updates[3]}. "
                      f"After the first update it was {updates[1]}. "
                      f"Models default to the first ({updates[0]}) or last ({updates[3]}) value.")
        return make_task("working_memory", "easy", question, options, letter,
                        explanation, 4, 0)

    elif difficulty == "medium":
        # Same key updated 6 times; 3 distractor key updates interspersed
        # Ask for the THIRD value — buried in the middle
        updates = values[:6]
        other_cat = random.choice([c for c in WM_CATEGORIES if c["key"] != key])
        distractor_vals = random.sample(other_cat["values"], 3)
        stmts = []
        stmts.append(f"The {key} is {updates[0]}.")
        stmts.append(f"The {other_cat['key']} is {distractor_vals[0]}.")
        stmts.append(f"The {key} has been updated to {updates[1]}.")
        stmts.append(f"The {other_cat['key']} has been changed to {distractor_vals[1]}.")
        stmts.append(f"The {key} is now {updates[2]}.")
        stmts.append(f"The {other_cat['key']} is now {distractor_vals[2]}.")
        stmts.append(f"The {key} has been revised to {updates[3]}.")
        stmts.append(f"The {key} was subsequently changed to {updates[4]}.")
        stmts.append(f"The {key} has been finalised as {updates[5]}.")
        question = (" ".join(stmts) +
                   f" What was the {key} at its third recorded value?")
        correct = updates[2]
        wrong = [updates[0], updates[3], updates[5]]
        options, letter = randomise_options(correct, wrong)
        explanation = (f"The {key} sequence: {' → '.join(updates)}. "
                      f"The third value is {updates[2]}. "
                      f"Models default to the first ({updates[0]}) or last ({updates[5]}) value.")
        return make_task("working_memory", "medium", question, options, letter,
                        explanation, 6, 3)

    elif difficulty == "hard":
        # Key updated 7 times; 3 distractor keys updated multiple times
        # Ask for the FOURTH value — deeply buried in the middle
        updates = values[:7]
        other_cats = random.sample([c for c in WM_CATEGORIES if c["key"] != key], 3)
        stmts = []
        stmts.append(f"The {key} is {updates[0]}.")
        stmts.append(f"The {other_cats[0]['key']} is {random.choice(other_cats[0]['values'])}.")
        stmts.append(f"The {key} has changed to {updates[1]}.")
        stmts.append(f"The {other_cats[1]['key']} is {random.choice(other_cats[1]['values'])}.")
        stmts.append(f"The {key} is now {updates[2]}.")
        stmts.append(f"The {other_cats[2]['key']} is {random.choice(other_cats[2]['values'])}.")
        stmts.append(f"The {key} has been revised to {updates[3]}.")
        stmts.append(f"The {other_cats[0]['key']} has changed to {random.choice(other_cats[0]['values'])}.")
        stmts.append(f"The {key} is updated to {updates[4]}.")
        stmts.append(f"The {other_cats[1]['key']} is now {random.choice(other_cats[1]['values'])}.")
        stmts.append(f"The {key} has been set to {updates[5]}.")
        stmts.append(f"The {other_cats[2]['key']} has changed to {random.choice(other_cats[2]['values'])}.")
        stmts.append(f"The {key} is finally confirmed as {updates[6]}.")
        question = (" ".join(stmts) +
                   f" What was the {key} at its fifth recorded value?")
        correct = updates[4]
        wrong = [updates[0], updates[2], updates[6]]
        options, letter = randomise_options(correct, wrong)
        explanation = (f"The {key} sequence: {' → '.join(updates)}. "
                      f"The fifth value is {updates[4]}. "
                      f"This is deeply buried among 7 updates with 5 interfering keys — "
                      f"models default to first ({updates[0]}) or last ({updates[6]}) values.")
        return make_task("working_memory", "hard", question, options, letter,
                        explanation, 7, 5)


# ─────────────────────────────────────────────────────────────
# GENERATOR 5: INHIBITORY CONTROL (v2.1)
# Key fix: uses deeply ingrained real patterns from training data
# Override rule is subtle and buried — not highlighted
# ─────────────────────────────────────────────────────────────

def generate_inhibitory_control(difficulty):
    if difficulty == "easy":
        # Override of a deeply trained number sequence pattern
        scenarios = [
            {
                "setup": "For the purpose of this task, the number that follows 9 is 1 (counting wraps around in a cycle of 9). What number comes after 9?",
                "correct": "1 — the stated rule says 9 wraps back to 1",
                "wrong": ["10 — that is the standard sequence",
                         "0 — zero follows 9 sometimes", "Cannot be determined"],
                "exp": "The stated rule overrides the standard sequence. A model with weak inhibitory control answers 10."
            },
            {
                "setup": "In this system only, the word that is the opposite of HOT is WET (not COLD). What is the opposite of HOT in this system?",
                "correct": "WET — the stated rule defines HOT's opposite as WET",
                "wrong": ["COLD — that is the normal opposite",
                         "WARM", "Cannot be determined"],
                "exp": "The stated rule overrides the standard antonym. A model with weak inhibitory control answers COLD."
            },
            {
                "setup": "For this exercise only, red lights mean GO and green lights mean STOP. A driver sees a red light. What should they do according to these rules?",
                "correct": "Go — the stated rule says red means GO",
                "wrong": ["Stop — red normally means stop",
                         "Slow down", "Cannot be determined"],
                "exp": "The stated rule reverses traffic signals. A model with weak inhibitory control applies the real-world rule."
            },
        ]
        sc = random.choice(scenarios)
        options, letter = randomise_options(sc["correct"], sc["wrong"])
        return make_task("inhibitory_control", "easy", sc["setup"], options,
                        letter, sc["exp"], 1, 0)

    elif difficulty == "medium":
        # Establish strong context pattern across 4 examples, then subtly state override
        nums = [random.sample(range(1, 20), 3) for _ in range(5)]
        examples = nums[:4]
        test = nums[4]
        question = (f"Observe the pattern: "
                   f"Group 1: {examples[0]} — answer is {max(examples[0])}. "
                   f"Group 2: {examples[1]} — answer is {max(examples[1])}. "
                   f"Group 3: {examples[2]} — answer is {max(examples[2])}. "
                   f"Group 4: {examples[3]} — answer is {max(examples[3])}. "
                   f"Note: for Group 5 and beyond, select the median value instead. "
                   f"Group 5: {test} — what is the answer?")
        sorted_test = sorted(test)
        correct = str(sorted_test[1])  # median
        wrong_max = str(max(test))
        wrong_min = str(min(test))
        wrong_sum = str(sum(test))
        options, letter = randomise_options(
            correct, [wrong_max, wrong_min, wrong_sum])
        explanation = (f"The new rule says select the median of Group 5: {sorted_test}. "
                      f"The median is {correct}. "
                      f"A model with weak inhibitory control answers {wrong_max} (the maximum), "
                      f"following the established pattern.")
        return make_task("inhibitory_control", "medium", question, options,
                        letter, explanation, 4, 0)

    elif difficulty == "hard":
        # 6 examples establishing pattern; new rule stated casually mid-paragraph;
        # two distractors reinforce old pattern; question deliberately mirrors old format
        examples = [random.sample(range(1, 25), 3) for _ in range(7)]
        old_answers = [max(e) for e in examples[:6]]
        test = examples[6]
        sorted_test = sorted(test)
        new_correct = sorted_test[0]  # minimum — rule change
        question = (
            f"Results so far: "
            + " | ".join([f"Round {i+1}: {examples[i]} → {old_answers[i]}"
                         for i in range(6)])
            + f". As you can see, the pattern has been consistent. "
            f"Going forward the scoring methodology has been updated — "
            f"teams now receive points equal to the lowest value. "
            f"Historical performance: Round 1 winner scored {old_answers[0]}, "
            f"all-time high was {max(old_answers)}. "
            f"Round 7: {test} — what is the score?"
        )
        correct = str(new_correct)
        wrong = [str(max(test)),
                str(sorted_test[1]),
                f"Same as before: {max(test)}"]
        options, letter = randomise_options(correct, wrong)
        explanation = (f"New rule: lowest value. Round 7 {test} sorted = {sorted_test}. "
                      f"Lowest = {new_correct}. "
                      f"Historical references ({old_answers[0]}, {max(old_answers)}) "
                      f"are distractors reinforcing the old maximum pattern.")
        return make_task("inhibitory_control", "hard", question, options,
                        letter, explanation, 6, 2)


# ─────────────────────────────────────────────────────────────
# GENERATOR 6: COUNTING (v2.1)
# Expanded pools — same structure, more variety
# ─────────────────────────────────────────────────────────────

def generate_counting(difficulty):
    # Note: sentences (originally "hard") are easier for models than
    # words (originally "medium") because sentence context aids letter location.
    # Difficulty levels swapped: medium now uses sentences, hard uses words.
    if difficulty == "easy":
        item = random.choice(COUNTING_WORDS_EASY)
        word, letter, correct_count = item
        question = f"How many times does the letter '{letter}' appear in the word {word}?"
        correct = str(correct_count)
        offsets = [-2, -1, 1, 2]
        random.shuffle(offsets)
        wrongs = []
        for d in offsets:
            c = str(correct_count + d)
            if c != correct and int(c) >= 0 and c not in wrongs:
                wrongs.append(c)
            if len(wrongs) == 3:
                break
        options, letter_opt = randomise_options(correct, wrongs)
        spelled = "-".join(list(word))
        explanation = f"{spelled}: '{letter}' appears {correct_count} time(s)."
        return make_task("counting", "easy", question, options, letter_opt,
                        explanation, 1, 0)

    elif difficulty == "medium":
        # Sentences placed at medium (easier than words due to context)
        item = random.choice(COUNTING_SENTENCES_HARD)
        sentence, letter, correct_count = item
        question = (
            f"Count every occurrence of the letter '{letter}' "
            f"(both uppercase and lowercase) in this sentence: "
            f'"{sentence}" How many times does it appear?'
        )
        correct = str(correct_count)
        offsets = [-3, -2, -1, 1, 2, 3]
        random.shuffle(offsets)
        wrongs = []
        for d in offsets:
            c = str(correct_count + d)
            if c != correct and int(c) >= 0 and c not in wrongs:
                wrongs.append(c)
            if len(wrongs) == 3:
                break
        options, letter_opt = randomise_options(correct, wrongs)
        explanation = (
            f"The letter '{letter}' appears {correct_count} times in the sentence. "
            f"Counting across a sentence is moderately challenging for LLMs."
        )
        return make_task("counting", "medium", question, options, letter_opt,
                        explanation, 1, 0)

    elif difficulty == "hard":
        # Words placed at hard (harder than sentences due to tokenisation)
        item = random.choice(COUNTING_WORDS_MEDIUM)
        word, letter, correct_count = item
        question = (
            f"How many times does the letter '{letter}' appear "
            f"(uppercase or lowercase) in the word {word}?"
        )
        correct = str(correct_count)
        offsets = [-3, -2, -1, 1, 2, 3]
        random.shuffle(offsets)
        wrongs = []
        for d in offsets:
            c = str(correct_count + d)
            if c != correct and int(c) >= 0 and c not in wrongs:
                wrongs.append(c)
            if len(wrongs) == 3:
                break
        options, letter_opt = randomise_options(correct, wrongs)
        explanation = (
            f"In '{word}': the letter '{letter}' appears exactly {correct_count} time(s). "
            f"Word-level character counting is the hardest counting task for LLMs "
            f"because models process words as atomic tokens, not character sequences."
        )
        return make_task("counting", "hard", question, options, letter_opt,
                        explanation, 1, 0)



# ─────────────────────────────────────────────────────────────
# GENERATOR 7: ANCHORING BIAS (v2.1)
# Key fix: anchor is mathematically plausible in the calculation
# Model must IGNORE a number that looks like it belongs
# ─────────────────────────────────────────────────────────────

def generate_anchoring_bias(difficulty):
    if difficulty == "easy":
        # Anchor is a number that looks like it could be added/used
        # Correct answer = sum of two values
        # Wrong options: individual values + a plausible near-miss (total +/- small amount)
        a = random.randint(40, 90)
        b = random.randint(20, 39)
        total = a + b
        near_miss = total + random.choice([-5, -3, 3, 5, 7])
        item = random.choice(["reports", "files", "units", "packages", "items"])
        person1 = random.choice(REAL_NAMES)
        person2 = random.choice([n for n in REAL_NAMES if n != person1])
        dept = random.choice(["department", "team", "division"])
        question = (f"The {dept} logged {near_miss} {item} last week. "
                   f"This week, {person1} completed {a} {item} "
                   f"and {person2} completed {b} {item}. "
                   f"How many {item} did {person1} and {person2} complete this week in total?")
        correct = str(total)
        wrong = [str(a), str(b), str(near_miss)]
        options, letter = randomise_options(correct, wrong)
        explanation = (f"The answer is {a} + {b} = {total}. "
                      f"Last week's total ({near_miss}) is the salient anchor — "
                      f"close in magnitude to the correct answer. "
                      f"{a} and {b} are individual component anchors.")
        return make_task("anchoring_bias", "easy", question, options, letter,
                        explanation, 1, 1)

    elif difficulty == "medium":
        # Anchor number could plausibly be part of the answer calculation
        # Correct answer requires subtraction; anchors are the minuend and a plausible wrong result
        total = random.randint(150, 300)
        part1 = random.randint(60, total - 40)
        correct_val = total - part1
        item = random.choice(["units", "boxes", "files", "items", "records"])
        dept = random.choice(["department", "team", "division", "unit"])
        person = random.choice(REAL_NAMES)
        question = (f"The {dept} processed {total} {item} this month. "
                   f"{part1} of them were handled in the first two weeks. "
                   f"The annual target is {total * random.randint(10,14)} {item}. "
                   f"{person} managed all {item} not handled in the first two weeks. "
                   f"How many {item} did {person} manage?")
        correct = str(correct_val)
        wrong = [str(total), str(part1), str(total + part1)]
        options, letter = randomise_options(correct, wrong)
        explanation = (f"The answer is {total} - {part1} = {correct_val}. "
                      f"{total} and {part1} are salient anchors. "
                      f"Model must subtract, not pick a stated number.")
        return make_task("anchoring_bias", "medium", question, options, letter,
                        explanation, 1, 2)

    elif difficulty == "hard":
        # Hard: answer requires two-step calculation (multiply then subtract)
        # Three anchors are each plausible single-step results
        rate = random.randint(8, 15)
        days = random.randint(5, 9)
        total_produced = rate * days
        defective = random.randint(10, 30)
        correct_val = total_produced - defective
        item = random.choice(["parts", "units", "samples", "components"])
        machine = random.choice(["Machine A", "Unit B", "Line C", "Station D"])
        question = (f"{machine} produces {rate} {item} per day. "
                   f"It ran for {days} days this cycle. "
                   f"The warehouse currently holds {total_produced + random.randint(50,100)} {item} in total. "
                   f"Quality control rejected {defective} {item} as defective. "
                   f"How many non-defective {item} did {machine} produce this cycle?")
        correct = str(correct_val)
        wrong = [str(total_produced), str(defective), str(rate)]
        options, letter = randomise_options(correct, wrong)
        explanation = (f"Step 1: {rate} × {days} = {total_produced} total produced. "
                      f"Step 2: {total_produced} - {defective} = {correct_val} non-defective. "
                      f"{total_produced}, {defective}, and {rate} are all salient anchors.")
        return make_task("anchoring_bias", "hard", question, options, letter,
                        explanation, 2, 3)


# ─────────────────────────────────────────────────────────────
# GENERATOR 8: THEORY OF MIND (v2.1)
# Key fix: novel non-standard professional scenarios
# unlikely to appear in standard training data
# ─────────────────────────────────────────────────────────────

def generate_theory_of_mind(difficulty):
    # Easy:   simplest first-order false belief (object moved, agent left unaware)
    #         Uses first 8 TOM_EASY scenarios — most unambiguous
    # Medium: second-order belief (what does A think B believes?)
    # Hard:   third-order belief (what does A think B thinks C believes?)
    if difficulty == "easy":
        pool = TOM_EASY[:8]    # simplest first-order scenarios only
    elif difficulty == "medium":
        pool = TOM_MEDIUM      # second-order professional scenarios
    elif difficulty == "hard":
        pool = TOM_HARD        # third-order belief

    item = random.choice(pool)
    correct_text = item["opts"][item["ans"]]
    wrong_texts = [v for k, v in item["opts"].items() if k != item["ans"]]
    options, letter = randomise_options(correct_text, wrong_texts)
    steps_map = {"easy": 2, "medium": 4, "hard": 6}
    dist_map = {"easy": 0, "medium": 1, "hard": 2}
    return make_task("theory_of_mind", difficulty, item["q"],
                    options, letter, item["exp"],
                    steps_map[difficulty], dist_map[difficulty])


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_task(task):
    issues = []
    if len(task["options"]) != 4:
        issues.append(f"Expected 4 options, got {len(task['options'])}")
    if task["answer"] not in ["A", "B", "C", "D"]:
        issues.append(f"Invalid answer: {task['answer']}")
    if task["answer"] not in task["options"]:
        issues.append("Answer not in options")
    for k, v in task["options"].items():
        if not isinstance(v, str) or not v.strip():
            issues.append(f"Option {k} empty")
    if len(set(task["options"].values())) != 4:
        issues.append("Duplicate options")
    for f in ["category", "difficulty", "num_steps", "num_distractors"]:
        if f not in task:
            issues.append(f"Missing: {f}")
    if not task.get("question", "").strip():
        issues.append("Empty question")
    if not task.get("explanation", "").strip():
        issues.append("Empty explanation")
    return len(issues) == 0, issues


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

GENERATORS = {
    "reversal_curse":          generate_reversal_curse,
    "compositional_reasoning": generate_compositional,
    "syllogistic_reasoning":   generate_syllogistic,
    "working_memory":          generate_working_memory,
    "inhibitory_control":      generate_inhibitory_control,
    "counting":                generate_counting,
    "anchoring_bias":          generate_anchoring_bias,
    "theory_of_mind":          generate_theory_of_mind,
}

DISTRIBUTION = {cat: {"easy": 50, "medium": 50, "hard": 50}
                for cat in GENERATORS}


def generate_dataset(distribution=None, seed=42):
    random.seed(seed)
    if distribution is None:
        distribution = DISTRIBUTION
    dataset, stats = [], {"total": 0, "passed": 0, "failed": 0, "by_cat": {}}
    for cat, diffs in distribution.items():
        stats["by_cat"][cat] = {"total": 0}
        gen = GENERATORS[cat]
        for diff, count in diffs.items():
            generated, attempts = 0, 0
            while generated < count and attempts < count * 10:
                attempts += 1
                try:
                    task = gen(diff)
                    passed, issues = validate_task(task)
                    if passed:
                        dataset.append(task)
                        generated += 1
                        stats["total"] += 1
                        stats["passed"] += 1
                        stats["by_cat"][cat]["total"] += 1
                    else:
                        stats["failed"] += 1
                        print(f"  [FAIL] {cat}/{diff}: {issues}")
                except Exception as ex:
                    stats["failed"] += 1
                    print(f"  [ERROR] {cat}/{diff}: {ex}")
    random.shuffle(dataset)
    return dataset, stats


def print_summary(stats, dataset):
    print("\n" + "=" * 60)
    print("FACET v2 — Dataset Generation Summary")
    print("=" * 60)
    print(f"Total : {stats['total']}  |  Passed: {stats['passed']}  |  Failed/retried: {stats['failed']}")
    print()
    for cat, info in stats["by_cat"].items():
        print(f"  {cat:<32} {info['total']:>4}")
    print(f"  {'TOTAL':<32} {stats['total']:>4}")
    print()
    shown = set()
    for task in dataset:
        c = task["category"]
        if c not in shown:
            shown.add(c)
            print(f"\n[{c.upper()} | {task['difficulty']}]")
            print(f"Q: {task['question'][:120]}...")
            for k, v in task["options"].items():
                m = "  <-- correct" if k == task["answer"] else ""
                print(f"   ({k}) {v[:60]}{m}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FACET v2 Generator")
    parser.add_argument("--type", choices=list(GENERATORS.keys()))
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--output", default="facet_v2_dataset.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate:
        print("Validating 3 tasks per category per difficulty...")
        all_ok = True
        for cat, gen in GENERATORS.items():
            for diff in ["easy", "medium", "hard"]:
                for _ in range(3):
                    try:
                        task = gen(diff)
                        ok, issues = validate_task(task)
                        if not ok:
                            print(f"  FAIL [{cat}/{diff}]: {issues}")
                            all_ok = False
                    except Exception as e:
                        print(f"  ERROR [{cat}/{diff}]: {e}")
                        all_ok = False
        print("All checks passed." if all_ok else "Some checks failed.")

    elif args.type:
        diff = args.difficulty or "easy"
        gen = GENERATORS[args.type]
        for _ in range(args.n):
            task = gen(diff)
            print(f"\n[{task['category'].upper()} | {task['difficulty']}]")
            print(f"Q: {task['question']}")
            for k, v in task["options"].items():
                m = "  <-- correct" if k == task["answer"] else ""
                print(f"   ({k}) {v}{m}")
            print(f"Exp: {task['explanation']}")
    else:
        print(f"Generating FACET v2 dataset (seed={args.seed})...")
        dataset, stats = generate_dataset(seed=args.seed)
        print_summary(stats, dataset)
        out = {
            "name": "FACET", "version": "2.1",
            "generated_at": datetime.now().isoformat(),
            "seed": args.seed, "total_tasks": len(dataset),
            "categories": list(GENERATORS.keys()),
            "tasks": dataset,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nDataset saved to: {args.output}")