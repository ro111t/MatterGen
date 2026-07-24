"""
db_schema.py — SQLite schema definitions for CareerMemory.

All CREATE TABLE statements live here so career_memory.py
stays focused on business logic.
"""


SCHEMA_CAMPAIGNS = """
    CREATE TABLE IF NOT EXISTS campaigns (
        id              TEXT PRIMARY KEY,
        name            TEXT NOT NULL,
        domain          TEXT NOT NULL,
        objective       TEXT NOT NULL,
        start_time      REAL,
        end_time        REAL,
        iterations      INTEGER DEFAULT 0,
        total_generated INTEGER DEFAULT 0,
        total_screened  INTEGER DEFAULT 0,
        best_score      REAL    DEFAULT 0.0,
        success_rate    REAL    DEFAULT 0.0,
        summary         TEXT
    )
"""

SCHEMA_PRINCIPLES = """
    CREATE TABLE IF NOT EXISTS principles (
        id                   TEXT PRIMARY KEY,
        domain               TEXT NOT NULL,
        statement            TEXT NOT NULL,
        confidence           REAL DEFAULT 0.5,
        supporting_campaigns TEXT DEFAULT '[]',
        refuting_campaigns   TEXT DEFAULT '[]',
        property_target      TEXT,
        structural_motif     TEXT,
        source_type          TEXT DEFAULT 'inferred',
        created_at           REAL,
        updated_at           REAL
    )
"""

SCHEMA_FAILURES = """
    CREATE TABLE IF NOT EXISTS failure_attributions (
        id                   TEXT PRIMARY KEY,
        campaign_id          TEXT,
        domain               TEXT,
        formula              TEXT,
        failure_mode         TEXT,
        structural_features  TEXT,
        property_predictions TEXT,
        attributed_cause     TEXT,
        created_at           REAL,
        FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
    )
"""

SCHEMA_CROSS_DOMAIN = """
    CREATE TABLE IF NOT EXISTS cross_domain_links (
        id                  TEXT PRIMARY KEY,
        source_domain       TEXT,
        target_domain       TEXT,
        source_principle_id TEXT,
        analogy_description TEXT,
        confidence          REAL DEFAULT 0.5,
        created_at          REAL
    )
"""

SCHEMA_CANDIDATES = """
    CREATE TABLE IF NOT EXISTS candidates (
        id               TEXT PRIMARY KEY,
        campaign_id      TEXT,
        domain           TEXT,
        formula          TEXT,
        score            REAL,
        passed_screening INTEGER,
        hypothesis_ids   TEXT DEFAULT '[]',
        principle_ids    TEXT DEFAULT '[]',
        properties       TEXT DEFAULT '{}',
        iteration        INTEGER,
        created_at       REAL,
        FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
    )
"""

SCHEMA_HYPOTHESES = """
    CREATE TABLE IF NOT EXISTS hypotheses (
        id                  TEXT PRIMARY KEY,
        campaign_id         TEXT,
        iteration           INTEGER,
        statement           TEXT,
        basis               TEXT,
        source_principle_ids TEXT DEFAULT '[]',
        source_domains      TEXT DEFAULT '[]',
        outcome             TEXT DEFAULT 'pending',
        confidence_before   REAL DEFAULT 0.5,
        confidence_after    REAL,
        created_at          REAL
    )
"""

ALL_SCHEMAS = [
    SCHEMA_CAMPAIGNS,
    SCHEMA_PRINCIPLES,
    SCHEMA_FAILURES,
    SCHEMA_CROSS_DOMAIN,
    SCHEMA_CANDIDATES,
    SCHEMA_HYPOTHESES,
]
