import os

import psycopg
from dotenv import load_dotenv


load_dotenv()

DB = os.getenv("POSTGRES_DB")
USER = os.getenv("POSTGRES_USER")
PASSWORD = os.getenv("POSTGRES_PASSWORD")
HOST = os.getenv("POSTGRES_HOST", "localhost")
PORT = os.getenv("POSTGRES_PORT", "5432")

if not all([DB, USER, PASSWORD]):
    raise RuntimeError(
        "Database configuration is incomplete. "
        "Define POSTGRES_DB, POSTGRES_USER, and POSTGRES_PASSWORD in your .env file."
    )

DATABASE_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"


def get_connection() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, autocommit=True)


def ensure_schema() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS plates (
                id SERIAL PRIMARY KEY,
                plate_text_norm TEXT UNIQUE NOT NULL
            )
            """
        )
        cur.execute(
            """
            INSERT INTO plates (plate_text_norm)
            VALUES
                ('XX11111'),
                ('XX22222'),
                ('XX33333')
            ON CONFLICT (plate_text_norm) DO NOTHING
            """
        )


def plate_exists(plate_text_norm: str) -> bool:
    if not plate_text_norm:
        return False

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM plates WHERE plate_text_norm = %s LIMIT 1",
            (plate_text_norm,),
        )
        return cur.fetchone() is not None


def insert_plate(plate_text_norm: str) -> bool:
    if not plate_text_norm:
        return False

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO plates (plate_text_norm)
            VALUES (%s)
            ON CONFLICT (plate_text_norm) DO NOTHING
            RETURNING id
            """,
            (plate_text_norm,),
        )
        return cur.fetchone() is not None


def list_plates(limit: int = 100) -> list[str]:
    safe_limit = max(1, min(limit, 1000))

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT plate_text_norm
            FROM plates
            ORDER BY id DESC
            LIMIT %s
            """,
            (safe_limit,),
        )
        return [row[0] for row in cur.fetchall()]

