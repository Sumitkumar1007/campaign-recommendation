from __future__ import annotations

from dataclasses import dataclass

import psycopg
from psycopg import sql


@dataclass
class PostgresConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str


def connect_db(config: PostgresConfig) -> psycopg.Connection:
    return psycopg.connect(
        host=config.host,
        port=config.port,
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        autocommit=False,
    )


def qualified_identifier(schema: str, table: str) -> sql.Composed:
    return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))
