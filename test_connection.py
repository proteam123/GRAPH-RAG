import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase, TrustCustomCAs
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable


def main() -> int:
    load_dotenv()

    uri = os.getenv("NEO4J_URI")
    password = os.getenv("NEO4J_PASSWORD")
    missing = [
        name
        for name, value in (
            ("NEO4J_URI", uri),
            ("NEO4J_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        return 1

    username = os.getenv("NEO4J_USERNAME", "neo4j")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    driver_config = {}
    trusted_ca_file = os.getenv("NEO4J_TRUSTED_CA_FILE")
    if trusted_ca_file:
        ca_path = Path(trusted_ca_file).expanduser()
        if not ca_path.is_file():
            print(f"NEO4J_TRUSTED_CA_FILE does not point to a file: {ca_path}", file=sys.stderr)
            return 1
        driver_config["trusted_certificates"] = TrustCustomCAs(str(ca_path))

    try:
        with GraphDatabase.driver(
            uri,
            auth=(username, password),
            **driver_config,
        ) as driver:
            driver.verify_connectivity()
            with driver.session(database=database) as session:
                record = session.run("RETURN 1 AS result").single()
    except AuthError:
        print("Authentication failed: check NEO4J_USERNAME and NEO4J_PASSWORD.", file=sys.stderr)
        return 1
    except ServiceUnavailable as exc:
        print(
            f"Could not reach Neo4j at {uri}: {exc}\n"
            "Check the URI, that the instance is running (free Aura instances pause after "
            "inactivity), and your network/firewall.",
            file=sys.stderr,
        )
        return 1
    except Neo4jError as exc:
        print(f"Neo4j connection test failed: {exc}", file=sys.stderr)
        return 1

    if record is None or record["result"] != 1:
        print("Neo4j connected, but the test query returned an unexpected result.", file=sys.stderr)
        return 1

    print(f"Neo4j connection successful (database: {database}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())