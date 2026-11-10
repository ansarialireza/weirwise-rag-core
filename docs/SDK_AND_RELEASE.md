# Core SDK and Release Conventions

## Separate distributions

| Distribution | Project/version | Runtime and public entry |
|---|---|---|
| Core server | `weirwise-core`, currently `1.0.0`, Python `>=3.11` | FastAPI service, ingestion worker, CLI, and REST API. |
| Python SDK | `weirwise`, currently `1.0.0`, Python `>=3.9` | Typed synchronous HTTP client under `packages/python-sdk/`, with `py.typed`. |

The SDK uses Core's REST contract; it is independently packaged and tested. Keep SDK models,
resource methods, and exception mapping aligned with the REST API without coupling the server to
the SDK. Platform currently uses its own HTTP adapter and does not import Core internals.

## Compatibility and release rules

- REST routes/schemas and SDK public classes, methods, model fields, exception behavior, and
  supported Python versions are compatibility-sensitive.
- Add tests and migration/deprecation guidance before removing or changing public behavior.
- Keep server and SDK release notes/changelogs accurate to their own packages. Product roadmap
  versions (V1.0, V2.0, etc.) are not Core package/API release numbers.
- Preserve `py.typed` and verify the SDK wheel includes it when changing packaging.
- Use the repository's CI commands and release workflow as the source for package build/publish
  steps; do not claim a package was built or released unless that action was actually run.

For development and checks, see [`../CONTRIBUTING.md`](../CONTRIBUTING.md),
[`TESTING.md`](TESTING.md), and the package's [`../packages/python-sdk/README.md`](../packages/python-sdk/README.md).
