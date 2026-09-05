# Security Policy

## Supported Versions

OpenModels is pre-1.0 (currently in alpha/beta). Only the latest published release is
supported with security fixes.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report it privately via
[GitHub Security Advisories](https://github.com/Gnpd/openmodels/security/advisories/new),
or by emailing a.gutierrez@g-npd.com. Include as much detail as you can (affected version,
reproduction steps, impact) so we can assess and respond quickly.

## Deserialization Safety

OpenModels ships two output formats with very different trust requirements:

- **JSON** (`format_name="json"`, the default): the serialized payload is plain data
  (numbers, strings, arrays). Deserializing it does not execute arbitrary code, and it is
  safe to load JSON produced by an untrusted party in the same way loading any other JSON
  document is.

- **Pickle** (`format_name="pickle"`): under the hood this calls Python's `pickle.loads()`,
  which can execute arbitrary code as part of deserialization. This is a property of the
  pickle format itself, not something OpenModels adds or can remove.
  **Only load pickle files from sources you trust**, exactly as you would with `pickle`,
  `joblib`, or any other pickle-based tool. Never call
  `SerializationManager.load(..., format_name="pickle")` /
  `SerializationManager.deserialize(..., format_name="pickle")` on a file received from an
  untrusted or unauthenticated source.

  Note that this isn't `pickle.dumps()`/`joblib.dump()` on the fitted estimator itself -
  OpenModels always converts the model to the same plain dict documented in
  `docs/format.md` first (the one `estimator_class`/`params`/`attributes`/`metadata` shape
  the JSON format also uses), and only then hands that dict to `pickle.dumps()`. Choosing
  the pickle format changes how that dict is encoded on disk, not what gets serialized -
  the arbitrary-code-execution risk comes entirely from calling `pickle.loads()` at all,
  regardless of how harmless the encoded content happens to be.

If you don't have a specific reason to use the pickle format, prefer JSON — it's the
default, it's human-readable, and it doesn't carry this risk.
