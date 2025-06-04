## Contributing

We welcome contributions!

1. **Fork** this repository.
2. **Create a new branch** for your feature, fix, or improvement.
3. **Make your changes** (add a Tunnel, Tool, bug fix, etc).
4. **Open a Pull Request (PR)** to the main branch.
5. Your PR will be reviewed and, once approved, merged.
6. You’ll be listed as a project contributor!

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## Contribution Checklist

- [ ] Opened an issue to discuss any major changes *(issues are recommended, but PRs without an issue are also welcome)*
- [ ] Forked the repo & created a new branch
- [ ] Code includes docstrings and type hints (see below)
- [ ] Relevant tests/examples added/updated
- [ ] Ran the pipeline to confirm no errors
- [ ] Followed code style guide (see below)
- [ ] Docs updated if necessary
- [ ] Commit messages are clear
- [ ] Merged latest `main` if needed
- [ ] No secrets/credentials in PR

## Code Style Guide

- PEP8 style, 4 spaces indent, max 120 chars/line
- **Type annotations (type hints) are required for all Tunnel and Tool files.**
- Type hints are strongly recommended for all other new code.
- Use descriptive variable and function names
- Add docstrings to all functions/classes
- Prefer f-strings for formatting
- One tunnel/tool per file if possible
- Group imports: stdlib, 3rd-party, local
- Keep PRs clean and minimal
