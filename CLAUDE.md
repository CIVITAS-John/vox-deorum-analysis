# Project Instructions

- Avoid absolute paths; use relative paths whenever possible.

# Tool Usage

Bash commands that duplicate built-in tool functionality are **auto-rejected**. Always use the correct built-in tool instead.

| Task | Use This Tool | Do NOT Use |
|------|--------------|------------|
| Read files | `Read` | `cat`, `head`, `tail`, `less`, `more` |
| Edit files | `Edit` | `sed`, `awk`, inline shell edits |
| Create/overwrite files | `Write` | `echo >`, `cat <<EOF`, heredocs |
| Search file contents | `Grep` | `grep`, `rg`, `ag` |
| Find files by name/pattern | `Glob` | `find`, `ls`, `dir` |

Reserve `Bash` exclusively for operations that have no built-in equivalent (e.g., `git`, `npm`, `pip`, `python`, running scripts, system commands).

# Feature Pipeline

All feature engineering and selection is centralized in `models/utils/data_utils.py`. See the module docstring for:
- Full pipeline stages (load → adjust → relative → competitive → drop → select)
- Configuration hierarchy (`FEATURE_GROUPS` → `SELECTED_FEATURES` → `Model.DEFAULT_FEATURES`)
- Complete feature reference with formulas and descriptions
