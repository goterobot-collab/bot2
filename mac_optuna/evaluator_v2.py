#!/usr/bin/env python3
"""Framework v5 Evaluator — Binary scoring for auto-improvement loops."""

import os
import sys
import json
from pathlib import Path

def check(name, condition):
    """Binary check: PASS or FAIL."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    return 1 if condition else 0

def evaluate(project_path="."):
    p = Path(project_path)
    score = 0
    total = 0

    print(f"\n{'='*50}")
    print(f"Framework v5 Evaluator — {p.resolve()}")
    print(f"{'='*50}\n")

    # 1. Core files
    print("[Core Files]")
    total += 1; score += check("CLAUDE.md exists and has content",
        (p / "CLAUDE.md").exists() and (p / "CLAUDE.md").stat().st_size > 100)
    total += 1; score += check("CURRENT_TASK.md exists",
        (p / "CURRENT_TASK.md").exists())
    total += 1; score += check("EVOLUTION_LOG.md exists",
        (p / "EVOLUTION_LOG.md").exists())
    total += 1; score += check("knowledge/INDEX.md exists",
        (p / "knowledge" / "INDEX.md").exists())

    # 2. CLAUDE.md quality
    print("\n[CLAUDE.md Quality]")
    claude_md = ""
    if (p / "CLAUDE.md").exists():
        claude_md = (p / "CLAUDE.md").read_text()
    total += 1; score += check("Has learned errors section",
        "error" in claude_md.lower() or "lesson" in claude_md.lower())
    total += 1; score += check("Has verification rules",
        "verif" in claude_md.lower())
    total += 1; score += check("Not too long (< 200 lines)",
        claude_md.count('\n') < 200)

    # 3. Knowledge base
    print("\n[Knowledge Base]")
    total += 1; score += check("knowledge/failures/ has entries",
        any((p / "knowledge" / "failures").glob("*.md")) if (p / "knowledge" / "failures").exists() else False)
    total += 1; score += check("knowledge/patterns/ has entries",
        any((p / "knowledge" / "patterns").glob("*.md")) if (p / "knowledge" / "patterns").exists() else False)
    total += 1; score += check("knowledge/decisions/ has entries",
        any((p / "knowledge" / "decisions").glob("*.md")) if (p / "knowledge" / "decisions").exists() else False)

    # 4. Skills
    print("\n[Skills]")
    skills_dir = p / ".claude" / "skills"
    if skills_dir.exists():
        skills = [d.name for d in skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]
        total += 1; score += check(f"Has skills ({len(skills)} found)",
            len(skills) >= 3)
        total += 1; score += check("Has verify skill",
            "verify" in skills)
        total += 1; score += check("Has auto-improve skill",
            "auto-improve" in skills)
        total += 1; score += check("Has autoresearch skill",
            "autoresearch" in skills)
    else:
        total += 4
        for name in ["skills exist", "verify", "auto-improve", "autoresearch"]:
            check(name, False)

    # 5. Agents
    print("\n[Agents]")
    agents_dir = p / ".claude" / "agents"
    if agents_dir.exists():
        agents = list(agents_dir.glob("*.md"))
        total += 1; score += check(f"Has agents ({len(agents)} found)",
            len(agents) >= 3)
    else:
        total += 1; check("agents exist", False)

    # 6. Hooks
    print("\n[Hooks]")
    settings_file = p / ".claude" / "settings.json"
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
            hooks = settings.get("hooks", {})
            total += 1; score += check("SessionStart hook",
                "SessionStart" in hooks)
            total += 1; score += check("PostCompact hook",
                "PostCompact" in hooks)
            total += 1; score += check("Stop hook",
                "Stop" in hooks)
            total += 1; score += check("PostToolUse hook for Edit|Write",
                "PostToolUse" in hooks)
        except json.JSONDecodeError:
            total += 4
            check("settings.json valid JSON", False)
    else:
        total += 4
        check("settings.json exists", False)

    # 7. Commands
    print("\n[Commands]")
    cmds_dir = p / ".claude" / "commands"
    if cmds_dir.exists():
        cmds = list(cmds_dir.glob("*.md"))
        total += 1; score += check(f"Has slash commands ({len(cmds)} found)",
            len(cmds) >= 5)
    else:
        total += 1; check("commands exist", False)

    # 8. Research freshness
    print("\n[Research]")
    research_dir = p / "knowledge" / "research"
    if research_dir.exists():
        research_files = sorted(research_dir.glob("weekly_*.md"), reverse=True)
        total += 1; score += check("Has weekly research entries",
            len(research_files) > 0)
    else:
        total += 1; check("research directory exists", False)

    # Final score
    pct = (score / total * 100) if total > 0 else 0
    grade = score / total * 10 if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"Score: {score}/{total} ({pct:.0f}%) — Grade: {grade:.1f}/10")
    print(f"{'='*50}\n")

    return grade

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    grade = evaluate(path)
    sys.exit(0 if grade >= 6 else 1)
