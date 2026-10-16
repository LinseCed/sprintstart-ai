#!/usr/bin/env bash
# Mutation gate (AI_RULES.md R25): mutates the source files a branch changed and fails when the tests
# kill fewer than MUTATION_MIN percent of the mutants. A test that stays green while the code under it
# is broken does not test it — this is the check that catches that.
#
#   bash scripts/quality/mutation.sh [base-rev]   # the files a branch changed; fails below MUTATION_MIN
#   bash scripts/quality/mutation.sh --all        # every source file; reports the score, never fails
#
# Identical in all SprintStart repos. ratchet.conf provides MUTATION_FILES (pathspecs of mutable
# source files), MUTATION_MIN, run_mutation <files...> and mutation_counts (prints "killed total").

set -uo pipefail

ROOT=$(git rev-parse --show-toplevel) || exit 2
cd "$ROOT" || exit 2
. "$ROOT/scripts/quality/ratchet.conf"

all=0
if [ "${1:-}" = "--all" ]; then
    all=1
    shift
fi

base_ref=${1:-}
if [ -z "$base_ref" ]; then
    for ref in origin/dev origin/main; do
        git rev-parse -q --verify "$ref" >/dev/null && { base_ref=$ref; break; }
    done
fi
BASE=$(git merge-base HEAD "$base_ref")

if [ "$all" = 1 ]; then
    mapfile -t files < <(git ls-files -- "${MUTATION_FILES[@]}")
else
    mapfile -t files < <({ git diff --name-only --diff-filter=d "$BASE" -- "${MUTATION_FILES[@]}" 2>/dev/null
        git ls-files --others --exclude-standard -- "${MUTATION_FILES[@]}"; } | sort -u)
fi

if [ ${#files[@]} -eq 0 ]; then
    echo "mutation: no changed source files against $base_ref"
    exit 0
fi

echo "mutation: ${#files[@]} file(s)"
[ "$all" = 1 ] || printf '    %s\n' "${files[@]}"
run_mutation "${files[@]}" || { echo "mutation: the mutation run itself failed"; exit 2; }

read -r killed total < <(mutation_counts)
if [ "${total:-0}" -eq 0 ]; then
    echo "mutation: no mutants were generated"
    exit 0
fi

score=$((100 * killed / total))
echo "mutation: $killed of $total mutants killed = $score% (minimum $MUTATION_MIN%)"
if [ "$all" = 0 ] && [ "$score" -lt "$MUTATION_MIN" ]; then
    echo "mutation: the tests do not notice enough broken code; strengthen their assertions (R18, R25)"
    exit 1
fi
