#!/usr/bin/env bash
# PostToolUse (Edit|MultiEdit|Write): R4 and R6, checked on the file that was just written.
# Exit 2 hands the message back to the agent, which must undo the suppression or skip.

. "$(dirname "$0")/lib.sh"

rel=$(repo_relative "$(json_field file_path)")
[ -z "$rel" ] && exit 0
starts_with_any "$rel" "${SOURCE_PATHS[@]}" "${TEST_PATHS[@]}" || exit 0

base=$(base_rev)
problems=""

check() {
    local rule=$1 label=$2 regex=$3 before after
    [ -z "$regex" ] && return
    before=$(count_in_file "$rel" "$regex" "$base")
    after=$(count_in_file "$rel" "$regex" worktree)
    if [ "${after:-0}" -gt "${before:-0}" ]; then
        problems+="AI_RULES $rule: $rel now has $after $label (base: $before)."$'\n'
        problems+=$(grep -nE -- "$regex" "$ROOT/$rel" | head -5 | sed 's/^/    line /')$'\n'
    fi
}

check R4 "suppression(s)" "$SUPPRESSION_RE"
check R6 "skipped or focused test(s)" "$SKIP_RE"

if [ -n "$problems" ]; then
    {
        printf '%s' "$problems"
        echo "Remove it and restructure the code instead (R1). If you cannot, stop and report"
        echo "which rule blocks you and the refactoring you propose."
    } >&2
    exit 2
fi
exit 0
