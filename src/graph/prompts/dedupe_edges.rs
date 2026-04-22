use super::{to_prompt_json, Prompt};
use serde_json::Value;

const SYS: &str = "You are a fact deduplication assistant. NEVER mark facts with key differences as duplicates.";

pub struct ResolveEdgeCtx<'a> {
    pub existing_edges: &'a Value,
    pub edge_invalidation_candidates: &'a Value,
    pub new_edge: &'a Value,
}

pub fn resolve_edge(ctx: &ResolveEdgeCtx) -> Prompt {
    let user = format!(r#"
NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
{existing}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{inval}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{new_edge}
</NEW FACT>

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Determine which facts the NEW FACT contradicts from either list.
   - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
   - Return all contradicted idx values in contradicted_facts.
   - If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: idx=0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: duplicate_facts=[0], contradicted_facts=[] (identical factual information)

EXISTING FACT: idx=1, "Alice works at Acme Corp as a software engineer"
NEW FACT: "Alice works at Acme Corp as a senior engineer"
Result: duplicate_facts=[], contradicted_facts=[1] (same relationship but updated title — contradiction, NOT a duplicate)

EXISTING FACT: idx=2, "Bob ran 5 miles on Tuesday"
NEW FACT: "Bob ran 3 miles on Wednesday"
Result: duplicate_facts=[], contradicted_facts=[] (different events on different days — neither duplicate nor contradiction)
</EXAMPLE>

Return JSON: {{"duplicate_facts":[],"contradicted_facts":[]}}
"#,
        existing = to_prompt_json(ctx.existing_edges),
        inval = to_prompt_json(ctx.edge_invalidation_candidates),
        new_edge = to_prompt_json(ctx.new_edge),
    );
    Prompt { system: SYS.into(), user, schema: "resolve_edge" }
}
