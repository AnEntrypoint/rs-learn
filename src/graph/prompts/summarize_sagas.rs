use super::snippets::MAX_SUMMARY_CHARS;
use super::Prompt;

pub struct SagaCtx<'a> {
    pub saga_name: &'a str,
    pub existing_summary: &'a str,
    pub episodes: &'a [String],
}

pub fn summarize_saga(ctx: &SagaCtx) -> Prompt {
    let name = if ctx.saga_name.is_empty() { "Unknown" } else { ctx.saga_name };
    let episodes_text = if ctx.episodes.is_empty() {
        "(no messages)".to_string()
    } else {
        ctx.episodes.join("\n---\n")
    };
    let existing_section = if ctx.existing_summary.is_empty() {
        String::new()
    } else {
        format!(
            "\nPrevious summary of this conversation thread:\n{}\n\nThe following messages may include new content since the previous summary. Update the summary to incorporate any new information.\n",
            ctx.existing_summary
        )
    };
    let sys = format!(
        "You are a helpful assistant that summarizes conversation threads. Produce a single dense factual summary of the conversation. Keep the summary under {max} characters. State facts directly. Do not use filler verbs like \"mentioned\", \"discussed\", \"noted\", or \"stated\". Preserve names, dates, decisions, and outcomes. Begin with the main topic or outcome, not with \"This conversation\" or \"The thread\".",
        max = MAX_SUMMARY_CHARS
    );
    let user = format!(
        "Summarize the following conversation thread \"{name}\":
{existing}
Messages:
{episodes}

Return JSON: {{\"summary\":\"string\"}}",
        name = name,
        existing = existing_section,
        episodes = episodes_text,
    );
    Prompt { system: sys, user, schema: "summary" }
}
