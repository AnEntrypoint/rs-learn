use super::search::{Reranker, SearchConfig};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scope {
    Nodes,
    Edges,
    Communities,
    Episodes,
    Combined,
}

pub struct Recipe {
    pub name: &'static str,
    pub scope: Scope,
    pub cfg: SearchConfig,
}

fn base(r: Reranker) -> SearchConfig {
    SearchConfig { reranker: r, ..SearchConfig::default() }
}

pub fn all() -> Vec<Recipe> {
    vec![
        Recipe { name: "NODE_HYBRID_SEARCH_RRF", scope: Scope::Nodes, cfg: base(Reranker::Rrf) },
        Recipe { name: "NODE_HYBRID_SEARCH_MMR", scope: Scope::Nodes, cfg: base(Reranker::Mmr) },
        Recipe { name: "NODE_HYBRID_SEARCH_NODE_DISTANCE", scope: Scope::Nodes, cfg: base(Reranker::NodeDistance) },
        Recipe { name: "NODE_HYBRID_SEARCH_EPISODE_MENTIONS", scope: Scope::Nodes, cfg: base(Reranker::EpisodeMentions) },
        Recipe { name: "NODE_HYBRID_SEARCH_CROSS_ENCODER", scope: Scope::Nodes, cfg: base(Reranker::CrossEncoder) },
        Recipe { name: "EDGE_HYBRID_SEARCH_RRF", scope: Scope::Edges, cfg: base(Reranker::Rrf) },
        Recipe { name: "EDGE_HYBRID_SEARCH_MMR", scope: Scope::Edges, cfg: base(Reranker::Mmr) },
        Recipe { name: "EDGE_HYBRID_SEARCH_NODE_DISTANCE", scope: Scope::Edges, cfg: base(Reranker::NodeDistance) },
        Recipe { name: "EDGE_HYBRID_SEARCH_EPISODE_MENTIONS", scope: Scope::Edges, cfg: base(Reranker::EpisodeMentions) },
        Recipe { name: "EDGE_HYBRID_SEARCH_CROSS_ENCODER", scope: Scope::Edges, cfg: base(Reranker::CrossEncoder) },
        Recipe { name: "COMMUNITY_HYBRID_SEARCH_RRF", scope: Scope::Communities, cfg: SearchConfig { limit: 3, ..base(Reranker::Rrf) } },
        Recipe { name: "COMMUNITY_HYBRID_SEARCH_MMR", scope: Scope::Communities, cfg: SearchConfig { limit: 3, ..base(Reranker::Mmr) } },
        Recipe { name: "COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER", scope: Scope::Communities, cfg: SearchConfig { limit: 3, ..base(Reranker::CrossEncoder) } },
        Recipe { name: "COMBINED_HYBRID_SEARCH_RRF", scope: Scope::Combined, cfg: base(Reranker::Rrf) },
        Recipe { name: "COMBINED_HYBRID_SEARCH_MMR", scope: Scope::Combined, cfg: base(Reranker::Mmr) },
        Recipe { name: "COMBINED_HYBRID_SEARCH_CROSS_ENCODER", scope: Scope::Combined, cfg: base(Reranker::CrossEncoder) },
        Recipe { name: "EPISODE_HYBRID_SEARCH_RRF", scope: Scope::Episodes, cfg: base(Reranker::Rrf) },
    ]
}

pub fn by_name(name: &str) -> Option<Recipe> {
    all().into_iter().find(|r| r.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_count_matches_graphiti() {
        let names: Vec<&'static str> = all().iter().map(|r| r.name).collect();
        assert_eq!(names.len(), 17);
        assert!(names.contains(&"NODE_HYBRID_SEARCH_RRF"));
        assert!(names.contains(&"COMBINED_HYBRID_SEARCH_MMR"));
    }

    #[test]
    fn lookup_by_name() {
        let r = by_name("EDGE_HYBRID_SEARCH_NODE_DISTANCE").unwrap();
        assert_eq!(r.scope, Scope::Edges);
        assert_eq!(r.cfg.reranker, Reranker::NodeDistance);
    }
}
