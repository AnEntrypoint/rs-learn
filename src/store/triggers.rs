pub const TRIGGERS: &[&str] = &[
    "CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
        INSERT INTO nodes_fts(id,name,summary) VALUES(new.id,new.name,COALESCE(new.summary,''));
    END",
    "CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
        DELETE FROM nodes_fts WHERE id=old.id;
    END",
    "CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
        DELETE FROM nodes_fts WHERE id=old.id;
        INSERT INTO nodes_fts(id,name,summary) VALUES(new.id,new.name,COALESCE(new.summary,''));
    END",
    "CREATE TRIGGER IF NOT EXISTS edges_ai AFTER INSERT ON edges BEGIN
        INSERT INTO edges_fts(id,fact) VALUES(new.id,COALESCE(new.fact,''));
    END",
    "CREATE TRIGGER IF NOT EXISTS edges_ad AFTER DELETE ON edges BEGIN
        DELETE FROM edges_fts WHERE id=old.id;
    END",
    "CREATE TRIGGER IF NOT EXISTS edges_au AFTER UPDATE ON edges BEGIN
        DELETE FROM edges_fts WHERE id=old.id;
        INSERT INTO edges_fts(id,fact) VALUES(new.id,COALESCE(new.fact,''));
    END",
    "CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
        INSERT INTO episodes_fts(id,content) VALUES(new.id,new.content);
    END",
    "CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
        DELETE FROM episodes_fts WHERE id=old.id;
    END",
    "CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
        DELETE FROM episodes_fts WHERE id=old.id;
        INSERT INTO episodes_fts(id,content) VALUES(new.id,new.content);
    END",
    "CREATE TRIGGER IF NOT EXISTS reasoning_ai AFTER INSERT ON reasoning_bank BEGIN
        INSERT INTO reasoning_fts(id,strategy) VALUES(new.id,new.strategy);
    END",
    "CREATE TRIGGER IF NOT EXISTS reasoning_ad AFTER DELETE ON reasoning_bank BEGIN
        DELETE FROM reasoning_fts WHERE id=old.id;
    END",
    "CREATE TRIGGER IF NOT EXISTS reasoning_au AFTER UPDATE ON reasoning_bank BEGIN
        DELETE FROM reasoning_fts WHERE id=old.id;
        INSERT INTO reasoning_fts(id,strategy) VALUES(new.id,new.strategy);
    END",
    "CREATE TRIGGER IF NOT EXISTS communities_ai AFTER INSERT ON communities BEGIN
        INSERT INTO communities_fts(id,name,summary) VALUES(new.id,new.name,COALESCE(new.summary,''));
    END",
    "CREATE TRIGGER IF NOT EXISTS communities_ad AFTER DELETE ON communities BEGIN
        DELETE FROM communities_fts WHERE id=old.id;
    END",
    "CREATE TRIGGER IF NOT EXISTS communities_au AFTER UPDATE ON communities BEGIN
        DELETE FROM communities_fts WHERE id=old.id;
        INSERT INTO communities_fts(id,name,summary) VALUES(new.id,new.name,COALESCE(new.summary,''));
    END",
];
